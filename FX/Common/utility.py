import pandas as pd
import numpy as np
from scipy.stats import norm
import datetime as dt
import matplotlib.pyplot as plt
from typing import List
from scipy import interpolate
from matplotlib import cm
from scipy import optimize
from scipy.integrate import quad
from scipy.optimize import minimize


"""
import data and check for any NaNs
"""
def option_quotes() -> pd.DataFrame:
    path = "/Users/jivitesh/Desktop/FX & Equity Vols/FX/Resources/Data/Vol_Data.xlsx"
    xlsx = pd.ExcelFile(path)
    df = pd.read_excel(xlsx, 'EURUSD vol surface')
    df = df[1:]
    if df.isnull().any().sum() > 0:
        raise Exception("Data issue")
    return df
###############################################################

""" 
clean up data -> OTM only, remove 0 Bids, Add term, Add Mids
"""
def refactor_df(df: pd.DataFrame) -> pd.DataFrame:
    df['10_Put_Vol'] = (df['ATM'] + df['10D BF'] - 0.5 * df['10D RR']) * 0.01
    df['25_Put_Vol'] = (df['ATM'] + df['25D BF'] - 0.5 * df['25D RR']) * 0.01
    df['ATM_vol'] = df['ATM'] * 0.01
    df['25_Call_Vol'] = (df['ATM'] + df['25D BF'] + 0.5*df['25D RR'])*0.01
    df['10_Call_Vol'] = (df['ATM'] + df['10D BF'] + 0.5*df['10D RR'])*0.01

    df.rename(columns={df.columns[0]: "Maturity"}, inplace=True)
    # Term mapping
    maturity_to_term = {"D": 1, "W": 7, "M": 30, "Y": 365}
    df['Term'] = df['Maturity'].apply(lambda x:  int(x[:-1]) * maturity_to_term[x[-1]])/365

    df['10_Put_Strike'] = 0.0
    df['25_Put_Strike'] = 0.0
    df['ATM_Strike'] = 0.0
    df['25_Call_Strike'] = 0.0
    df['10_Call_Strike'] = 0.0
    df.reset_index(drop=True, inplace=True)

    return df
###############################################################

"""BS delta with forward"""
def implied_delta(type, fwd, strike, vol, expiry) -> float:
    d1 = (np.log(fwd/strike) + 0.5 * expiry * vol**2)/(vol * np.sqrt(expiry))
    if type == "Call":
        return norm.cdf(d1)
    else:
        return -norm.cdf(-d1)
###############################################################

"""imply strike from Delta"""
def implied_strike(Type, spot, rate_d, rate_f, vol, expiry, delta) -> float:
    fwd = spot * np.exp((rate_d-rate_f)*expiry)
    delta = - delta if Type == 'Put' else delta
    def solve_k(strike):
        return implied_delta(Type, fwd, strike, vol, expiry) - delta
    strike = optimize.brentq(solve_k, 0.01, 100)
    return strike
###############################################################


""" 
Black Call with Fwd
"""
def BS_Call(F: float, S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1 = (np.log(F / K) + ((sigma ** 2) * 0.5) * T) / (sigma * np.sqrt(T))  # F=S => r_dom = r_for = o
    d2 = d1 - (sigma * np.sqrt(T))
    return (F * norm.cdf(d1) - K * norm.cdf(d2)) * np.exp(-r * T)  # F=S => r_dom = r_for = o

###############################################################


"""refactor dataframe"""
def flatten_df(df: pd.DataFrame) -> pd.DataFrame:
    df_new = pd.DataFrame()
    for index, row in df.iterrows():
        strikes = row[['10_Put_Strike', '25_Put_Strike', 'ATM_Strike', '25_Call_Strike', '10_Call_Strike']]
        vols = row[['10_Put_Vol', '25_Put_Vol', 'ATM_vol', '25_Call_Vol', '10_Call_Vol']]
        deltas = [0.1, 0.25, 0.5, 0.75, 0.9]
        for i in range(5):
            df_new.at[index * 5 + i, 'Term'] = df.at[index, 'Term']
            df_new.at[index * 5 + i, 'Delta'] = deltas[i]
            df_new.at[index * 5 + i, 'Strike'] = strikes[i]
            df_new.at[index * 5 + i, 'Vol'] = vols[i]

    return df_new
###############################################################

""" 
plot scatter vol surface
reference: https://matplotlib.org/stable/gallery/mplot3d/scatter3d.html
"""
def plot_scattered_surface(df: pd.DataFrame) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(df['Delta'], df['Term'], df['Vol'], marker=".")

    ax.set_xlabel('Delta', color="red")
    ax.set_ylabel('Term', color="red")
    ax.set_zlabel('Vol', color="red")
    plt.title(label="Implied Vols", fontsize=20, color="black")
    plt.show()
###############################################################


"""
plot interpolated surface
"""
def plot_continuous_surface(df_surface:pd.DataFrame, label, y_axis='Delta') -> None:

    X=df_surface[y_axis]
    Y=df_surface['Term']
    Z=df_surface['Vol']
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_trisurf(X, Y, Z, cmap=cm.jet,linewidth=0.2,antialiased=True,shade=True)   #cmap=cm.coolwarm,jet
    ax.set_xlabel(y_axis, color="red")
    ax.set_ylabel('Term', color="red")
    ax.set_zlabel('Vol', color="red")
    plt.title(label=label +' Surface', fontsize=20, color="black")
    plt.show()
###############################################################


"""
Interpolate/Extrapolate surface and return smooth surface using methodologies:
1) Cubic
2) SABR
other options: thin_plate, multiquadric 
for spline, interpolation in Time is linear in variance
"""
class smooth_surface:
    def __init__(self, raw_surface: pd.DataFrame, method: str, delta_min: float,
                 delta_max: float, strike_delta: float, interp_terms: List[float]):
        self.data = raw_surface
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.strike_delta = strike_delta
        self.interp_terms = interp_terms
        if method == 'spline':
            self.surface_fx = self.spline()
        elif method == 'sabr':
            pass
        else:
            raise Exception("Method not supported")

    def get_surface(self) -> pd.DataFrame:
        return self.surface()

    # to be used pricing and calibrating. Assumption: linear interpolation in variance terms over surface
    def get_vol(self, t: float, k: float) -> float:
        """" t: term
            k: strike"""
        f = self.surface_fx
        return np.sqrt(f(t, k)/t)

    def spline(self) -> (np.ndarray, np.ndarray):
        terms = self.data['Term'].unique()
        # terms = np.round(terms, 6)
        strike_surface = pd.DataFrame(columns=['Term', 'Delta', 'Vol'])
        # Spatial interpolation and extrapolation
        for term in terms:
            df = self.data.loc[self.data['Term'] == term]
            # interp = interpolate.CubicSpline(df['Strike'], df['Vol'], bc_type="natural", extrapolate=False)
            interp = interpolate.CubicSpline(df['Delta'], df['Vol'], bc_type="natural", extrapolate=False)
            # extrap = interpolate.Rbf(df['Strike'], df['Vol'], function='multiquadric', extrapolate=True)
            def extrap(grid: np.ndarray):   # flat extrap in strike
                output = np.zeros(shape=grid.shape)
                i = 0
                for point in grid:
                    if point < min(df.Delta):
                        output[i] = df.Vol.iloc[0]
                    elif point > max(df.Delta):
                        output[i] = df.Vol.iloc[-1]
                    i += 1
                return output

            strike_grid = np.arange(self.delta_min, self.delta_max + self.strike_delta,
                                    self.strike_delta)  # spatial grid
            vol_grid = np.concatenate(
                (extrap(np.arange(self.delta_min, max(self.delta_min, min(df['Delta'])), self.strike_delta)),
                 interp(np.arange(max(min(df['Delta']), self.delta_min), min(max(df['Delta']), self.delta_max) +
                                  self.strike_delta, self.strike_delta).round(5)),
                 extrap(np.arange(max(df['Delta']) + self.strike_delta, self.delta_max + 0.001, self.strike_delta))))

            strike_surface_term = pd.DataFrame(data=[np.full(strike_grid.shape, term), strike_grid, vol_grid]).T
            strike_surface_term.columns = ['Term', 'Delta', 'Vol']
            strike_surface = pd.concat([strike_surface, strike_surface_term])
            del interp, extrap, vol_grid, df

        strike_surface = strike_surface.fillna(method='ffill')
        strike_surface.drop_duplicates(keep='first', inplace=True)
        strike_surface.reset_index(drop=True, inplace=True)

        # Temporal interpolation => linear in variance
        x = np.array(terms)  # terms
        y = np.arange(self.delta_min, self.delta_max + self.strike_delta, self.strike_delta)  # Strikes
        z = np.ndarray(0)
        for term in terms:
            df = strike_surface.loc[strike_surface['Term'] == term]
            df.loc[:, ['Var_T']] = (df['Vol'] ** 2) * df['Term']  # Variance
            z = np.concatenate((z, df['Var_T'].to_numpy()))
            del df

        # variance for interpolated terms
        f = interpolate.interp2d(x, y, z, kind='linear')
        return f

    def surface(self):
        f = self.surface_fx
        terms = self.data['Term'].unique()
        XX = np.asarray(self.interp_terms)
        YY = np.arange(self.delta_min, self.delta_max + self.strike_delta, self.strike_delta)
        X, Y = np.meshgrid(XX, YY)
        Z = np.zeros((XX.shape[0], YY.shape[0]))
        for i in range(XX.shape[0]):
            for j in range(YY.shape[0]):
                Z[i, j] = np.sqrt(f(X[j, i], Y[j, i])/ X[j, i])

        # compile surface
        final_surface = pd.DataFrame(columns=['Term', 'Delta', 'Vol'])
        i = 0
        for maturity in XX:
            final_surface_maturity = pd.DataFrame(data=[np.full(YY.shape, maturity), YY, Z[i]]).T
            final_surface_maturity.columns = ['Term', 'Delta', 'Vol']
            final_surface = pd.concat([final_surface, final_surface_maturity])
            i += 1
        final_surface.reset_index(drop=True, inplace=True)
        return final_surface

    def sabr(self):
        #todo
        pass
###############################################################


"""
Dupire local vol implementation by Gatheral J.
Ref: https://www.amazon.com/Volatility-Surface-Practitioners-Guide/dp/0471792519
Red
Variance form preferred as denominator (dc2/dk2) tends to 0 in Volatility matrix terms.
Important to check Butterfly arbitrage to avoid negative instantaneous variance.
Using Finite derivatives method
"""
class Dupire_local:
    def __init__(self, implied_surface_obj: smooth_surface, spot: float, domestic_r: float, foreign_r: float, method:str):
        self.spot = spot
        self.iv_fx = implied_surface_obj
        self.r = domestic_r
        self.q = foreign_r
        if method == 'lv_from_iv':
            self.surface_fx = self.lv_fom_iv
        elif method == 'lv_fom_call':
            pass
        elif method == 'lv_fom_var':
            pass
        else:
            raise Exception("Method not supported")

    def get_surface(self) -> pd.DataFrame:
        return self.surface()

    def lv_fom_iv(self, t: float, k: float) -> float:
        # calculate all derivatives i.e dv/dt, dv/dk, d2v/dk2
        dvol_dt = (self.iv_fx.get_vol(t+0.1, k) - self.iv_fx.get_vol(t, k))/0.1
        if k >= self.iv_fx.delta_max - 0.01:
            dvol_dk = (self.iv_fx.get_vol(t, k) - self.iv_fx.get_vol(t, k-2*0.01))/(2*0.01)
            d2vol_dk2 = (self.iv_fx.get_vol(t, k) - 2*self.iv_fx.get_vol(t, k-0.01) + self.iv_fx.get_vol(t, k-2*0.01))/(0.01**2)
        elif k <= self.iv_fx.delta_min + 0.01:
            dvol_dk = (self.iv_fx.get_vol(t, k+2*0.01) - self.iv_fx.get_vol(t, k))/(2*0.01)
            d2vol_dk2 = (self.iv_fx.get_vol(t, k+2*0.01) - 2*self.iv_fx.get_vol(t, k+0.01) + self.iv_fx.get_vol(t, k))/(0.01**2)
        else:
            dvol_dk = (self.iv_fx.get_vol(t, k+0.01) - self.iv_fx.get_vol(t, k-0.01))/(2*0.01)
            d2vol_dk2 = (self.iv_fx.get_vol(t, k+0.01) - 2*self.iv_fx.get_vol(t, k) + self.iv_fx.get_vol(t, k-0.01))/(0.01**2)

        iv = self.iv_fx.get_vol(t, k)    # implied vol from surface

        d1 = (np.log(self.spot/k) + ((self.r-self.q) + 0.5*(iv**2))*t)/(iv*np.sqrt(t))
        numerator = iv**2 + 2*iv*t*dvol_dt + 2*(self.r-self.q)*iv*k*t*dvol_dk
        denominator1 = (1 + k*d1*np.sqrt(t)*dvol_dk)**2
        denominator2 = iv*t*(k**2)*(d2vol_dk2 - d1*(dvol_dk**2)*np.sqrt(t))

        return np.sqrt(min(max(numerator / (denominator1 + denominator2), 1e-6), 1))

    def lv_from_var(self, t: float, k: float) -> float:
        # calculate all derivatives of varaince i.e dw/dt, dw/dk, d2w/dk2.  Variance = w**2 * t
        dw_dt = (self.iv_fx.get_vol(t + 0.1, k)**2 * t - self.iv_fx.get_vol(t, k)**2 * t) / 0.1

    def surface(self):
        f = self.surface_fx
        terms = self.iv_fx.data['Term'].unique()
        # XX = np.arange(terms[0], terms[-1], self.iv_fx.time_delta)  # temporal grid
        XX = np.asarray(self.iv_fx.interp_terms)
        XX[-1] = XX[-1] - 0.1  # else dt derivative is 0 for flat
        YY = np.arange(self.iv_fx.delta_min, self.iv_fx.delta_max + self.iv_fx.strike_delta, self.iv_fx.strike_delta) # spatial
        X, Y = np.meshgrid(XX, YY)
        Z = np.zeros((XX.shape[0], YY.shape[0]))
        for i in range(XX.shape[0]):
            for j in range(YY.shape[0]):
                Z[i, j] = f(X[j, i], Y[j, i])

        # compile surface
        final_surface = pd.DataFrame(columns=['Term', 'Delta', 'Vol'])
        i = 0
        for maturity in XX:
            final_surface_maturity = pd.DataFrame(data=[np.full(YY.shape, maturity), YY, Z[i]]).T
            final_surface_maturity.columns = ['Term', 'Delta', 'Vol']
            final_surface = pd.concat([final_surface, final_surface_maturity])
            i += 1
        final_surface.reset_index(drop=True, inplace=True)

        print("manually removing ",  len(final_surface.loc[final_surface['Vol'] > 0.5]), " points")
        return final_surface.loc[final_surface['Vol'] < 0.5]
###############################################################


"""
Arbitrage Checks
"""
class arbitrage_checks:
    def init(self, surface: smooth_surface):
        self.fx = surface

    def call_spread(self, k1, k2, t, S) -> bool:   # increasing with strike
        sigma1 = self.fx.get_vol(t, k1)
        sigma2 = self.fx.get_vol(t, k2)
        call1 = BS_Call(S, S, k1, t, 0.0, sigma1)
        call2 = BS_Call(S, S, k2, t, 0.0, sigma2)
        if call1 < call2:
            return False
        return True

    def convexity_spread(self, k1, k2, k3, t, S) -> bool:   # increasing with strike
        sigma1 = self.fx.get_vol(t, k1)
        sigma2 = self.fx.get_vol(t, k2)
        sigma3 = self.fx.get_vol(t, k3)
        call1 = BS_Call(S, S, k1, t, 0.0, sigma1)
        call2 = BS_Call(S, S, k2, t, 0.0, sigma2)
        call3 = BS_Call(S, S, k3, t, 0.0, sigma3)
        if (call1 - 2*call2 + call3) < 0:
            return False
        return True

    def calendar_spread(self, k, t1, t2, S) -> bool:   # increasing with time
        sigma1 = self.fx.get_vol(t1, k)
        sigma2 = self.fx.get_vol(t2, k)
        call1 = BS_Call(S, S, k, t1, 0.0, sigma1)
        call2 = BS_Call(S, S, k, t2, 0.0, sigma2)
        if call1 > call2:
            return False
        return True
###############################################################


""" 
imply BS Vols with Forward
using Newton Raphson algo for root finding.
Vega = BS closed form i.e. S*sqrt(T)*PDF(d1)
"""
def implied_vol_point(F: 'forward', S: 'Spot', K: 'Strike', r: 'rate',
                      T: 'Term', Type: "Call/Put", Price: "Target") -> 'vol':
    def vega(S: float, F: float, K: float,   # F=S
             T: float, r: float, sigma: float) -> float:
        d1 = (np.log(F/K) + ((sigma**2)*0.5)*T)/(sigma * np.sqrt(T))  # F=S => r_dom = r_for = o
        return S * np.sqrt(T) * norm.pdf(d1)   # r_dom = 0, yield_df = 1


    sigma = 0.25  # initial guess
    # sigma = np.sqrt(2*np.pi/T)*Price/S
    tolerance = 0.001
    max_iterations = 100
    if F == None:
        F = S     # 0 rates
    r = 0.0
    for i in range(max_iterations):
        cost_fx = BS_Call(F, S, K, T, r, sigma) - Price

        if abs(cost_fx) < tolerance:
            break
        # next iteration
        vega_val = vega(S, F, K, T, r, sigma)
        if vega_val < 1e-6:
            return 0.0
        sigma = sigma - cost_fx/vega_val
        if i == max_iterations - 1:
            print(" Max iteration reached, target diff: ", cost_fx,)
    return sigma
###############################################################


"""
Heston Parametric implementation
Borrow code partially from Git project (objective fx)
"""
class Heston_parametric:
    def __init__(self, S0, r, v0, k, theta, rho, eta):
        self.S0 = S0
        self.v0 = v0
        self.r = r
        self.k = k
        self.theta = theta
        self.eta = eta
        self.rho = rho
        self.gamma = eta ** 2 / 2.0

    def integrand(self, u, j, x, v, tau):
        return np.real(np.exp(self.C(j, u, tau) * self.theta + self.D(j, u, tau) * v + 1j * u * x) / (u * 1j))

    def P(self, j, x, tau):
        return 0.5 + 1.0 / np.pi * (quad(self.integrand, 0.0, np.inf, args=(j, x, self.v0, tau)))[0]

    def C(self, j, u, tau):
        g = self.rminus(j, u) / self.rplus(j, u)
        return self.k * (self.rminus(j, u) * tau - 2.0 / self.eta ** 2 * np.log(
            (1.0 - g * np.exp(-self.d(j, u) * tau)) / (1.0 - g)))

    def d(self, j, u):
        return np.sqrt(self.beta(j, u) ** 2 - 4 * self.alpha(j, u) * self.gamma)

    def rminus(self, j, u):
        return (self.beta(j, u) - self.d(j, u)) / (2 * self.gamma)

    def rplus(self, j, u):
        return (self.beta(j, u) + self.d(j, u)) / (2 * self.gamma)

    def beta(self, j, u):
        return self.k - self.rho * self.eta * j - self.rho * self.eta * u * 1j

    def alpha(self, j, u):
        return -u ** 2 / 2 - u * 1j / 2 + j * u * 1j

    def D(self, j, u, tau):
        g = self.rminus(j, u) / self.rplus(j, u)
        return self.rminus(j, u) * (1.0 - np.exp(-self.d(j, u) * tau)) / (1.0 - g * np.exp(-self.d(j, u) * tau))

    def price(self, strike, tau):   # call price, use put call parity for put price
        B = np.exp(-self.r * tau)
        F = self.S0 / B
        x = np.log(F / strike)
        return B * (F * self.P(1, x, tau) - strike * self.P(0, x, tau))
###############################################################

"""
Heston Model class to fit parameters
and to use parameters for pricing, risk
"""
class Heston_model:
    def __init__(self, S0, r, df_raw_surface, strike_min, strike_max):
        self.S0 = S0
        self.r = r
        self.data = df_raw_surface
        self.strike_min = strike_min
        self.strike_max = strike_max
        self.strike_delta = 0.1   # temporal grid
        self.time_delta = 0.5   # spatial grid

    """
    objective fx to be solved within bid ask vol spread or arbitrary spread around Mid vols
    """
    def Heston_obj_fx(self, params):
        df_heston_slice = self.slice_data.loc[(self.data.Strike >= self.strike_min) &
                                              (self.data.Strike <= self.strike_max)]
        strikes = df_heston_slice['Strike'].values
        iv_mids = df_heston_slice['Vol'].values
        v0, k, theta, rho, eta = [x for x in params]

        total_error = 0
        for i in range(len(strikes)):
            strike = strikes[i]
            heston_obj = Heston_parametric(self.S0,  self.r, v0, k, theta, rho, eta)
            heston_call = heston_obj.price(strike, self.term)
            heston_impvol = implied_vol_point(self.S0, self.S0, strike, self.r, self.term, "Call", heston_call)  #F=S

            if heston_impvol < iv_mids[i] * 0.995:  # 5% bid ask assumption for quick fit. Use Bid Vols otherwise
                error = (iv_mids[i] * 0.95 - heston_impvol) ** 2
            elif heston_impvol > iv_mids[i] * 1.005:  # 5% bid ask assumption for quick fit. Use Ask Vols otherwise
                error = (iv_mids[i] * 1.05 - heston_impvol) ** 2
            else:
                error = 0
            total_error += error
        return total_error

    def fit_quotes(self, use_saved, override_saved) -> None:
        use_saved = False if override_saved else use_saved
        params_path = "/Users/jivitesh/Desktop/FX & Equity Vols/FX/Resources/Data/paramters_saved.csv"
        if use_saved:
            self.params = pd.read_csv(params_path)
        if override_saved:
            all_fitted = []
            params_guess = [0.01, 0.6, 0.01, -0.2, 0.4]   # 'v0', 'k', 'theta', 'rho', 'eta'   9% mean assumption, near 0 corr FX, 0.35 vol of var, speed 0.8, initial variance (should be diff by term) but assuming
            params_bounds = [[0.001, 0.04], [0.1, 2], [0.001, 0.04], [-1, 1], [0.05, 1]]   # 'v0', 'k', 'theta', 'rho', 'eta'

            for term in self.data["Term"].unique():
                print(term)
                self.term = term
                self.slice_data = self.data.loc[self.data["Term"] == term]
                fitted = minimize(self.Heston_obj_fx, x0=params_guess, bounds=params_bounds, tol=1e-9)
                all_fitted.append(fitted.x)
                print(term, " done")
            all_fitted_df = pd.DataFrame(all_fitted)
            all_fitted_df.columns = ['v0', 'k', 'theta', 'rho', 'eta']
            all_fitted_df.to_csv(params_path, index=False)
            self.params = all_fitted_df

    """
    two options:
    1. interpolate calibrated surface
    2. interpolate parameters
    """
    def get_vol(self, time, strike):
        if not hasattr(self, 'interp_surface_fx'):  # to avoid reruns
            self.interp_vol()
        fx = self.interp_surface_fx
        return np.sqrt(fx(time, strike)/time)

    def get_surface(self):
        return self.surface()

    """
    implied BS vol from Heston call price
    with linear varinace interp in time
    """
    def interp_vol(self):
        terms = self.data["Term"].unique()
        strike_grid = np.arange(self.strike_min, self.strike_max + self.strike_delta, self.strike_delta)
        surface = pd.DataFrame(columns=['Term', 'Strike', 'Vol'])

        for index, row in self.params.iterrows():
            v0, k, theta, rho, eta = row  # 'v0', 'k', 'theta', 'rho', 'eta'
            vol_grid = np.zeros(strike_grid.shape)

            i=0
            for strike in strike_grid:
                heston_obj = Heston_parametric(self.S0, self.r, v0, k, theta, rho, eta)
                heston_call = heston_obj.price(strike, terms[index])
                heston_impvol = implied_vol_point(self.S0, self.S0, strike, self.r, terms[index], "Call", heston_call)  #F=S
                vol_grid[i] = heston_impvol
                i += 1

            surface_slice = pd.DataFrame(data=[np.full(strike_grid.shape, terms[index]), strike_grid, vol_grid]).T
            surface_slice.columns = ['Term', 'Strike', 'Vol']
            surface = pd.concat([surface, surface_slice])
        surface.reset_index(drop=True, inplace=True)

        surface.loc[:, ['Var_T']] = (surface['Vol'] ** 2) * surface['Term']  # Variance
        # Temporal interpolation => linear in variance
        x = np.array(terms)  # terms
        y = strike_grid
        z = np.ndarray(0)
        for term in terms:
            df = surface.loc[surface['Term'] == term]
            z = np.concatenate((z, df['Var_T'].to_numpy()))
            del df

        self.interp_surface_fx = interpolate.interp2d(x, y, z, kind='linear')

    def surface(self):

        terms = self.data["Term"].unique()
        XX = np.arange(terms[0], terms[-1] + self.time_delta, self.time_delta)  # temporal grid
        YY = np.arange(self.strike_min, self.strike_max + self.strike_delta, self.strike_delta)  # spatial grid
        X, Y = np.meshgrid(XX, YY)
        Z = np.zeros(X.shape)  # use either X or Y
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self.get_vol(X[i, j], Y[i, j])

        # compile surface
        final_surface = pd.DataFrame(columns=['Term', 'Strike', 'Vol'])
        i = 0
        for maturity in XX:
            final_surface_maturity = pd.DataFrame(data=[np.full(YY.shape, maturity), YY, Z.T[i]]).T
            final_surface_maturity.columns = ['Term', 'Strike', 'Vol']
            final_surface = pd.concat([final_surface, final_surface_maturity])
            final_surface.reset_index(drop=True, inplace=True)
            i += 1

        return final_surface
###############################################################


"""
down and in payoff
"""
def barrier_opt(barrier_type, strike, barrier, expiry, vol_surface, surface_type,
                spot, rate, iterations: int, steps: int):
    # iterations=100
    paths_obj = numerical_method(vol_surface, spot, rate)
    paths_obj_up = numerical_method(vol_surface, spot + 0.01, rate)  # same seed,1 cent move
    paths_obj_down = numerical_method(vol_surface, spot - 0.01, rate)  # same seed,1 cent move
    paths_obj_time = numerical_method(vol_surface, spot, rate)  # for theta
    paths_obj_vol = numerical_method(vol_surface, spot, rate)  # for Vega

    if surface_type == 'Dupire':
        paths = paths_obj.mc_simulations(expiry, iterations, steps, seed=31)
        paths_up = paths_obj_up.mc_simulations(expiry, iterations, steps, seed=31)
        paths_down = paths_obj_down.mc_simulations(expiry, iterations, steps, seed=31)
        paths_time = paths_obj_time.mc_simulations(expiry + 1/365, iterations, steps, seed=31)   # 1 day move
        paths_vol = paths_obj_vol.mc_simulations(expiry, iterations, steps, seed=31, vol_bump=0.01)
    else:   # surface_type == 'Heston':
        paths = paths_obj.mc_heston(expiry, iterations, steps, seed=31).T
        paths_up = paths_obj_up.mc_heston(expiry, iterations, steps, seed=31).T
        paths_down = paths_obj_down.mc_heston(expiry, iterations, steps, seed=31).T
        paths_time = paths_obj_time.mc_heston(expiry+1/365, iterations, steps, seed=31).T  # 1 day move
        paths_vol = paths_obj_vol.mc_heston(expiry, iterations, steps, seed=31, vol_bump=0.01).T

    paths_list = [paths_down, paths, paths_up, paths_time, paths_vol]
    Price_list = []  # Price, Price_up, Price_down
    for j in range(5):
        path_price = 0
        if barrier_type == 'Down&InPut':
            for i in range(iterations):
                if all(path > barrier for path in paths_list[j][i,:])==True:
                    path_price += 0
                else:
                    path_price += max(strike - paths_list[j][i, :][-1], 0) * np.exp(-rate * expiry)
            Price_list.append(path_price/iterations)
        else:
            print("Payoff not implemented")

    price = Price_list[1]
    delta = (Price_list[2] - Price_list[0])/(2*0.01)
    gamma = (Price_list[2] - 2*Price_list[1] + Price_list[0])/(0.01**2)
    theta = (Price_list[3] - Price_list[1])/(1)  # 1 day move
    vega = (Price_list[4] - Price_list[1])/(1)   # 1 vol parallel move

    return [price, delta, gamma, theta, vega]
###############################################################


"""
barrier analytic formula.
Ref: chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/http://homepage.ntu.edu.tw/~jryanwang/courses/
Financial%20Computation%20or%20Financial%20Engineering%20(graduate%20level)/FE_Ch08%20Barrier%20Option.pdf
Ref: Reiner and Rubinstein, Breaking Down the Barriers.
"""
def barrier_analytic(spot, strike, barrier, vol, expiry, rate=0.0,
                       div=0.0, barrier_type="Donwn&InCall", obs_type="continuous"):

    # Adjustment for Discrete monitoring. Reference: Broadie, Glasserman, and Kou
    total_observations = expiry * 365  # assuming daily, change to business day etc.
    barrier = barrier * np.exp(np.sign(barrier - spot) * 0.5826 * vol * np.sqrt(expiry / total_observations)) \
        if obs_type == "continuous" else barrier

    # Vanilla analytic parameters
    d1 = (np.log(spot / strike) + ((rate - div) + 0.5 * vol ** 2) * expiry) / (vol * np.sqrt(expiry))
    d2 = d1 - vol * np.sqrt(expiry)
    call_price = spot * np.exp(-div * expiry) * norm.cdf(d1) - strike * np.exp(-rate * expiry) * norm.cdf(d2)
    put_price = - spot * np.exp(-div * expiry) * norm.cdf(-d1) + strike * np.exp(-rate * expiry) * norm.cdf(-d2)

    # Barrier analytic parameters
    gamma_ = ((rate - div) + 0.5 * vol ** 2) / (vol ** 2)
    eta_ = np.log((barrier ** 2) / (spot * strike)) / (vol * np.sqrt(expiry)) + gamma_ * vol * np.sqrt(expiry)
    nu_ = np.log(spot / barrier) / (vol * np.sqrt(expiry)) + gamma_ * vol * np.sqrt(expiry)
    lambda_ = np.log(barrier / spot) / (vol * np.sqrt(expiry)) + gamma_ * vol * np.sqrt(expiry)

    if barrier_type == "Donwn&InCall":
        price = spot * np.exp(-div * expiry) * (barrier / spot) ** (2 * gamma_) * norm.cdf(eta_)- strike * \
                np.exp(-rate * expiry) * (barrier / spot) ** (2 * gamma_ - 2) * norm.cdf(eta_ - vol * np.sqrt(expiry))
        Down_OutCall = call_price - price
    elif barrier_type == "Donwn&InPut":
        price = - spot * np.exp(-div * expiry) * norm.cdf(-nu_) + strike * np.exp(-rate * expiry) * \
                norm.cdf(-nu_ + vol * np.sqrt(expiry)) + spot * np.exp(-div * expiry) * (barrier / spot) ** (2 * gamma_)\
                * (norm.cdf(eta_) - norm.cdf(lambda_)) - strike * np.exp(-rate * expiry) * (barrier / spot) ** \
                (2 * gamma_ - 2) * (norm.cdf(eta_ - vol * np.sqrt(expiry)) - norm.cdf(lambda_ - vol * np.sqrt(expiry)))
        Down_OutPut = put_price - price
    else:
        pass
        # todo: Up barrier implementation

    return price
###############################################################


"""  
Supports Monte Carlo for stock diffusion and Heston scheme

"""
class numerical_method:
    def __init__(self, vol_surface, spot, rate=0.0):
        self.spot = spot
        self.vols = vol_surface
        self.r = rate

    def mc_simulations(self, expiry: float, iterations: int, steps: int, seed: int, vol_bump: float = 0.0):  # Euler log process to preserve positivity
        dt = expiry/float(steps)
        paths_array = np.zeros((iterations, steps), np.float64)
        paths_array[:, 0] = self.spot
        np.random.seed(seed)
        rand_array = np.random.standard_normal(size=(iterations, steps-1))

        for i in range(iterations):
            for j in range(1, steps):
                # np.random.seed(10)
                rand = rand_array[i, j-1]
                vol = self.vols.lv_fom_iv(j * dt, paths_array[i, j-1]) + vol_bump
                paths_array[i][j] = paths_array[i][j-1] * np.exp((self.r - 0.5*vol**2)*dt + vol * np.sqrt(dt) * rand)

        return paths_array

    def mc_heston(self, expiry: float, iterations: int, steps: int, seed: int, vol_bump: float = 0.0):
        # scheme, negvar, numPaths, rho, S_0, V_0, T, kappa, theta, sigma, r = 0.0, q = 0.0, seed = 31
        scheme = 'Euler'
        negvar = 'Reflect'
        numPaths = iterations
        S_0 = self.spot
        T = expiry
        params = self.vols.params.iloc[-6]  # simplistic assumption of constant term parameters
        rho = params.rho
        V_0 = params.v0
        kappa = params.k
        theta = params.theta
        sigma = params.eta
        r = q = 0.0

        dt = T/steps    #0.001
        num_time = int(T / dt)
        S = np.zeros((num_time, numPaths))   # num_time+1
        S[0, :] = S_0
        V = np.zeros((num_time, numPaths))   # num_time+1
        V[0, :] = V_0
        Vcount0 = 0

        np.random.seed(seed)
        Zv_array = np.random.randn(steps, numPaths)
        Zs_array = rho * Zv_array + np.sqrt(1 - rho ** 2) * np.random.randn(steps, numPaths)
        for i in range(numPaths):       # borrowed code: heston Euler discritization
            for t_step in range(1, num_time):   # num_time+1

                # Zv = np.random.randn(1)
                # Zs = rho * Zv + np.sqrt(1 - rho ** 2) * np.random.randn(1)
                Zv = Zv_array[t_step, i]
                Zs = Zs_array[t_step, i]

                if scheme == 'Euler':
                    V[t_step, i] = V[t_step - 1, i] + kappa * (theta - V[t_step - 1, i]) * dt + sigma * np.sqrt(
                        V[t_step - 1, i]) * np.sqrt(dt) * Zv
                elif scheme == 'Milstein':
                    V[t_step, i] = V[t_step - 1, i] + kappa * (theta - V[t_step - 1, i]) * dt + sigma * np.sqrt(
                        V[t_step - 1, i]) * np.sqrt(dt) * Zv + 1 / 4 * sigma ** 2 * dt * (Zv ** 2 - 1)

                if V[t_step, i] <= 0:    # handle negative variance
                    Vcount0 = Vcount0 + 1
                    if negvar == 'Reflect':
                        V[t_step, i] = abs(V[t_step, i])
                    elif negvar == 'Trunca':
                        V[t_step, i] = max(V[t_step, i], 0)

                V[t_step - 1, i] = (np.sqrt(V[t_step - 1, i]) + vol_bump) ** 2
                S[t_step, i] = S[t_step - 1, i] * np.exp((r - q - V[t_step - 1, i] / 2) * dt +
                                                         np.sqrt(V[t_step - 1, i]) * np.sqrt(dt) * Zs)
        return S     # V, Vcount0

    def finite_difference(self):    # implicit, explicit, Crank Nicolson
        pass

    def binomial_tree(self):
        pass
###############################################################
