import pandas as pd
import numpy as np
from scipy.stats import norm
import datetime as dt
import matplotlib.pyplot as plt
from typing import List
from scipy import interpolate
from matplotlib import cm
from scipy.integrate import quad
from scipy.optimize import minimize
import time


"""
import data and check for any NaNs
"""
def option_quotes() -> List[pd.DataFrame]:
    path = "/Users/jivitesh/Desktop/FX & Equity Vols/Equity/Resources/Data/Vol_Data.xlsx"
    xlsx = pd.ExcelFile(path)
    df_sep = pd.read_excel(xlsx, 'SPX 15Sep23 expiry')
    df_oct = pd.read_excel(xlsx, 'SPX 20Oct23 expiry')
    df_nov = pd.read_excel(xlsx, 'SPX 17Nov23 expiry')
    df_dec = pd.read_excel(xlsx, 'SPX 15Dec23 expiry')
    df_jan = pd.read_excel(xlsx, 'SPX 19Jan24 expiry')
    dfs = [df_sep, df_oct, df_nov, df_dec, df_jan]
    for df in dfs:
        if df.isnull().any().sum() > 0:
            raise Exception("Data issue")
    return dfs
###############################################################


""" 3 different methods to imply european vols
1. Least squares regression Reference: chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://cdn.cboe.com/api/global/us_indices/governance/Cboe_European-Style_Option_%20Implied_Volatility_Calculation_%20Methodology.pdf
2. imply forward using Put Call parity with the strike price such that the
 absolute difference between the call and put prices is minimum.
3. imply forward iteratively such that put and call vols set intersect
    Reference: chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://www.rivacon.com/wp-content/uploads/american_fwd.pdf
reference:  """
def imply_forward(df_data_ts: List[pd.DataFrame], spot: float) -> List[float]:
    # using 1 method above
    # ignore quotes where bid price is 0, select moneyness within 8%
    fwd_ts = []
    for df in df_data_ts:
        df = df.loc[(df['Call Bid'] > 0) & (df['Put Bid'] > 0)]
        df = df.loc[(df['Strike'] > spot * 0.92) & (df['Strike'] < spot * 1.08)]
        df["Call Mid"] = 0.5 * (df["Call Bid"] + df["Call Ask"])
        df["Put Mid"] = 0.5 * (df["Put Bid"] + df["Put Ask"])

        # ordinary least square estimation of alpha, beta
        # Y = 𝛼 + 𝛽𝑋, 𝑌𝑖 = put(𝐾𝑖)−call(𝐾𝑖)+S, 𝑋𝑖 = 𝐾𝑖  # typo in CBOE paper -S -> +S
        df["Yi"] = df["Put Mid"] - df["Call Mid"] + spot
        df["Xi"] = df["Strike"]

        # 𝛽= Σ(𝑋𝑖−𝑋)(𝑌𝑖−𝑌)/(𝑋𝑖−𝑋)**2, 𝛼= 𝑌− 𝛽𝑋
        X, Y = df["Xi"].mean(), df["Yi"].mean()
        df["Xi-X"] = df["Xi"] - X
        df["Yi-Y"] = df["Yi"] - Y
        beta = sum(df["Xi-X"] * df["Yi-Y"]) / sum(df["Xi-X"] ** 2)  # discount factor
        alpha = Y - beta * X  # discrete dividend

        discount_factor = beta
        div = alpha
        fwd = (spot - div) / discount_factor
        fwd_ts.append(fwd)
    return fwd_ts
###############################################################


""" 
clean up data -> OTM only, remove 0 Bids, Add term, Add Mids
"""
def refactor_df(df_data_ts: List[pd.DataFrame], fwd_ts: List[float],
                terms: List[dt.date], ref_date: dt.date, spot: float) -> List[pd.DataFrame]:
    i = 0
    df_final = []
    for df in df_data_ts:
        # set term
        t = terms[i] - ref_date
        df['T'] = round(t.days/365, 3)

        # clean up calls
        df_call = df.loc[(df['Strike'] >= fwd_ts[i])]
        df_call = df_call.loc[(df_call['Call Bid'] > 0)]
        df_call['Type'] = 'Call'
        df_call['Date'] = terms[i]
        df_call["Mid"] = 0.5 * (df_call["Call Bid"] + df_call["Call Ask"])

        # clean up puts
        df_put = df.loc[(df['Strike'] < fwd_ts[i])]
        df_put = df_put.loc[(df_put['Put Bid'] > 0)]
        df_put['Type'] = 'Put'
        df_put['Date'] = terms[i]
        df_put["Mid"] = 0.5 * (df_put["Put Bid"] + df_put["Put Ask"])

        # include ITM Put from max OTM Call till 6000 to avoid to use more info
        # OTM_strike_max = max(df_call['Strike'])
        # ITM_strike_max = max(df_data_ts[-1].loc[(df_data_ts[-1]['Call Bid'] > 0)]['Strike'])
        # df_put_ITM = df.loc[(df['Strike'] > OTM_strike_max) & (df['Strike'] <= ITM_strike_max)]
        # df_put_ITM['Type'] = 'Put'
        # df_put_ITM['Date'] = terms[i]
        # df_put_ITM["Mid"] = 0.5 * (df_put_ITM["Put Bid"] + df_put_ITM["Put Ask"])

        # concatenate and select columns
        df_call_put = pd.concat([df_put, df_call])   # df_put_ITM
        df_call_put = df_call_put[['Date', 'Strike', 'Type', 'T', 'Mid']]
        df_call_put['S'] = spot
        df_call_put['F'] = fwd_ts[i]
        df_call_put['r'] = 0.0  # todo: imply rate from df regression above??
        df_call_put.reset_index(drop=True, inplace=True)
        df_final.append(df_call_put)
        i += 1
    return df_final
###############################################################


""" 
Black Call with Fwd
"""
def BS_Call(F: float, S: float, K: float, T: float, r:float, sigma:float) -> float:
    d1 = (np.log(F / K) + ((sigma ** 2) * 0.5) * T) / (sigma * np.sqrt(T))
    d2 = d1 - (sigma * np.sqrt(T))
    return (F * norm.cdf(d1) - K * norm.cdf(d2))* np.exp(-r * T)
###############################################################


""" 
Black Put with Fwd
"""
def BS_Put(F: float, S: float, K: float, T: float, r:float, sigma:float) -> float:
    d1 = (np.log(F / K) + ((sigma ** 2) * 0.5) * T) / (sigma * np.sqrt(T))
    d2 = d1 - (sigma * np.sqrt(T))
    return (-F * norm.cdf(-d1) + K * norm.cdf(-d2)) * np.exp(-r * T)
###############################################################


""" 
imply BS Vols with Forward
using Newton Raphson algo for root finding.
Vega = BS closed form i.e. S*sqrt(T)*PDF(d1)
"""
def implied_vols(df_data_ts_refactored: List[pd.DataFrame],
                 terms: List[dt.date], use_saved, override_saved) -> List[pd.DataFrame]:
    vols_path = "/Users/jivitesh/Desktop/FX & Equity Vols/Equity/Resources/Data/vols_saved.csv"
    use_saved = False if override_saved else use_saved
    if use_saved:
        return pd.read_csv(vols_path)

    if override_saved:
        j = 0
        df_raw_surface = pd.DataFrame()
        for df in df_data_ts_refactored:
            df['iv'] = df.apply(lambda x: implied_vol_point(x.F, x.S, x.Strike, x.r, x['T'], x.Type, x.Mid), axis=1)
            print(df.iv[df.iv == 0.0].count(), " strikes ignored with near 0 vega for: ", terms[j])
            df_raw_surface = pd.concat([df_raw_surface, df.loc[(df['iv'] > 0)]])
            j += 1

        df_raw_surface.reset_index(drop=True, inplace=True)
        df_raw_surface.to_csv(vols_path, index=False)
        return df_raw_surface
###############################################################


""" 
imply BS Vols with Forward
using Newton Raphson algo for root finding.
Vega = BS closed form i.e. S*sqrt(T)*PDF(d1)
"""
def implied_vol_point(F: 'forward', S: 'Spot', K: 'Strike', r: 'rate',
                      T: 'Term', Type: "Call/Put", Price: "Target") -> 'vol':
    def vega(S: float, F: float, K: float,
             T: float, r: float, sigma: float) -> float:
        d1 = (np.log(F/K) + ((sigma**2)*0.5)*T)/(sigma * np.sqrt(T))
        return S * np.sqrt(T) * norm.pdf(d1)

    sigma = 0.6  # initial guess
    # sigma = np.sqrt(2*np.pi/T)*Price/S   # starting guess
    tolerance = 0.001
    max_iterations = 100
    if F == None:
        F = S     # 0 rates
    r = 0.0
    for i in range(max_iterations):
        if Type == 'Call':
            cost_fx = BS_Call(F, S, K, T, r, sigma) - Price
        else:
            cost_fx = BS_Put(F, S, K, T, r, sigma) - Price

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
Interpolate/Extrapolate surface and return smooth surface using methodologies:
1) Cubic
2) SABR
thin_plate, multiquadric 
for spline, interpolation in Time is linear in variance
"""
class smooth_surface:
    def __init__(self, raw_surface: pd.DataFrame, method: str, strike_min: int,
                 strike_max: int, strike_delta: int, time_delta: float):
        self.data = raw_surface
        self.strike_min = strike_min
        self.strike_max = strike_max
        self.strike_delta = strike_delta
        self.time_delta = time_delta
        if method == 'spline':
            self.surface_fx = self.spline()
        elif method == 'sabr':
            pass
        else:
            raise Exception("Method not supported")

    def get_surface(self) -> pd.DataFrame:
        return self.surface(self.surface_fx)

    # to be used pricing and calibrating. Assumption: linear interpolation in variance terms over surface
    def get_vol(self, t: float, k: float) -> float:
        """" t: term
            k: strike"""
        f = self.surface_fx
        return np.sqrt(f(t, k)/t)

    def spline(self):
        terms = self.data['T'].unique()
        strike_surface = pd.DataFrame(columns=['T', 'Strike', 'Vol'])
        # Spatial interpolation and extrapolation
        for term in terms:
            df = self.data.loc[self.data['T'] == term]
            # interp = interpolate.CubicSpline(df['Strike'], df['iv'], bc_type="natural", extrapolate=False)
            # extrap = interpolate.Rbf(df['Strike'], df['iv'], function='multiquadric', extrapolate=True)

            interp = interpolate.CubicSpline(df['Strike'], df['iv'], bc_type="natural", extrapolate=False)
            extrap = interpolate.Rbf(df['Strike'], df['iv'], function='thin_plate', smooth=-3, extrapolate=True)

            vol_grid = np.concatenate((extrap(np.arange(self.strike_min, max(self.strike_min, min(df['Strike'])), self.strike_delta)),
                                       interp(np.arange(max(min(df['Strike']), self.strike_min), min(max(df['Strike']), self.strike_max )+ 1, self.strike_delta)),   # 1 instrad of strike_delta to include last strike
                                       extrap(np.arange(max(df['Strike']) + self.strike_delta, self.strike_max + self.strike_delta, self.strike_delta))))
            strike_grid = np.arange(self.strike_min, self.strike_max + self.strike_delta, self.strike_delta)  # spatial grid

            strike_surface_term = pd.DataFrame(data=[np.full(strike_grid.shape, term), strike_grid, vol_grid]).T
            strike_surface_term.columns = ['T', 'Strike', 'Vol']
            strike_surface = pd.concat([strike_surface, strike_surface_term])
            del interp, extrap, vol_grid, df

        # Temporal interpolation => linear in variance
        x = np.array(terms)  # terms
        y = strike_grid
        z = np.ndarray(0)
        for term in terms:
            df = strike_surface.loc[strike_surface['T']==term]
            df.loc[:,['Var_T']] = (df['Vol'] ** 2) * df['T']  # Variance
            z = np.concatenate((z, df['Var_T'].to_numpy()))
            del df

        # variance for interpolated terms
        f = interpolate.interp2d(x, y, z, kind='linear')
        return f    

    def surface(self, f: 'surface_fx'):
        terms = self.data['T'].unique()
        XX = np.arange(terms[0], terms[-1], 0.02)  # temporal grid
        YY = np.arange(self.strike_min, self.strike_max + self.strike_delta, self.strike_delta)

        X, Y = np.meshgrid(XX, YY)
        Z = np.zeros(X.shape)  # use either X or Y
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = np.sqrt(f(X[i, j], Y[i, j]) / X[i, j])

        # compile surface
        final_surface = pd.DataFrame(columns=['T', 'Strike', 'Vol'])
        i = 0
        for maturity in XX:
            final_surface_maturity = pd.DataFrame(data=[np.full(YY.shape, maturity), YY, Z.T[i]]).T
            final_surface_maturity.columns = ['T', 'Strike', 'Vol']
            i += 1
            final_surface = pd.concat([final_surface, final_surface_maturity])
            final_surface.reset_index(drop=True, inplace=True)

        return final_surface

    def sabr(self):
        #todo
        pass
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
plot scatter vol surface
reference: https://matplotlib.org/stable/gallery/mplot3d/scatter3d.html
"""
def plot_scattered_surface(df_ivs: pd.DataFrame) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(df_ivs['Strike'], df_ivs['T'], df_ivs['iv'], marker=".")
    ax.set_xlabel('Strike', color="red")
    ax.set_ylabel('Term', color="red")
    ax.set_zlabel('Vol', color="red")
    plt.title(label="Implied Vols", fontsize=20, color="black")
    # plt.show()
###############################################################


"""
plot interpolated surface
"""
def plot_continuous_surface(df_surface:pd.DataFrame, label) -> None:
    X = df_surface['Strike']
    Y = df_surface['T']
    Z = df_surface['Vol']
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_trisurf(X, Y, Z, cmap=cm.jet,linewidth=0.2,antialiased=True,shade=True)   #cmap=cm.coolwarm,jet
    ax.set_xlabel('Strike', color="red")
    ax.set_ylabel('Term', color="red")
    ax.set_zlabel('Vol', color="red")
    plt.title(label=label+' Surface', fontsize=20, color="black")
    plt.show()
###############################################################


"""
Dupire local vol implementation by Gatheral J.
Ref: https://www.amazon.com/Volatility-Surface-Practitioners-Guide/dp/0471792519
Variance form preferred as denominator (dc2/dk2) tends to 0 in Volatility matrix terms.
Important to check Butterfly arbitrage to avoid negative instantaneous variance.
Using Finite derivatives method
"""
#Todo: add all implementations (Call price, Vol, Variance)
class Dupire_local:
    def __init__(self, implied_surface_obj: smooth_surface, spot: float, rate: float, div: float, method: str):
        self.spot = spot
        self.iv_fx = implied_surface_obj
        self.r = rate
        self.q = div
        if method == 'lv_from_iv':
            self.surface_fx = self.lv_fom_iv
        elif method == 'lv_fom_call':
            pass
        elif method == 'lv_fom_var':
            pass
        else:
            raise Exception("Method not supported")

    def get_surface(self) -> pd.DataFrame:
        return self.surface(self.surface_fx)

    #reference: chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://www.maths.ox.ac.uk/system/files/attachments/Calibration_1.pdf
    def lv_fom_iv(self, t: float, k: float) -> float:
        terms = self.iv_fx.data['T'].unique()
        K_bump = 50   # 1% move
        # calculate all derivatives i.e dv/dt, dv/dk, d2v/dk2
        dvol_dt = (self.iv_fx.get_vol(t+0.02, k) - self.iv_fx.get_vol(t, k))/0.02

        for i in range(len(terms) - 1):
            if t <= terms[0]:
                key_map = terms[0]
                break
            if t >= terms[-1]:
                key_map = terms[len(terms)]
                break
            if terms[i] < t <= terms[i + 1]:
                key_map = terms[i+1]
                break
        del i

        slope_min_strike = min(self.iv_fx.data.loc[self.iv_fx.data['T'] == key_map].Strike) + 2*K_bump
        slope_max_strike = max(self.iv_fx.data.loc[self.iv_fx.data['T'] == key_map].Strike) -2*K_bump

        if k <= slope_min_strike:
            dvol_dk = (self.iv_fx.get_vol(key_map, slope_min_strike + 2*K_bump) - self.iv_fx.get_vol(key_map, slope_min_strike)) / (K_bump * 2)
            d2vol_dk2 = (self.iv_fx.get_vol(key_map, slope_min_strike + 2*K_bump) - 2 * self.iv_fx.get_vol(key_map, slope_min_strike+K_bump)
                         + self.iv_fx.get_vol(key_map, slope_min_strike)) / (K_bump ** 2)
        elif k >= slope_max_strike:
            dvol_dk = (self.iv_fx.get_vol(key_map, slope_max_strike) - self.iv_fx.get_vol(key_map, slope_max_strike - 2*K_bump)) / (K_bump * 2)
            d2vol_dk2 = (self.iv_fx.get_vol(key_map, slope_max_strike) - 2 * self.iv_fx.get_vol(key_map, slope_max_strike-K_bump)
                         + self.iv_fx.get_vol(key_map, slope_max_strike-2*K_bump)) / (K_bump ** 2)
        else:
            dvol_dk = (self.iv_fx.get_vol(t, k+K_bump) - self.iv_fx.get_vol(t, k-K_bump))/(K_bump *2)
            d2vol_dk2 = (self.iv_fx.get_vol(t, k+K_bump) - 2*self.iv_fx.get_vol(t, k) + self.iv_fx.get_vol(t, k-K_bump))/(K_bump**2)
            if abs(d2vol_dk2) < 1e-6:
                d2vol_dk2 = 0.0

        iv = self.iv_fx.get_vol(t, k)    # implied vol from surface
        d1 = (np.log(self.spot/k) + (self.r + 0.5*(iv**2))*t)/(iv*np.sqrt(t))
        numerator = iv**2 + 2*iv*t*dvol_dt + 2*(self.r-self.q)*iv*k*t*dvol_dk
        denominator1 = (1 + k*d1*np.sqrt(t)*dvol_dk)**2
        denominator2 = iv*t*(k**2)*(d2vol_dk2 - d1*(dvol_dk**2)*np.sqrt(t))

        #todo check if either numerator or denominator is <0
        # return min(max(np.sqrt(numerator/(denominator1+denominator2)), 1e-3), 1)
        return np.sqrt(min(max(numerator / (denominator1 + denominator2), 1e-6), 1))

    def surface(self, f: 'surface_fx'):
        K_adj = 250  # adjust strike grid for local surface, since flat variance at boundary
        terms = self.iv_fx.data['T'].unique()

        XX = np.arange(terms[0], terms[-1], self.iv_fx.time_delta)  # temporal grid
        YY = np.arange(self.iv_fx.strike_min + K_adj, self.iv_fx.strike_max + self.iv_fx.strike_delta-K_adj,
                       self.iv_fx.strike_delta)

        X, Y = np.meshgrid(XX, YY)
        Z = np.zeros(X.shape)  # use either X or Y
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = f(X[i, j], Y[i, j])

        # compile surface
        final_surface = pd.DataFrame(columns=['T', 'Strike', 'Vol'])
        i = 0
        for maturity in XX:
            final_surface_maturity = pd.DataFrame(data=[np.full(YY.shape, maturity), YY, Z.T[i]]).T
            final_surface_maturity.columns = ['T', 'Strike', 'Vol']
            i += 1
            final_surface = pd.concat([final_surface, final_surface_maturity])
            final_surface.reset_index(drop=True, inplace=True)

        return final_surface
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
        self.strike_delta = 50   # temporal grid
        self.time_delta = 0.05   # spatial grid

    """
    objective fx to be solved within bid ask vol spread or arbitrary spread around Mid vols
    """
    def Heston_obj_fx(self, params):
        df_heston_slice = self.slice_data.loc[(self.data.Strike >= self.strike_min) &
                                              (self.data.Strike <= self.strike_max)]
        strikes = df_heston_slice['Strike'].values
        iv_mids = df_heston_slice['iv'].values
        v0, k, theta, rho, eta = [x for x in params]

        total_error = 0
        for i in range(len(strikes)):
            strike = strikes[i]
            heston_obj = Heston_parametric(self.S0,  self.r, v0, k, theta, rho, eta)
            heston_call = heston_obj.price(strike, self.term)
            heston_impvol = implied_vol_point(self.S0, self.S0, strike, self.r, self.term, "Call", heston_call)  #F=S

            if heston_impvol < iv_mids[i] * 0.95:  # 5% bid ask assumption for quick fit. Use Bid Vols otherwise
                error = (iv_mids[i] * 0.95 - heston_impvol) ** 2
            elif heston_impvol > iv_mids[i] * 1.05:  # 5% bid ask assumption for quick fit. Use Ask Vols otherwise
                error = (iv_mids[i] * 1.05 - heston_impvol) ** 2
            else:
                error = 0
            total_error += error
        return total_error

    def fit_quotes(self, use_saved, override_saved) -> None:
        use_saved = False if override_saved else use_saved
        params_path = "/Users/jivitesh/Desktop/FX & Equity Vols/Equity/Resources/Data/paramters_saved.csv"
        if use_saved:
            self.params = pd.read_csv(params_path)
        if override_saved:
            all_fitted = []
            params_guess = [0.03, 0.9, 0.05, -0.6, 0.3]   # 'v0', 'k', 'theta', 'rho', 'eta'
            params_bounds = [[0.001, 1], [0.001, 5], [0.001, 1], [-1, 1], [0, 1]]   # 'v0', 'k', 'theta', 'rho', 'eta'

            for term in self.data["T"].unique():
                self.term = term
                self.slice_data = self.data.loc[self.data["T"] == term]
                fitted = minimize(self.Heston_obj_fx, x0=params_guess, bounds=params_bounds, tol=1e-6)
                all_fitted.append(fitted.x)
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
        terms = self.data["T"].unique()
        strike_grid = np.arange(self.strike_min, self.strike_max + self.strike_delta, self.strike_delta)
        surface = pd.DataFrame(columns=['T', 'Strike', 'Vol'])

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
            surface_slice.columns = ['T', 'Strike', 'Vol']
            surface = pd.concat([surface, surface_slice])
        surface.reset_index(drop=True, inplace=True)

        surface.loc[:, ['Var_T']] = (surface['Vol'] ** 2) * surface['T']  # Variance
        # Temporal interpolation => linear in variance
        x = np.array(terms)  # terms
        y = strike_grid
        z = np.ndarray(0)
        for term in terms:
            df = surface.loc[surface['T'] == term]
            z = np.concatenate((z, df['Var_T'].to_numpy()))
            del df

        self.interp_surface_fx =  interpolate.interp2d(x, y, z, kind='linear')

    def surface(self):

        terms = self.data["T"].unique()
        XX = np.arange(terms[0], terms[-1], self.time_delta)  # temporal grid
        YY = np.arange(self.strike_min, self.strike_max + self.strike_delta, self.strike_delta)  # spatial grid
        X, Y = np.meshgrid(XX, YY)
        Z = np.zeros(X.shape)  # use either X or Y
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self.get_vol(X[i, j], Y[i, j])

        # compile surface
        final_surface = pd.DataFrame(columns=['T', 'Strike', 'Vol'])
        i = 0
        for maturity in XX:
            final_surface_maturity = pd.DataFrame(data=[np.full(YY.shape, maturity), YY, Z.T[i]]).T
            final_surface_maturity.columns = ['T', 'Strike', 'Vol']
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
    # iterations=1000
    paths_obj = numerical_method(vol_surface, spot, rate)
    paths_obj_up = numerical_method(vol_surface, spot + 1, rate)  # same seed,$1 move
    paths_obj_down = numerical_method(vol_surface, spot - 1, rate)  # same seed,$1 move
    paths_obj_time = numerical_method(vol_surface, spot, rate)  # for theta
    paths_obj_vol = numerical_method(vol_surface, spot, rate)  # for Vega

    if surface_type == 'Dupire':
        paths = paths_obj.mc_simulations(expiry, iterations, steps, seed=31)
        paths_up = paths_obj_up.mc_simulations(expiry, iterations, steps, seed=31)
        paths_down = paths_obj_down.mc_simulations(expiry, iterations, steps, seed=31)
        paths_time = paths_obj_time.mc_simulations(expiry+1/365, iterations, steps, seed=31)   # 1 day move
        paths_vol = paths_obj_vol.mc_simulations(expiry, iterations, steps, seed=31, vol_bump=0.01)
    else:  # surface_type == 'Heston':
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
    delta = (Price_list[2] - Price_list[0])/(2*1)
    gamma = (Price_list[2] - 2*Price_list[1] + Price_list[0])/(1**2)
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
If time allows will add recombining tree & PDE (intrinsic, Crank Nicolson)
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
        vol_array = np.zeros(iterations)

        # start = time.time()
        for j in range(1, steps):
            for i in range(iterations):
                 vol_array[i] = self.vols.lv_fom_iv(j * dt, paths_array[i, j - 1]) + vol_bump
            rand = rand_array[:, j - 1]
            paths_array[:, j] = paths_array[:, j - 1] * np.exp((self.r - 0.5 * vol_array ** 2) * dt
                                                              + vol_array * np.sqrt(dt) * rand)

        # start = time.time()
        # for i in range(iterations):
        #     for j in range(1, steps):
        #         # np.random.seed(10)
        #         rand = rand_array[i, j - 1]
        #         vol = self.vols.lv_fom_iv(j * dt, paths_array[i, j - 1]) + vol_bump
        #         paths_array[i][j] = paths_array[i][j - 1] * np.exp(
        #             (self.r - 0.5 * vol ** 2) * dt + vol * np.sqrt(dt) * rand)

        # print(round(time.time() - start, 2), " seconds")
        return paths_array

    """Heston discretization: supports Euler and Milstein
    Variance can be truncated to 0 or reflect by taking absolute"""
    def mc_heston(self, expiry: float, iterations: int, steps: int, seed: int, vol_bump: float = 0.0):
        # scheme, negvar, numPaths, rho, S_0, V_0, T, kappa, theta, sigma, r = 0.0, q = 0.0, seed = 31
        scheme = 'Milstein'
        negvar = 'Reflect'
        numPaths = iterations
        S_0 = self.spot
        T = expiry
        params = self.vols.params.iloc[3]  # simplistic assumption of using 3 month parameters
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


