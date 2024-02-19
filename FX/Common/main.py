from FX.Common import utility
import datetime as dt
import numpy as np
import time

if __name__ == '__main__':
    start = time.time()
    ###############################################################
    # 0. market & static data

    df_data_ts = utility.option_quotes()
    spot = 1.0910  # close price
    ref_date = dt.date(2023, 8, 15)
    rate_d = 0.0  # domestic rate
    rate_f = 0.0  # foreign rate

    ###############################################################
    # 1. imply strikes, vols and clean data

    df_data_ts = utility.refactor_df(df_data_ts)

    # imply strikes
    for index, row in df_data_ts.iterrows():
        df_data_ts.at[index, '10_Put_Strike'] = utility.implied_strike("Put", spot, rate_d, rate_f,
                                                                      row['10_Put_Vol'], row['Term'], 0.1)

        df_data_ts.at[index, '25_Put_Strike'] = utility.implied_strike("Put", spot, rate_d, rate_f,
                                                                      row['25_Put_Vol'], row['Term'], 0.25)

        df_data_ts.at[index, 'ATM_Strike'] = spot * np.exp((rate_d-rate_f)*row['Term'])

        df_data_ts.at[index, '25_Call_Strike'] = utility.implied_strike("Call", spot, rate_d, rate_f,
                                                                      row['25_Call_Vol'], row['Term'], 0.25)

        df_data_ts.at[index, '10_Call_Strike'] = utility.implied_strike("Call", spot, rate_d, rate_f,
                                                                      row['10_Call_Vol'], row['Term'], 0.1)

    df_data_ts_flatten = utility.flatten_df(df_data_ts)
    # utility.plot_scattered_surface(df_data_ts_flatten)  # plot implied vols slices

    ###############################################################
    # 2. interpolate/extrapolate smooth smile by term for min/max strike range

    terms = list(df_data_ts_flatten.Term.unique().round(5))
    terms = terms + [6.0, 7.0, 8.0, 9.0]   # additional terms
    terms.sort()
    implied_surface_obj = utility.smooth_surface(df_data_ts_flatten, 'spline',
                                                 delta_min=0.1,
                                                 delta_max=0.9,
                                                 strike_delta=0.05, interp_terms=terms)
    interp_surface = implied_surface_obj.get_surface()

    # plot smooth implied vol surface
    # utility.plot_continuous_surface(interp_surface, 'Implied Vol')
    ###############################################################
    # 3. calibrate local vol surface: Dupire

    for index, row in interp_surface.iterrows():
        interp_surface.at[index, 'Strike'] = utility.implied_strike("Call", spot, rate_d, rate_f,
                                                                    row['Vol'], row['Term'], row['Delta'])

    interp_surface.Delta = interp_surface.Delta.round(4)
    interp_surface.loc[interp_surface.Delta == 0.5, 'Strike'] = spot
    interp_surface.sort_values(['Term', 'Strike'], ascending=[True, True], inplace=True)
    interp_surface.reset_index(drop=True, inplace=True)

    local_surface_obj = utility.Dupire_local(implied_surface_obj, spot, domestic_r=0.0,
                                             foreign_r=0.0, method="lv_from_iv")
    local_surface = local_surface_obj.get_surface()
    # utility.plot_continuous_surface(local_surface, 'Local Vol')
    ###############################################################
    # 4. calibrate stochastic vol surface: Heston

    stochastic_surface_obj = utility.Heston_model(spot, rate_d - rate_f, df_data_ts_flatten,
                                                  strike_min=spot * 0.85, strike_max=spot * 1.15)
    stochastic_surface_obj.fit_quotes(use_saved=True, override_saved=False)
    stochastic_surface = stochastic_surface_obj.get_surface()
    stochastic_surface = stochastic_surface.loc[(stochastic_surface.Term > 1)]
    # utility.plot_continuous_surface(stochastic_surface, label='Stochastic Vol', y_axis='Strike')
    ###############################################################
    # 5. Price and Risk analytics

    strike = spot
    barrier = spot*0.95
    expiry = 3.0
    vol = implied_surface_obj.get_vol(expiry, strike)

    # Donw&InPut discrete monitoring Price and Greeks with Local Vol surface.
    # Assuming no Vol change dynamics with Spot moves.
    price_lv, delta_lv, gamma_lv, theta_lv, vega_lv = utility.barrier_opt(barrier_type='Down&InPut', strike=strike,
                                                                          barrier=barrier,
                                                           expiry=expiry, vol_surface=local_surface_obj,
                                                           surface_type='Dupire', spot=spot, rate=0.0,
                                                           iterations=100000, steps=100)

    # Donw&InPut discrete monitoring Price and Greeks with Stochastic Vol surface.
    # Assuming no Vol change dynamics with Spot moves.
    price_sv, delta_sv, gamma_sv, theta_sv, vega_sv = utility.barrier_opt(barrier_type='Down&InPut',
                                                                          strike=strike, barrier=barrier,
                                                           expiry=expiry, vol_surface=stochastic_surface_obj,
                                                           surface_type='Heston', spot=spot,
                                                           rate=0.0, iterations=100000, steps=100)

    # Analytic Price & Greeks with BS vols from Implied surface
    price_cf = utility.barrier_analytic(spot, strike, barrier, vol, expiry, rate=0.0, div=0.0,
                                                      barrier_type="Donwn&InPut", obs_type="continuous")

    price_list = [price_cf.item(), price_lv, price_sv]
    ###############################################################

    print(round(time.time() - start, 2), " seconds")
