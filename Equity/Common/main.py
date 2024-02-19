from Equity.Common import utility
import datetime as dt
import time

if __name__ == '__main__':
    start = time.time()

    ###############################################################
    # 0. market & static data

    # market data
    df_data_ts = utility.option_quotes()
    terms = [dt.date(2023, 9, 15), dt.date(2023, 10, 23), dt.date(2023, 11, 17),
             dt.date(2023, 12, 15), dt.date(2024, 1, 19)]

    # static data
    spot = 4458.15   # close 4478.87  open 4437.86
    ref_date = dt.date(2023, 8, 15)

    ###############################################################
    # 1. imply forwards, vols and clean data

    fwd_ts = utility.imply_forward(df_data_ts, spot)   # imply forward term structure
    df_data_ts_refactored = utility.refactor_df(df_data_ts, fwd_ts, terms, ref_date, spot)   # refactor df based on fwd
    df_raw_surface = utility.implied_vols(df_data_ts_refactored, terms, use_saved=True, override_saved=False)
    # utility.plot_scattered_surface(df_raw_surface)  # plot implied vols slices
    #todo check iv fit error

    ###############################################################
    # 2. interpolate/extrapolate smooth smile by term for min/max strike range

    implied_surface_obj = utility.smooth_surface(df_raw_surface, 'spline', strike_min=3750, strike_max=5250,
                                                  strike_delta=50, time_delta=0.05)
    interp_surface = implied_surface_obj.get_surface()
    # plot smooth implied vol surface
    # utility.plot_continuous_surface(interp_surface, label="Implied Vol")

    ###############################################################
    # 3. calibrate local vol surface: Dupire

    local_surface_obj = utility.Dupire_local(implied_surface_obj, spot, rate=0.0, div=0.0, method="lv_from_iv")
    local_surface = local_surface_obj.get_surface()
    # utility.plot_continuous_surface(local_surface, label='Local Vol')
    # todo round trip calibration, test fit

    ###############################################################
    # 4. calibrate stochastic vol surface: Heston

    stochastic_surface_obj = utility.Heston_model(spot, 0.0, df_raw_surface, strike_min=4000, strike_max=5000)
    stochastic_surface_obj.fit_quotes(use_saved=True, override_saved=False)
    stochastic_surface = stochastic_surface_obj.get_surface()
    # utility.plot_continuous_surface(stochastic_surface, label='Stochastic Vol')

    ###############################################################
    # 5. Price and Risk analytics

    strike = 4400
    barrier = 4300
    expiry = 3.0/12.0
    vol = implied_surface_obj.get_vol(expiry, strike)   # implied Vol surface

    # Donw&InPut discrete monitoring Price and Greeks with Local Vol surface.
    # Assuming no Vol change dynamics with Spot moves.
    price_lv, delta_lv, gamma_lv, theta_lv, vega_lv = utility.barrier_opt(barrier_type='Down&InPut',
                                                                          strike=strike, barrier=barrier,
                                                           expiry=expiry, vol_surface=local_surface_obj,
                                                           surface_type='Dupire', spot=spot,
                                                           rate=0.0, iterations=100000, steps=100)

    # Donw&InPut discrete monitoring Price and Greeks with Stochastic Vol surface.
    # Assuming no Vol dynamics with Spot moves.
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