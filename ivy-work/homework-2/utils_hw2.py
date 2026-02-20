import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

"""
STRIPPING CAPS
"""
def black_price_floorlet(fwd_vol, tau, strike, fwd_rate, discount_factor, 
                         notional = 100.0, frequency = 0.25):
    """
    This function uses Black's formula to price a floorlet.
    
    Equation:
    Floorlet(t) = (notional * frequency) * Z(t,T) * [ K * N(-d2) - F_t * N(-d1) ]

    Args:
    - fwd_vol (float): Black forward volatility
    - tau (float): time to option expiration (T - frequency)
    - strike (float): "K" or the T-maturity swap rate
    - fwd_rate (float): forward rate F_t for [tau, T]
    - discount_factor (float): discount factor to payment date T
    - notional (float): default to 100.0
    - frequency (float): accrual fraction (default quarterly = 0.25)

    Returns (float): the present value of the floorlet.
    """
    if fwd_vol <= 0 or tau <= 0:
        #intrinsic value case
        return notional * frequency * discount_factor * max(strike - fwd_rate, 0.0)
    
    d1 = (np.log(fwd_rate / strike) + (0.5 * fwd_vol**2 * tau)) / (fwd_vol * np.sqrt(tau))
    d2 = d1 - (fwd_vol * np.sqrt(tau))
    
    B_put = discount_factor * (strike * norm.cdf(-d2) - fwd_rate * norm.cdf(-d1))

    floorlet_price = notional * frequency * B_put

    return float(floorlet_price)


def black_price_caplet(fwd_vol, tau, strike, fwd_rate, discount_factor, 
                       notional = 100.0, frequency = 0.25):
    """
    This function uses Black's formula to price a caplet.
    
    Equation:
    Caplet(t) = (notional * frequency) * Z(t,T) * [ F_t * N(d1) - K * N(d2) ]

    Args:
    - fwd_vol (float): Black forward volatility
    - tau (float): time to option expiration (T - frequency)
    - strike (float): "K" or the T-maturity swap rate
    - fwd_rate (float): forward rate F_t for [tau, T]
    - discount_factor (float): discount factor to payment date T
    - notional (float): default to 100.0
    - frequency (float): accrual fraction (default quarterly = 0.25)

    Returns (float): the present value of the caplet.
    """
    if fwd_vol <= 0 or tau <= 0:
        #intrinsic value case
        return notional * frequency * discount_factor * max(fwd_rate - strike, 0.0)
    
    d1 = (np.log(fwd_rate / strike) + (0.5 * fwd_vol**2 * tau)) / (fwd_vol * np.sqrt(tau))
    d2 = d1 - (fwd_vol * np.sqrt(tau))
    
    B_call = discount_factor * (fwd_rate * norm.cdf(d1) - strike * norm.cdf(d2))

    caplet_price = notional * frequency * B_call

    return float(caplet_price)


def black_implied_vol_caplet(caplet_pv, tau, strike, fwd_rate, discount_factor, 
                             notional = 100.0, frequency = 0.25):
    """
    This function solves for the forward vol that reproduces the given caplet price using Black's formula.
    
    Args:
    - caplet_pv (float): stripped caplet price (in dollars)
    - tau (float): time to option expiration (T - frequency)
    - strike (float): "K" or the T-maturity swap rate
    - fwd_rate (float): forward rate F_t for [tau, T]
    - discount_factor (float): discount factor to payment date T
    - notional (float): default to 100.0
    - frequency (float): accrual fraction (default quarterly = 0.25)

    Returns (float): the forward volatility (annualized)
    """
    def objective(vol):
        price = black_price_caplet(vol, tau, strike, fwd_rate, discount_factor, notional, frequency)
        return price - caplet_pv
    
    if caplet_pv <= 1e-10: #caplet has no meaningful vol if price is effectively zero
        return np.nan
    
    try:
        implied_vol = brentq(objective, 1e-6, 10.0, xtol=1e-8, maxiter=500)
    except ValueError:
        implied_vol = np.nan #price is outside the range achievable by the vol bounds

    return implied_vol
    

"""
PRICING SWAPTIONS
"""
def calc_swap_annuity(discounts_df, swap_start, swap_tenor, frequency = 0.25):
    """
    This function computes the swap annuity.

    Equation:
    A(0) = sum_{i=1..M} [ tau * Z(0,T_i) ]

    Args:
    - discounts_df (pd.Dataframe): dataframe that has the columns "tenor" and "discounts"
    - swap_start (float): when the swap begins
    - swap_tenor (float): length of underlying swap
    - frequency (int): accrual fraction (default quarterly = 0.25)

    Returns (float): annuity A(0)
    """
    num_payments = int(round(swap_tenor / frequency))
    payment_dates = (swap_start + frequency * np.arange(1, num_payments + 1))

    discount_factors_on_payments = np.interp(payment_dates, discounts_df["tenor"], discounts_df["discounts"])

    return float(np.sum(frequency * discount_factors_on_payments))


def calc_forward_swap_rate(discounts_df, swap_start, swap_tenor, frequency = 0.25):
    """
    Docstring for calc_forward_swap_rate

    Equation:
    c_sw^{fwd}(0;T0,Tn;frequency) = frequency * (Z(0,T0) - Z(0,Tn)) / sum_{i=1..M} Z(0,T_i)
    
    Args:
    - discounts_df (pd.Dataframe): dataframe that has the columns "tenor" and "discounts"
    - swap_start (float): when the swap begins
    - swap_tenor (float): length of underlying swap
    - frequency (int): accrual fraction (default quarterly = 0.25)

    Returns (float): the forward swap rate in decimal
    """
    swap_end = swap_start + swap_tenor

    discount_start = np.interp(swap_start, discounts_df["tenor"], discounts_df["discounts"])
    discount_end = np.interp(swap_end, discounts_df["tenor"], discounts_df["discounts"])

    annuity = calc_swap_annuity(discounts_df, swap_start, swap_tenor, frequency)

    return (discount_start - discount_end) / annuity


def black_price_payer_swaption(black_vol, swap_start, strike, fwd_swap_rate, annuity,
                              notional = 100.0):
    """
    This function uses Black's formula to price a payer swaption (call on forward swap rate).

    Equation:
    PV = notional * A(0) * [ S0*N(d1) - K*N(d2) ]

    Args:
    - black_vol (float): volatility in decimal
    - swap_start (float): when the swap begins
    - strike (float): "K" or the T-maturity swap rate
    - fwd_swap_rate (float): the forward swap rate in decimal
    - annuity (float): A(0)
    - notional (float): default 100.0 (match your convention)

    Returns (float): present value
    """
    if black_vol <= 0 or swap_start <= 0:
        return float(notional * annuity * max(fwd_swap_rate - strike, 0.0))

    if fwd_swap_rate <= 0 or strike <= 0:
        raise ValueError("Black swaption formula requires positive S0 and K.")

    d1 = (np.log(fwd_swap_rate / strike) + 0.5 * black_vol**2 * swap_start) / (black_vol * np.sqrt(swap_start))
    d2 = d1 - (black_vol * np.sqrt(swap_start))

    return float(notional * annuity * (fwd_swap_rate * norm.cdf(d1) - strike * norm.cdf(d2)))


