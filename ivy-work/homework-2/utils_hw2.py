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
    
