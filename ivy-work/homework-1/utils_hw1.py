import pandas as pd
import numpy as np
import math as math

"""
BLACK'S FORMULA FOR BOND OPTIONS
"""
def normal_cdf(zscore):
    """
    N(z): standard normal CDF.
    Interpretable as: Prob(Z <= z) where Z ~ N(0,1).
    """
    return 0.5 * (1.0 + math.erf(zscore / math.sqrt(2.0)))


def calc_call_option(time_to_expiration, discount_factor, fwd_price, strike, vol, face_value = 100):
    """
    This function calculates the price of a call option on a vanilla bond using Black's Formula.

    Equation:
    C0 = Z(0,T) * [ F0 * N(d1) - K * N(d2) ] * face_value

        such that: d1 = (ln(F0/K) + 0.5*σ^2*T) / (σ*sqrt(T))
                   d2 = d1 - σ*sqrt(T)

    Args:
    - time_to_expiration (float): the time between expiry T and valuation date t.
    - discount_factor (float):
    - fwd_price (float): forward underlying 
    - strike (float): the strike price K
    - vol (float): implied vol for that market in decimal
    - face_value (float): the face value of the vanilla bond

    Returns (float): the value of the call option 
    """
    intrinsic_forward = max(fwd_price - strike, 0.0)
    if time_to_expiration <= 0 or vol <= 0:
        return discount_factor * intrinsic_forward * face_value
    
    d1 = (math.log(fwd_price / strike) + 0.5 * vol**2 * time_to_expiration) / (vol * math.sqrt(time_to_expiration))
    d2 = d1 - (vol * math.sqrt(time_to_expiration))

    Nd1 = normal_cdf(d1)
    Nd2 = normal_cdf(d2)

    return discount_factor * (fwd_price * Nd1 - strike * Nd2) * face_value

