"""Implied volatility calculation from option prices."""

import math
from typing import Optional, Tuple

import numpy as np
from scipy import stats


def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    is_call: bool,
) -> float:
    """
    Calculate Black-Scholes option price.

    Args:
        S: Spot price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        sigma: Volatility
        is_call: True for call, False for put

    Returns:
        Option price
    """
    if T <= 0 or sigma <= 0:
        # Return intrinsic value
        if is_call:
            return max(S - K, 0)
        else:
            return max(K - S, 0)

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if is_call:
        price = S * stats.norm.cdf(d1) - K * math.exp(-r * T) * stats.norm.cdf(d2)
    else:
        price = K * math.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)

    return price


def black_scholes_vega(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> float:
    """
    Calculate Black-Scholes vega.

    Args:
        S: Spot price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        sigma: Volatility

    Returns:
        Vega (sensitivity to volatility)
    """
    if T <= 0 or sigma <= 0:
        return 0.0

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return S * math.sqrt(T) * stats.norm.pdf(d1)


def calculate_iv_bisection(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    is_call: bool,
    tol: float = 1e-6,
    max_iter: int = 100,
    vol_min: float = 0.001,
    vol_max: float = 5.0,
) -> Optional[float]:
    """
    Calculate implied volatility using bisection method.

    Args:
        price: Market option price
        S: Spot price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        is_call: True for call, False for put
        tol: Convergence tolerance
        max_iter: Maximum iterations
        vol_min: Minimum volatility bound
        vol_max: Maximum volatility bound

    Returns:
        Implied volatility or None if cannot find
    """
    if T <= 0 or price <= 0:
        return None

    # Check bounds
    price_at_min = black_scholes_price(S, K, T, r, vol_min, is_call)
    price_at_max = black_scholes_price(S, K, T, r, vol_max, is_call)

    if price < price_at_min or price > price_at_max:
        return None

    # Bisection
    low, high = vol_min, vol_max

    for _ in range(max_iter):
        mid = (low + high) / 2
        price_mid = black_scholes_price(S, K, T, r, mid, is_call)

        if abs(price_mid - price) < tol:
            return mid

        if price_mid < price:
            low = mid
        else:
            high = mid

    return (low + high) / 2


def calculate_iv_newton(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    is_call: bool,
    tol: float = 1e-6,
    max_iter: int = 50,
    initial_guess: float = 0.2,
) -> Optional[float]:
    """
    Calculate implied volatility using Newton-Raphson method.

    Faster than bisection when vega is well-behaved.

    Args:
        price: Market option price
        S: Spot price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        is_call: True for call, False for put
        tol: Convergence tolerance
        max_iter: Maximum iterations
        initial_guess: Starting volatility guess

    Returns:
        Implied volatility or None if cannot converge
    """
    if T <= 0 or price <= 0:
        return None

    sigma = initial_guess

    for _ in range(max_iter):
        price_est = black_scholes_price(S, K, T, r, sigma, is_call)
        vega = black_scholes_vega(S, K, T, r, sigma)

        if abs(vega) < 1e-10:
            # Vega too small, fall back to bisection
            return calculate_iv_bisection(price, S, K, T, r, is_call, tol, max_iter)

        diff = price_est - price

        if abs(diff) < tol:
            return sigma

        sigma = sigma - diff / vega

        # Bound check
        if sigma <= 0.001:
            sigma = 0.001
        if sigma > 5.0:
            sigma = 5.0

    # Did not converge, try bisection as fallback
    return calculate_iv_bisection(price, S, K, T, r, is_call, tol, max_iter)


def calculate_forward_from_options(
    calls: np.ndarray,
    puts: np.ndarray,
    strikes: np.ndarray,
    S: float,
    r: float,
    T: float,
) -> float:
    """
    Calculate forward price from put-call parity.

    From put-call parity: C - P = S - K*exp(-rT)
    Rearranging: F = K + (C - P) * exp(rT)

    We find the strike where C ≈ P (ATM forward strike).

    Args:
        calls: Call prices
        puts: Put prices
        strikes: Strike prices
        S: Spot price
        r: Risk-free rate
        T: Time to expiration

    Returns:
        Estimated forward price
    """
    if len(calls) == 0 or len(puts) == 0:
        # Fall back to cost-of-carry model
        return S * math.exp(r * T)

    # Find where call - put is closest to zero (ATM forward)
    diff = calls - puts
    atm_idx = np.argmin(np.abs(diff))

    # Use put-call parity at ATM
    K_atm = strikes[atm_idx]
    C_atm = calls[atm_idx]
    P_atm = puts[atm_idx]

    # F = K + (C - P) * exp(rT)
    forward = K_atm + (C_atm - P_atm) * math.exp(r * T)

    return forward


def calculate_iv_from_mid_prices(
    strikes: np.ndarray,
    mid_prices: np.ndarray,
    is_call: np.ndarray,
    underlying_price: float,
    tte_years: float,
    risk_free_rate: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate implied volatilities from mid prices.

    Args:
        strikes: Strike prices
        mid_prices: Mid prices (bid+ask)/2
        is_call: Boolean array indicating call (True) or put (False)
        underlying_price: Current underlying price
        tte_years: Time to expiration in years
        risk_free_rate: Risk-free rate

    Returns:
        (implied_vols, valid_mask) - IV array and boolean mask of valid calculations
    """
    n = len(strikes)
    ivs = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)

    for i in range(n):
        iv = calculate_iv_newton(
            price=mid_prices[i],
            S=underlying_price,
            K=strikes[i],
            T=tte_years,
            r=risk_free_rate,
            is_call=is_call[i],
        )

        if iv is not None and 0.01 <= iv <= 3.0:
            ivs[i] = iv
            valid[i] = True

    return ivs, valid
