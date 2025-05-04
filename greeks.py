from scipy.stats import norm
from math import exp, sqrt, pi, log


def d_1(S, K, T, r, sigma):
    """
    Calculate d1 for the Black-Scholes model.

    Parameters:
    S : float : Current stock price
    K : float : Strike price
    T : float : Time to expiration (in years)
    r : float : Risk-free interest rate (annualized)
    sigma : float : Volatility of the underlying asset (annualized)

    Returns:
    float : d1 value
    """

    return (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))


def d_2(S, K, T, r, sigma):
    """
    Calculate d2 for the Black-Scholes model.
    Parameters:
    S : float : Current stock price
    K : float : Strike price
    T : float : Time to expiration (in years)
    r : float : Risk-free interest rate (annualized)
    sigma : float : Volatility of the underlying asset (annualized)
    Returns:
    float : d2 value
    """
    from math import log, sqrt

    return d_1(S, K, T, r, sigma) - sigma * sqrt(T)


def Nd_1(d1):
    """
    Calculate the cumulative distribution function for d1.

    Parameters:
    d1 : float : d1 value

    Returns:
    float : CDF value
    """

    return norm.cdf(d1)


def Nd_2(d2):
    """
    Calculate the cumulative distribution function for d2.

    Parameters:
    d2 : float : d2 value

    Returns:
    float : CDF value
    """

    return norm.cdf(d2)


def N_d_1(d1):
    """
    Calculate the cumulative distribution function for d1.

    Parameters:
    d1 : float : d1 value

    Returns:
    float : CDF value
    """

    return norm.cdf(d1)


def N_d_2(d2):
    """
    Calculate the cumulative distribution function for d2.

    Parameters:
    d2 : float : d2 value

    Returns:
    float : CDF value
    """
    return norm.cdf(d2)


def pdf(x):
    """
    Standard normal PDF.
    """
    return (1.0 / sqrt(2 * pi)) * exp(-0.5 * x * x)


def option_price(option_type, S, K, T, r, sigma):
    """
    Black-Scholes option price.

    option_type : 'call' or 'put'
    S, K, T, r, sigma : as usual
    """
    # --- early payoff at expiry ---
    if T <= 0:
        if option_type.lower() == "call":
            return max(S - K, 0.0)
        elif option_type.lower() == "put":
            return max(K - S, 0.0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    d1 = d_1(S, K, T, r, sigma)
    d2 = d_2(S, K, T, r, sigma)
    if option_type.lower() == "call":
        return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == "put":
        return K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def delta(option_type, S, K, T, r, sigma):
    # --- at expiry delta is step function ---
    if T <= 0:
        if option_type.lower() == "call":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0

    d1 = d_1(S, K, T, r, sigma)
    if option_type.lower() == "call":
        return norm.cdf(d1)
    elif option_type.lower() == "put":
        return norm.cdf(d1) - 1
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def gamma(S, K, T, r, sigma):
    if T <= 0:
        return 0.0
    d1 = d_1(S, K, T, r, sigma)
    return pdf(d1) / (S * sigma * sqrt(T))


def vega(S, K, T, r, sigma):
    if T <= 0:
        return 0.0
    d1 = d_1(S, K, T, r, sigma)
    return S * pdf(d1) * sqrt(T)


def theta(option_type, S, K, T, r, sigma):
    if T <= 0:
        return 0.0
    d1 = d_1(S, K, T, r, sigma)
    d2 = d_2(S, K, T, r, sigma)
    term1 = -(S * pdf(d1) * sigma) / (2 * sqrt(T))
    if option_type.lower() == "call":
        term2 = -r * K * exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == "put":
        term2 = r * K * exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return term1 + term2


def rho(option_type, S, K, T, r, sigma):
    if T <= 0:
        return 0.0
    d2 = d_2(S, K, T, r, sigma)
    if option_type.lower() == "call":
        return K * T * exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == "put":
        return -K * T * exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


if __name__ == "__main__":
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    print("Call price:", option_price("call", S, K, T, r, sigma))
    print("Put  price:", option_price("put", S, K, T, r, sigma))
