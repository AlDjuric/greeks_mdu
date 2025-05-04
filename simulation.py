import numpy as np
import matplotlib.pyplot as plt
from greeks import option_price, delta, gamma, vega, theta, rho


def simulate_path(S0, K, T, r, sigma, option_type="call", steps=20, seed=None):
    """
    Simulate one path of the underlying and compute Greeks along the way.
    steps: number of time‐steps (e.g. trading days)
    """
    if seed is not None:
        np.random.seed(seed)
    dt = T / steps
    times = np.linspace(0, T, steps + 1)
    S = np.empty(steps + 1)
    S[0] = S0

    # Pre‐allocate arrays
    price = np.empty_like(S)
    delta_arr = np.empty_like(S)
    gamma_arr = np.empty_like(S)
    vega_arr = np.empty_like(S)
    theta_arr = np.empty_like(S)
    rho_arr = np.empty_like(S)

    # simulate S
    for i in range(steps):
        z = np.random.randn()
        S[i + 1] = S[i] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)

    # compute greeks at each time
    for i, t in enumerate(times):
        T_rem = max(T - t, 0)
        price[i] = option_price(option_type, S[i], K, T_rem, r, sigma)
        delta_arr[i] = delta(option_type, S[i], K, T_rem, r, sigma)
        gamma_arr[i] = gamma(S[i], K, T_rem, r, sigma)
        vega_arr[i] = vega(S[i], K, T_rem, r, sigma)
        theta_arr[i] = theta(option_type, S[i], K, T_rem, r, sigma)
        rho_arr[i] = rho(option_type, S[i], K, T_rem, r, sigma)

    return times, S, price, delta_arr, gamma_arr, vega_arr, theta_arr, rho_arr


if __name__ == "__main__":
    # example
    T = 1  # years
    r = 0.05
    σ = 0.2
    S0 = 100
    K = 100
    steps = 20  # e.g. trading days
    ts, S, P, D, G, Vg, Th, Rho = simulate_path(
        S0, K, T, r, σ, option_type="call", steps=steps, seed=42
    )

    # convert the year‐fractions to days
    days = ts * steps

    plt.figure(figsize=(10, 6))
    plt.plot(days, S, label="Underlying S")
    plt.plot(days, P, label="Option Price")
    plt.xlabel("Time (trading days)")
    plt.ylabel("Price")
    plt.legend()
    plt.savefig("simulation.png", dpi=150)
    print("Saved plot to simulation.png")
