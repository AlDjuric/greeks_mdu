import pytest
from math import exp
from greeks import option_price


@pytest.fixture
def params():
    return {
        "S": 100.0,
        "K": 100.0,
        "T": 1.0,
        "r": 0.05,
        "sigma": 0.2,
    }


def test_call_option_price(params):
    c = option_price("call", **params)
    # Known value ≈ 10.4506
    assert c == pytest.approx(10.4506, rel=1e-4)


def test_put_option_price(params):
    p = option_price("put", **params)
    # Known value ≈ 5.5735
    assert p == pytest.approx(5.5735, rel=1e-4)


def test_put_call_parity(params):
    C = option_price("call", **params)
    P = option_price("put", **params)
    S, K, T, r = params["S"], params["K"], params["T"], params["r"]
    lhs = C - P
    rhs = S - K * exp(-r * T)
    assert lhs == pytest.approx(rhs, rel=1e-6)
