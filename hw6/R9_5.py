import numpy as np
import math
import matplotlib.pyplot as plt
from functools import cache
from pathlib import Path

def phi(x):
    if 0 <= x <= 10:
        return math.exp(-0.4527 * x ** 0.86 + 0.0218)  # return 1 at 0 instead?
    elif x > 10:
        return math.sqrt(math.pi / x) * math.exp(-x / 4) * (1 - 10 / (7 * x))
    else:
        raise ValueError('Phi negative argument')

def inv_phi(x):
    eps = 1e-16
    low, high = 0, 100  # 100 is enough
    while high - low > eps:
        mid = (low + high) / 2
        val = phi(mid)
        if abs(val - x) < eps:
            low = mid
            break
        if val > x:
            low = mid - eps
        else:
            high = mid + eps
    return low

@cache
def mu_c(l, dv, dc, var):
    mu0 = 2 / var
    if l == 0:
        return 0
    return inv_phi(1 - (1 - phi(mu0 + (dv - 1) * mu_c(l - 1, dv, dc, var))) ** (dc - 1))

def mu_v(mu_c_prev, dv, var):
    mu0 = 2 / var
    return mu0 + (dv - 1) * mu_c_prev

def mu_cv(mu_v_curr, dc):
    return inv_phi(1 - (1 - phi(mu_v_curr)) ** (dc - 1))

def test_phi_decreasing():
    xs = np.linspace(0, 20, 100)
    ys = np.array([*map(phi, xs)])
    for y1, y2 in zip(ys, ys[1:]):
        assert y1 >= y2
    print('[info] phi decreasing testcase passed.')
    
def test_inv_phi():
    fails = 0
    for _ in range(100):
        x = 100 * np.random.rand()
        y = phi(x)
        z = inv_phi(y)
        if abs(z - x) > 1e-3:
            fails += 1
    if fails == 0:
        print('[info] inv_phi testcase passed.')
    else:
        print(f'[info] {fails} number of failures.')

def plot_mu_vs_l():
    print("[info] Plotting...")
    dv, dc = 3, 6
    snrs_dB = [1.162, 1.163, 1.165, 1.170]
    rate = 1 - dv / dc
    legend = []
    for snr in snrs_dB:
        var = 1 / (2 * rate * 10 ** (snr / 10))
        xs = range(0, 1000)
        ys = [mu_c(l, dv, dc, var) for l in xs]
        legend.append(f'SNR = {snr} (dB)')
        plt.plot(xs, ys)
    plt.axis([0, 1000, 0, 10])
    plt.legend(legend)
    plt.xlabel('Iterations')
    plt.ylabel(r'$\mu^{(c)}$')
    Path("plots").mkdir(parents=True, exist_ok=True)
    plt.savefig("plots/R9_5_temp.png")
    # plt.show()

def plot_P10():
    dv, dc = 3, 6
    snr_dB = 1.163 # 1.170
    rate = 1 - dv / dc
    var = 1 / (2 * rate * 10 ** (snr_dB / 10))
    ls = range(0, 500)
    mu_cs = [mu_c(l, dv, dc, var) for l in ls]
    mu_vs = [0] + [mu_v(x, dv, var) for x in mu_cs[:-1]]
    mu_cs2 = [mu_cv(x, dc) for x in mu_vs]
    plt.plot(mu_cs, mu_vs)
    plt.plot(mu_vs, mu_cs2)
    plt.title(f'SNR = {snr_dB} (dB)')
    plt.legend([r'$\mu_{l}^{(v)}(\mu_{l - 1}^{(c)})$', r'$\mu_{l}^{(c)}(\mu_{l}^{(v)})$'])
    Path("plots").mkdir(parents=True, exist_ok=True)
    plt.savefig("plots/P10_temp.png")
    # plt.show()

if __name__ == '__main__':
    test_phi_decreasing()
    test_inv_phi()
    plot_P10()
