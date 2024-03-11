import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from functools import cache
from pathlib import Path

def phi(x):
    if 0 <= x <= 10:
        return math.exp(-0.4527 * x ** 0.86 + 0.0218)
    elif x > 10:
        return math.sqrt(math.pi / x) * math.exp(-x / 4) * (1 - 10 / (7 * x))
    else:
        return 1

def inv_phi(x):
    return fsolve(lambda ys: [phi(ys[0]) - x], [1])[0]

@cache
def mu_cl(l, dv, dc, var):
    if l == 0:
        return 0
    return inv_phi(1 - (1 - phi(2 / var + (dv - 1) * mu_cl(l - 1, dv, dc, var))) ** (dc - 1))

def mu_v(mu_c_prev, dv, var):
    return 2 / var + (dv - 1) * mu_c_prev

def mu_c(mu_v_curr, dc):
    return inv_phi(1 - (1 - phi(mu_v_curr)) ** (dc - 1))

def is_below_threshold(var, dv, dc, lmax=1000):
    p, q = 0, 0
    for _ in range(lmax):
        q = mu_v(p, dv, var)
        p = mu_c(q, dc)
        if p > 2:
            return False
    return True

def find_bp_threshold(dv, dc):
    rate = 1 - dv / dc
    low, high = 0, 10
    while high - low > 1e-6:
        snr = (low + high) / 2
        var = 1 / (2 * rate * 10 ** (snr / 10))
        if is_below_threshold(var, dv, dc):
            low = snr
        else:
            high = snr
    return low

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
        if abs(z - x) > 1e-6:
            fails += 1
    if fails == 0:
        print('[info] inv_phi testcase passed.')
    else:
        print(f'[info] {fails} number of failures.')

def plot_9_5():
    print("[info] Plotting...")
    dv, dc = 3, 6
    snrs_dB = [1.162, 1.163, 1.165, 1.170]
    rate = 1 - dv / dc
    legend = []
    for snr in snrs_dB:
        var = 1 / (2 * rate * 10 ** (snr / 10))
        xs = range(0, 1000)
        ys = [mu_cl(l, dv, dc, var) for l in xs]
        legend.append(f'SNR = {snr} (dB)')
        plt.plot(xs, ys)
    plt.axis([0, 1000, 0, 10])
    plt.legend(legend)
    plt.grid(True)
    plt.xlabel('Iterations')
    plt.ylabel(r'$\mu^{(c)}$')
    Path("plots").mkdir(parents=True, exist_ok=True)
    plt.savefig("plots/R9_5_temp.png")
    plt.clf()
    # plt.show()

def plot_P10():
    dv, dc = 3, 6
    snr_dB = 1.63
    rate = 1 - dv / dc
    var = 1 / (2 * rate * 10 ** (snr_dB / 10))
    xs = np.linspace(0, 11)
    mu_vs = [mu_v(x, dv, var) for x in xs]
    mu_cs = [mu_c(x, dc) for x in xs]
    plt.plot(xs, mu_vs)
    plt.plot(mu_cs, xs)
    plt.title(f'SNR = {snr_dB} (dB)')
    plt.axis([0, 4, 0, 11])
    plt.legend([r'$\mu_{l}^{(v)}(\mu_{l - 1}^{(c)})$', r'$\mu_{l}^{(c)}(\mu_{l}^{(v)})$'])
    Path("plots").mkdir(parents=True, exist_ok=True)
    plt.savefig("plots/P10_temp.png")
    plt.clf()
    # plt.show()

if __name__ == '__main__':
    test_phi_decreasing()
    test_inv_phi()
    plot_9_5()
    plot_P10()
    ensambles = [(3, 6), (4, 8), (5, 10)]
    thresholds = [find_bp_threshold(*params) for params in ensambles]
    for (dv, dc), threshold in zip(ensambles, thresholds):
        print(f'({dv}, {dc}): {threshold:.04f}')
