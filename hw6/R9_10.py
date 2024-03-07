import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from pathlib import Path

def J(sigma):
    xs = np.random.normal(0.5 * sigma ** 2, sigma, size=(1000000,))
    return 1 - np.mean(np.log2(1 + np.exp(-xs)))

def Jinv(x):
    eps = 1e-15
    low, high = 0, 8
    while high - low > 10 * eps:
        mid = (low + high) / 2
        val = J(mid)
        if abs(val - x) < eps:
            low = mid
            break
        if val < x:
            low = mid - eps
        else:
            high = mid + eps
    return low
    # return fsolve(lambda xs: [J(xs[0]) - x], [1])[0]

def IEV(IAV, dv, var_ch):
    return J(np.sqrt((dv - 1) * Jinv(IAV) ** 2 + var_ch))

def IAC(IEC, dc):
    return 1 - J(Jinv(1 - IEC) / np.sqrt(dc - 1))

def plot_EXIT():
    dv, dc = 3, 6
    rate = 1 - dv / dc
    snrs_dB = [0.25, 1.165, 3.0]
    xs = np.linspace(0, 1)
    IACs = [IAC(IEC, dc) for IEC in xs]
    plt.plot(xs, IACs)
    legend = ['-']
    for snr in snrs_dB:
        legend.append(f'SNR = {snr} (dB)')
        var_ch = 8 * rate * 10 ** (snr / 10)
        IEVs = [IEV(IAV, dv, var_ch) for IAV in xs]
        plt.plot(xs, IEVs)
    plt.title(f'SNR = {snr} (dB)')
    plt.legend(legend)
    plt.grid(True)
    plt.xlabel(r'$I_{AV} (I_{EC})$')
    plt.ylabel(r'$I_{EV} (I_{AC})$')
    Path("plots").mkdir(parents=True, exist_ok=True)
    plt.savefig("plots/R9_10_temp.png")

if __name__ == '__main__':
    plot_EXIT()
