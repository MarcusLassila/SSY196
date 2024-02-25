import numpy as np
import matplotlib.pyplot as plt
import math
from collections import defaultdict
from pathlib import Path

def to_dB(x):
    return 10 * np.log10(x)

def from_dB(x):
    return 10 ** (x / 10)

def phi(x):
    ex = math.exp(x)
    try:
        return math.log((ex + 1) / (ex - 1))
    except ZeroDivisionError:
        return 10.0

def sign_and_magnitude(x):
    sign = 1 if x >= 0 else -1
    return sign, abs(x)

class BI_AWGN:

    def __init__(self, noise_std):
        self.noise_std = noise_std

    def __call__(self, x):
        assert x in (-1, 1)
        return np.random.normal(loc=x, scale=self.noise_std)
    
    def LLR(self, y):
        return 2 * y / self.noise_std ** 2

class VN:

    def __init__(self, idx):
        self.idx = idx
        self.val = 0
        self.connections = defaultdict(float)

    def __str__(self):
        return f"X{self.idx}"

    def __hash__(self):
        return hash(repr(self))

class CN:

    def __init__(self):
        self.connections = defaultdict(float)

    def __str__(self):
        return " + ".join(f"X{vn.idx}" for vn in self.connections) + " = 0"

    def __hash__(self):
        return hash(repr(self))

class TannerGraph:

    def __init__(self, H):
        self.H = np.array([row[:] for row in H])
        self.CNs = [CN() for _ in range(len(H))]
        self.VNs = [VN(i) for i in range(len(H[0]))]
        for i, row in enumerate(self.H):
            for j, val in enumerate(row):
                if val:
                    self.CNs[i].connections[self.VNs[j]] = 0
                    self.VNs[j].connections[self.CNs[i]] = 0

def spa_decoder(rx, channel):
    H = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint8).reshape(3, 7)
    assert len(rx) == len(H[0])
    graph = TannerGraph(H)

    def initialize():
        for vn, y in zip(graph.VNs, rx):
            vn.val = channel.LLR(y)
            for cn in vn.connections.keys():
                vn.connections[cn] = vn.val
    
    def update_CN():
        for cn in graph.CNs:
            for vn in cn.connections.keys():
                p = 1
                s = 0
                for vn_ in cn.connections.keys():
                    if vn != vn_:
                        sign, mag = sign_and_magnitude(vn_.connections[cn])
                        s += phi(mag)
                        p *= sign
                r = p * phi(s)
                # r = math.prod(math.tanh(0.5 * vn_.connections[cn])  if vn != vn_)
                # if r == 1.0:
                #     r = 0.99
                # elif r == -1.0:
                #     r = -0.99
                cn.connections[vn] = r # 2 * math.atanh(r)
    
    def update_VN():
        for vn in graph.VNs:
            for cn, msg in vn.connections.items():
                s = sum(cn_.connections[vn] for cn_ in vn.connections.keys()) - msg
                vn.connections[cn] = vn.val + s
    
    def prediction():
        totals = [vn.val + sum(cn.connections[vn] for cn in vn.connections.keys()) for vn in graph.VNs]
        return np.array([1 if x < 0 else 0 for x in totals])

    initialize()
    for _ in range(20):
        update_CN()
        update_VN()
        pred = prediction()
        if not np.any(H @ pred):
            break
    
    return pred

def estimate_BER(snr_dB, num_acc_errors=int(1e2)):
    rate = 4 / 7
    noise_std = (2 * rate * from_dB(snr_dB)) ** -0.5
    channel = BI_AWGN(noise_std)

    pb = 0
    num_samples = 0
    while pb < num_acc_errors:
        tx = np.ones(7, dtype=np.uint8) 
        rx = [*map(channel, tx)]
        tx_est = spa_decoder(rx, channel)
        pb += tx_est.sum()
        num_samples += 1
    pb /= num_samples * 7
    return pb

def plot_BER_vs_SNR():
    print("[info] Computing BER vs SNR curves...", flush=True)
    snrs = np.linspace(-5, 8)
    pbs = [*map(estimate_BER, snrs)]
    Path("plots").mkdir(parents=True, exist_ok=True)
    plt.plot(snrs, pbs)
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel(r"$p_b$")
    plt.yscale("log")
    plt.grid(True)
    plt.savefig("plots/p5_18_BER_vs_SNR_temp.png")
    # plt.show()

if __name__ == "__main__":
    plot_BER_vs_SNR()
