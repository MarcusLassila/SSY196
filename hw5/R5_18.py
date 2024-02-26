import numpy as np
import matplotlib.pyplot as plt
import math
from collections import defaultdict
from pathlib import Path

def to_dB(x):
    return 10 * np.log10(x)

def from_dB(x):
    return 10 ** (x / 10)

def atanh_safe(x):
    limit = 1 - 1e-9
    return math.atanh(min(max(x, -limit), limit))

class BI_AWGN:

    def __init__(self, noise_std):
        self.noise_std = noise_std

    def __call__(self, x):
        assert x in (-1, 1)
        return np.random.normal(loc=x, scale=self.noise_std)
    
    def LLR(self, y):
        return 2 * y / self.noise_std ** 2

class HammingCode74:
    parity_check_matrix = np.array("1 0 1 0 1 0 1 0 1 1 0 0 1 1 0 0 0 1 1 1 1".split(),
                                   dtype=np.uint8).reshape(3, 7)
    rate = 4 / 7

class VN:

    def __init__(self, idx):
        self.idx = idx
        self.val = None
        self.check_nodes = defaultdict(float)

    def __str__(self):
        return f"X{self.idx}"

class CN:

    def __init__(self):
        self.var_nodes = defaultdict(float)

    def __str__(self):
        return " + ".join(f"X{vn.idx}" for vn in self.var_nodes) + " = 0"

class TannerGraph:

    def __init__(self, parity_check_matrix):
        self.CNs = [CN() for _ in range(len(parity_check_matrix))]
        self.VNs = [VN(i) for i in range(len(parity_check_matrix[0]))]
        for i, row in enumerate(parity_check_matrix):
            for j, val in enumerate(row):
                if val:
                    self.CNs[i].var_nodes[self.VNs[j]] = 0
                    self.VNs[j].check_nodes[self.CNs[i]] = 0

class SPA_Decoder:

    def __init__(self, channel, code):
        self.channel = channel
        self.code = code
        self.graph = TannerGraph(self.code.parity_check_matrix)

    def decode(self, rx):
        assert len(rx) == len(self.code.parity_check_matrix[0])
        self.initialize(rx)
        for _ in range(10):
            self.update_CNs()
            self.update_VNs()
            pred = self.prediction()
            if not np.any(self.code.parity_check_matrix @ pred):
                break
        return pred

    def initialize(self, rx):
        for vn, y in zip(self.graph.VNs, rx):
            vn.val = self.channel.LLR(y)
            for cn in vn.check_nodes.keys():
                vn.check_nodes[cn] = vn.val

    def update_CNs(self):
        for cn in self.graph.CNs:
            for vn in cn.var_nodes.keys():
                p = math.prod(math.tanh(0.5 * x.check_nodes[cn])
                              for x in cn.var_nodes.keys()
                              if x != vn)
                cn.var_nodes[vn] = 2 * atanh_safe(p)

    def update_VNs(self):
        for vn in self.graph.VNs:
            for cn, msg in vn.check_nodes.items():
                s = sum(cn_.var_nodes[vn] for cn_ in vn.check_nodes.keys()) - msg
                vn.check_nodes[cn] = vn.val + s
    
    def prediction(self):
        totals = [vn.val + sum(cn.var_nodes[vn] for cn in vn.check_nodes.keys())
                  for vn in self.graph.VNs]
        return np.array([1 if x < 0 else 0 for x in totals])

def estimate_BER(snr_dB, num_acc_errors=int(1e4), max_iterations=int(1e6)):
    code = HammingCode74
    size = code.parity_check_matrix.shape[1]
    noise_std = (2 * code.rate * from_dB(snr_dB)) ** -0.5
    channel = BI_AWGN(noise_std)
    decoder = SPA_Decoder(channel, code)
    pb = 0
    num_samples = 0
    time_out_counter = 0
    while pb < num_acc_errors and time_out_counter < max_iterations:
        time_out_counter += 1
        tx = np.ones(size, dtype=np.uint8) 
        rx = [*map(channel, tx)]
        tx_est = decoder.decode(rx)
        pb += tx_est.sum()
        num_samples += 1
    pb /= num_samples * size
    return pb

def plot_BER_vs_SNR():
    print("[info] Computing BER vs SNR curves...", flush=True)
    snrs = np.linspace(-5, 9)
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
