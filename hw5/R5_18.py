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
    
    @staticmethod
    def to_01(xs):
        return np.select([xs == -1, xs == 1], [1, 0], xs)
    
    @staticmethod
    def from_01(xs):
        return np.select([xs == 0, xs == 1], [1, -1], xs)

class HammingCode74:
    parity_check_matrix = np.array("1 0 1 0 1 0 1 0 1 1 0 0 1 1 0 0 0 1 1 1 1".split(),
                                   dtype=np.uint8).reshape(3, 7)
    rate = 4 / 7

class VN:

    def __init__(self):
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
        self.VNs = [VN() for _ in range(len(parity_check_matrix[0]))]
        for i, row in enumerate(parity_check_matrix):
            for j, val in enumerate(row):
                if val:
                    self.CNs[i].var_nodes[self.VNs[j]] = 0
                    self.VNs[j].check_nodes[self.CNs[i]] = 0

class SPA_Decoder:

    def __init__(self, code, channel, max_decode_iters=10):
        self.code = code
        self.channel = channel
        self.graph = TannerGraph(self.code.parity_check_matrix)
        self.max_decode_iters = max_decode_iters

    def decode(self, rx):
        assert len(rx) == len(self.code.parity_check_matrix[0])
        self.initialize(rx)
        for _ in range(self.max_decode_iters):
            self.update_CNs()
            self.update_VNs()
            pred = self.prediction()
            if not np.any(self.code.parity_check_matrix @ pred % 2):
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
            for cn in vn.check_nodes.keys():
                s = sum(x.var_nodes[vn] for x in vn.check_nodes.keys() if x != cn)
                vn.check_nodes[cn] = vn.val + s
    
    def prediction(self):
        totals = [vn.val + sum(cn.var_nodes[vn] for cn in vn.check_nodes.keys())
                  for vn in self.graph.VNs]
        return np.array([1 if x < 0 else 0 for x in totals])

class MinDist_Decoder:
    
    def __init__(self, code):
        self.code = code
        self.size = len(code.parity_check_matrix[1])
        
    def generate_codewords(self):
        def search(k, cw):
            if k == self.size:
                if not np.any(self.code.parity_check_matrix @ cw % 2):
                    yield cw.copy()
                return
            yield from search(k + 1, cw)
            cw[k] = (cw[k] + 1) % 2
            yield from search(k + 1, cw)
        return search(0, np.zeros(self.size, dtype=np.int8))
            
    def decode(self, rx):
        assert len(rx) == self.size
        best_dist = float('inf')
        best_cw = None
        for cw in map(BI_AWGN.from_01, self.generate_codewords()):
            dist = math.dist(rx, cw)
            if dist < best_dist:
                best_dist = dist
                best_cw = cw
        return BI_AWGN.to_01(best_cw)

def estimate_BER(snr_dB,
                 code=HammingCode74,
                 decoder="SPA",
                 num_frame_errors=int(2e3),
                 max_iters=int(5e5),
                 spa_max_iters=10):
    np.random.seed(0)
    size = code.parity_check_matrix.shape[1]
    noise_std = (2 * code.rate * from_dB(snr_dB)) ** -0.5
    channel = BI_AWGN(noise_std)
    if decoder == "SPA":
        decoder = SPA_Decoder(code, channel, max_decode_iters=spa_max_iters)
    elif decoder == "MIN_DIST":
        decoder = MinDist_Decoder(code)
    else:
        raise ValueError("Unsupported decoder")
    frame_errors = 0
    pb = 0
    num_samples = 0
    time_out_counter = 0
    while frame_errors < num_frame_errors and time_out_counter < max_iters:
        time_out_counter += 1
        tx = np.ones(size, dtype=np.uint8)  # All zero codeword
        rx = [*map(channel, tx)]
        tx_est = decoder.decode(rx)
        frame_errors += np.any(tx_est)
        pb += tx_est.sum()
        num_samples += 1
    pb /= num_samples * size
    return pb

def plot_BER_vs_SNR():
    print("[info] Computing BER vs SNR curves...", flush=True)
    snrs_spa = np.linspace(-1, 8)
    snrs_min_dist = np.linspace(-1, 8)
    pbs_spa = [estimate_BER(snr, decoder="SPA") for snr in snrs_spa]
    pbs_min_dist = [estimate_BER(snr, decoder="MIN_DIST") for snr in snrs_min_dist]
    plt.plot(snrs_spa, pbs_spa)
    plt.plot(snrs_min_dist, pbs_min_dist)
    plt.legend(["SPA", "MIN DIST"])
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel(r"$p_b$")
    plt.yscale("log")
    plt.grid(True)
    Path("plots").mkdir(parents=True, exist_ok=True)
    plt.savefig("plots/p5_18_BER_vs_SNR_temp.png")
    # plt.show()

def plot_BER_vs_SNR_spa_iteration_sweep():
    print("[info] Computing BER vs SNR curves...", flush=True)
    snrs_spa = np.linspace(-1, 8)
    snrs_min_dist = np.linspace(-1, 8)
    legend = []
    for max_iters in 1, 2, 4, 8, 16:
        legend.append(f"SPA {max_iters}")
        pbs_spa = [estimate_BER(snr, decoder="SPA", spa_max_iters=max_iters)
                   for snr in snrs_spa]
        plt.plot(snrs_spa, pbs_spa)
    legend.append("MIN DIST")
    pbs_min_dist = [estimate_BER(snr, decoder="MIN_DIST") for snr in snrs_min_dist]
    plt.plot(snrs_min_dist, pbs_min_dist)
    plt.legend(legend)
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel(r"$p_b$")
    plt.yscale("log")
    plt.grid(True)
    Path("plots").mkdir(parents=True, exist_ok=True)
    plt.savefig("plots/p5_18_BER_vs_SNR_iters_temp.png")
    # plt.show()

if __name__ == "__main__":
    plot_BER_vs_SNR()
    #plot_BER_vs_SNR_spa_iteration_sweep()
