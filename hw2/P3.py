import P2
import R19
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def transpose(xs):
    n = len(xs)
    ret = [0] * n
    for i in range(n):
        for k, x in enumerate(xs):
            b = x & 1 << (n - 1 - i)
            b >>= n - 1 - i
            b <<= n - 1 - k
            ret[i] |= b
    return ret

def product_decoder(rx, num_iterations=1):
    ret = rx[:]
    for _ in range(num_iterations):
        for _ in 0, 1:
            for i, row in enumerate(ret):
                est = P2.syndrome_decoder(row)
                if est != row:
                    ret[i] = est
            ret = transpose(ret)
    return ret

def test_product_decoder():
    rx = [1 << (14 - i) for i in range(15)]
    tx_est = product_decoder(rx)
    assert all(row == 0 for row in tx_est)
    print("[info] Product decoder correctly decodes identity matrix!")

def estimate_BER(snr_dB, num_decoding_iterations=1, num_acc_errors=int(1e3)):
    rate = 11 ** 2 / 15 ** 2 # Rate for product code of two (15, 11) Hamming codes
    noise_std = (2 * rate * R19.from_dB(snr_dB)) ** -0.5
    y = R19.BI_AWGN(noise_std)
    def bsc(x):
        return 1 if y(x) >= 0 else 0

    pb = 0
    num_samples = 0
    # Suffices to test performance with all zero codeword for linear codes
    while pb < num_acc_errors:
        # All zero codeword mapped to -1 for the BI_AWGN channel
        tx = -1 * np.ones(15 ** 2, dtype=int)
        rx = [*map(P2.bin_iterable_to_int, np.array([*map(bsc, tx)]).reshape(15, 15))]
        tx_est = product_decoder(rx, num_decoding_iterations)
        pb += sum(row.bit_count() for row in tx_est)
        num_samples += 1
    pb /= num_samples * 15 ** 2
    return pb

@P2.measure_exec_time
def plot_BER_vs_SNR():
    print("[info] Computing BER vs SNR curves for product code...", flush=True)
    snrs = np.linspace(-5, 7)
    pbs = [estimate_BER(snr, 1) for snr in snrs]
    pbrs_iter = [estimate_BER(snr, 10) for snr in snrs]
    Path("plots").mkdir(parents=True, exist_ok=True)
    plt.plot(snrs, pbs)
    plt.plot(snrs, pbrs_iter)
    plt.legend(["iterations=1", "iterations=20"])
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel(r"$p_b$")
    plt.yscale("log")
    plt.grid(True)
    plt.savefig("plots/p3_BSC_BER.png")
    # plt.show()

if __name__ == "__main__":
    test_product_decoder()
    plot_BER_vs_SNR()
