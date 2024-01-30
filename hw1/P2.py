import R19  # Use some of the code from Ryan/Lin 1.9
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy import special
from time import perf_counter

PARITY_CHECK = (16511, 9103, 5555, 3797)
SYNDROME_MAP = (15, 14, 13, 12, 11, 10, 9, 7, 6, 5, 3, 1, 2, 4, 8)

def qfunc(x):
    return 0.5 * special.erfc(x * 2 ** -0.5)

def bin_iterable_to_int(iter):
    return int(''.join(map(str, iter)), 2)

def measure_exec_time(callable):
    def wrapper(*args, **kwargs):
        t0 = perf_counter()
        ret = callable(*args, **kwargs)
        t1 = perf_counter()
        print(f"[info] {callable.__name__} executed in {t1 - t0:.4f} seconds")
        return ret
    return wrapper

def syndrome_decoder(rx):
    z = bin_iterable_to_int((p & rx).bit_count() & 1 for p in PARITY_CHECK)
    if z:
        rx ^= 1 << SYNDROME_MAP.index(z)
    return rx

def find_codewords():
    '''
    Find all codewords to (15, 11) Hamming code.
    '''
    def search(cw, k):
        if k == 15:
            if all((p & cw).bit_count() & 1 == 0 for p in PARITY_CHECK):
                yield cw
        else:
            yield from search(cw, k + 1)
            yield from search(cw ^ 1 << k, k + 1)
    return set(search(0, 0))

def test_syndrome_decoder():
    codewords = find_codewords()
    assert len(codewords) == 2 ** 11
    for tx in codewords:
        est_tx = syndrome_decoder(tx)
        assert est_tx == tx
        for b in range(15):
            noise = 1 << b
            rx = tx ^ noise
            est_tx = syndrome_decoder(rx)
            assert est_tx == tx
    print("[info] Syndrome decoder corrects <= 1 error on each codeword!")

def estimate_BER(snr_dB, num_samples=int(2e5)):
    rate = 11 / 15 # Rate for (15, 11) Hamming code
    noise_std = (2 * rate * R19.from_dB(snr_dB)) ** -0.5
    y = R19.BI_AWGN(noise_std)
    def bsc(x):
        return 1 if y(x) >= 0 else 0

    pb = 0
    # Suffices to test performance with all zero codeword for linear codes
    desc = f"Simulating decoding with SNR={snr_dB:.2f} (dB)"
    for _ in tqdm(range(num_samples), desc=desc):
        # All zero codeword mapped to -1 for the BI_AWGN channel
        tx = -1 * np.ones(15, dtype=int) 
        rx = bin_iterable_to_int(map(bsc, tx))
        tx_est = syndrome_decoder(rx)
        pb += tx_est.bit_count()
    pb /= num_samples * len(tx)
    return pb

@measure_exec_time
def plot_BER_vs_SNR():
    snrs = np.linspace(-5, 5)
    pbs = [*map(estimate_BER, snrs)]
    uncoded_pbs = [qfunc(np.sqrt(2 * R19.from_dB(snr))) for snr in snrs]
    plt.plot(snrs, pbs)
    plt.plot(snrs, uncoded_pbs)
    plt.legend(["coded", "uncoded"])
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel(r"$p_b$")
    plt.yscale("log")
    plt.grid(True)
    plt.savefig("p2_BSC_rate_05.png")
    # plt.show()

if __name__ == "__main__":
    test_syndrome_decoder()
    plot_BER_vs_SNR()
