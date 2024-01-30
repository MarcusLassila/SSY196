import R19  # Use some of the code from Ryan/Lin 1.9
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy import special
from time import perf_counter

copy_pasted_numbers = '''
1 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 1 0 0 0 1 1 1 0 0 0 1 1 1 1
0 0 1 0 1 0 1 1 0 1 1 0 0 1 1 0 0 0 1 1 1 0 1 1 0 1 0 1 0 1
'''.split()
#
# Parity matrix only used for testing the decoder!!!
#
H = np.array([*map(int, copy_pasted_numbers)]).reshape(4, 15)

def qfunc(x):
    return 0.5 * special.erfc(x * 2 ** -0.5)

def measure_exec_time(callable):
    def wrapper(*args, **kwargs):
        t0 = perf_counter()
        ret = callable(*args, **kwargs)
        t1 = perf_counter()
        print(f"[info] {callable.__name__} executed in {t1 - t0:.4f} seconds")
        return ret
    return wrapper

def sparse_syndrome_decoder(rx):
    '''
    Not using the full parity check matrix, only nonzero entries.
    '''
    indices = [
        [0, 8, 9, 10, 11, 12, 13, 14],
        [1, 5, 6, 7, 11, 12, 13, 14],
        [2, 4, 6, 7, 9, 10, 13, 14],
        [3, 4, 5, 7, 8, 10, 12, 14],
    ]
    tx_est = rx.copy() # Avoid mutating input
    z = [tx_est[js].sum() % 2 for js in indices]
    deduce = set(range(15))
    for (b, js) in zip(z, indices):
        if b:
            deduce.intersection_update(js)
        else:
            deduce.difference_update(js)
    if deduce:
        tx_est[deduce.pop()] ^= 1
    return tx_est

def syndrome_decoder(rx):
    '''
    Using the full parity check matrix just for comparison.
    '''
    tx_est = rx.copy() # Avoid mutating input
    z = H @ tx_est % 2
    p = np.where(np.all(H.T == z, axis=1))
    if p[0]:
        tx_est[p[0][0]] ^= 1
    return tx_est

def find_codewords():
    '''
    Find all codewords to (15, 11) Hamming code.
    '''
    def search(cw, k):
        if k == 15:
            if np.all(H @ cw % 2 == 0):
                yield cw.copy()
        else:
            cw[k] = 0
            yield from search(cw, k + 1)
            cw[k] = 1
            yield from search(cw, k + 1)
    return [*search(np.zeros(15, dtype=int), 0)]

def test_syndrome_decoder():
    codewords = find_codewords()
    assert len(codewords) == 2 ** 11
    for tx in codewords:
        noise = np.zeros(15, dtype=int)
        rx = (tx + noise) % 2
        est_tx = sparse_syndrome_decoder(rx)
        assert np.array_equal(tx, est_tx)
        for b in range(15):
            noise[b] = 1
            rx = (tx + noise) % 2
            est_tx = sparse_syndrome_decoder(rx)
            assert np.array_equal(tx, est_tx)
            noise[b] = 0
    print("Syndrome decoder corrects <= 1 error on each codeword!")

def estimate_BER(snr_dB, num_samples=int(2e5)):
    rate = 11 / 15 # Rate for (15, 11) Hamming code
    noise_std = (2 * rate * R19.from_dB(snr_dB)) ** -0.5
    y = R19.BI_AWGN(noise_std)
    def bsc(x):
        return 1 if y(x) >= 0 else 0

    pb = 0
    zero_cw = np.zeros(15) # Suffices to test performance with all zero codeword for linear codes
    for _ in tqdm(range(num_samples), desc=f"Simulating decoding with SNR={snr_dB:.2f} (dB)"):
        tx = -1 * np.ones(15, dtype=int) # All zero codeword mapped to -1 for the BI_AWGN channel
        rx = np.array([*map(bsc, tx)])
        tx_est = sparse_syndrome_decoder(rx)
        pb += np.sum(tx_est != zero_cw)
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
