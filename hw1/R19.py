import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

def to_dB(x):
    return 10 * np.log10(x)

def from_dB(x):
    return 10 ** (x / 10)

def bin_entropy(x):
    return x * np.log2(1 / x) + (1 - x) * np.log2(1 / (1 - x))

def inv_bin_entropy(x):
    '''
    Inverse binary entropy function computed by binary search.
    Return value in range [0, 0.5].
    '''
    eps = 1e-6
    low, high = 0, 0.5
    while low + eps < high:
        mid = (low + high) / 2
        entr = bin_entropy(mid)
        if abs(entr - x) < eps:
            break
        if entr < x:
            low = mid
        else:
            high = mid
    return mid

class BI_AWGN:

    def __init__(self, noise_std):
        self.noise_std = noise_std

    def __call__(self):
        loc = 1.0 if np.random.random() < 0.5 else -1.0
        return np.random.normal(loc=loc, scale=self.noise_std)

    def density(self, x):
        scale = 1 / np.sqrt(2 * np.pi) / self.noise_std
        pos = scale * np.exp(-(x - 1) ** 2 / (2 * self.noise_std ** 2))
        neg = scale * np.exp(-(x + 1) ** 2 / (2 * self.noise_std ** 2))
        return 0.5 * (pos + neg)

    def capacity(self, num_samples=int(1e7)):
        monte_carlo = -sum(np.log2(self.density(self.__call__())) for _ in range(num_samples)) / num_samples
        guassian_entropy = -0.5 * np.log2(2 * np.pi * np.e * self.noise_std ** 2)
        return monte_carlo + guassian_entropy


rate = 0.1

def error_prob(snr_dB):
    noise_std = (2 * rate * from_dB(snr_dB)) ** -0.5
    bin_entr = 1 - BI_AWGN(noise_std).capacity() / rate
    return inv_bin_entropy(bin_entr)

snrs = np.linspace(-2.0, -1.3)
pbs  = [*map(error_prob, tqdm(snrs, desc=f"Computing figure 1.8 with rate={rate}"))]

plt.plot(snrs, pbs)
plt.title(rf"$p_b$ vs $E_b/N_0$ with $R={rate}$")
plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel(r"$p_b$")
plt.yscale('log')
plt.grid(True)
plt.show()
