import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import hw5.R5_18 as R5_18

import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path
from itertools import permutations
from time import perf_counter

BaseCode = R5_18.HammingCode74

def generate_circulants(size):
    for p in permutations(range(size)):
        yield np.eye(size)[:,p]

def find_girth(adj_mat):
    def search(node, visited):
        if node in visited:
            yield len(visited) - visited.index(node)
            return
        visited.append(node)
        idx, is_CN = node
        if is_CN:
            for j, x in enumerate(adj_mat[idx]):
                if x and (len(visited) < 2 or visited[-2] != (j, False)):
                    yield from search((j, False), visited)
        else:
            for i, x in enumerate(adj_mat[:,idx]):
                if x and (len(visited) < 2 or visited[-2] != (i, True)):
                    yield from search((i, True), visited)
        visited.pop()

    return min((l for i in range(len(adj_mat)) for l in search((i, True), [])), default=None)

def expand_base(base, size, max_iterations=100):
    m, n = base.shape
    ones = [(i, j) for i, row in enumerate(base) for j, val in enumerate(row) if val]
    circulants = [*generate_circulants(size)]
    H = np.zeros((size * m, size * n))
    max_girth = 0
    for _ in range(max_iterations):
        for i, j in ones:
            H[i * size: i * size + size, j * size: j * size + size] = random.choice(circulants)
        girth = find_girth(H)
        if girth > max_girth:
            max_girth = girth
            ret = np.array([row[:] for row in H], dtype=np.uint8)
    print(f'[info] Max girth found: {max_girth}')
    return ret

def plot_BER_vs_SNR(code_expanded):
    print("[info] Computing BER vs SNR curves...", flush=True)
    snrs_base = np.linspace(-1, 8)
    snrs_expanded = np.linspace(-1, 8)
    t0 = perf_counter()
    pbs_base = [R5_18.estimate_BER(snr, code=BaseCode, decoder="SPA", max_iters=int(5e5))
                for snr in snrs_base]
    t1 = perf_counter()
    print(f"[info] Base code BER finished in {t1 - t0:.4f} seconds.")
    pbs_expanded = [R5_18.estimate_BER(snr, code=code_expanded, decoder="SPA", max_iters=int(5e5))
                    for snr in snrs_expanded]
    t2 = perf_counter()
    print(f"[info] Expanded code BER finished in {t2 - t1:.4f} seconds.")
    plt.plot(snrs_base, pbs_base)
    plt.plot(snrs_expanded, pbs_expanded)
    plt.legend(["Base", "Expanded"])
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel(r"$p_b$")
    plt.yscale("log")
    plt.grid(True)
    Path("plots").mkdir(parents=True, exist_ok=True)
    plt.savefig("plots/R6_4_temp.png")
    # plt.show()

if __name__ == '__main__':
    H = expand_base(BaseCode.parity_check_matrix, 3)
    print(H)
    code = R5_18.Code(H, 4 / 7)
    plot_BER_vs_SNR(code)
