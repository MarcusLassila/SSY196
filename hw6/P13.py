import math

def bin_dist_tail(n, N, f):
    return sum(math.comb(N, k) * f ** k * (1 - f) ** (N - k) for k in range(n, N + 1))

def test(err, dv, dc, lmax=10000):
    p, q = err, 0
    for _ in range(lmax):
        q = 0.5 - 0.5 * (1 - 2 * p) ** (dc - 1)
        p = min((1 - err) * bin_dist_tail(t, dv - 1, q) + err * bin_dist_tail(dv - t, dv - 1, q)
                for t in range(0, dv))
        if p < 1e-9:
            return True
    return False

def find_treshold_regular(dv, dc):
    eps = 0
    while test(eps, dv, dc):
        eps += 0.0001
    return eps

if __name__ == '__main__':
    ensambles = [(3, 6), (4, 8), (5, 10)]
    thresholds = [find_treshold_regular(*params) for params in ensambles]
    for param, threshold in zip(ensambles, thresholds):
        print(f'{param}: {threshold:.4f}')
