def test(err, dv, dc, lmax=1000):
    p, q = err, 0
    for _ in range(lmax):
        q = 0.5 - 0.5 * (1 - 2 * p) ** (dc - 1)
        p = (1 - err) * q ** (dv - 1) + err * (1 - (1 - q) ** (dv - 1))
        if p < 1e-9:
            return True
    return False

def find_treshold_regular(dv, dc):
    err = 0
    while test(err, dv, dc):
        err += 0.0001
    return err

if __name__ == '__main__':
    ensambles = [(3, 6), (4, 8), (5, 10)]
    thresholds = [find_treshold_regular(*params) for params in ensambles]
    for param, threshold in zip(ensambles, thresholds):
        print(f'{param}: {threshold:.4f}')
