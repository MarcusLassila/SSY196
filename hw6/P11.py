def zero_convergence_regular(err, dv, dc, lmax=1000):
    p, q = err, 0
    for _ in range(lmax):
        q = 1 - (1 - p) ** (dc - 1)
        p = err * q ** (dv - 1)
        if p < 1e-9:
            return True
    return False

def find_bp_threshold_regular(dv, dc):
    err = 0
    while zero_convergence_regular(err, dv, dc):
        err += 0.0001
    return err

def zero_convergence_irregular(err, lam, rho, lmax=1000):
    p, q = err, 0
    for _ in range(lmax):
        q = 1 - rho(1 - p)
        p = err * lam(q)
        if p < 1e-9:
            return True
    return False

def find_bp_threshold_irregular(lam, rho):
    err = 0
    while zero_convergence_irregular(err, lam, rho):
        err += 0.0001
    return err

if __name__ == '__main__':
    lam = lambda x: 0.239 * x + 0.295 * x ** 2 + 0.033 * x ** 3 + 0.433 * x ** 10
    rho = lambda x: 0.430 * x ** 6 + 0.570 * x ** 7
    ensambles = [(3, 6), (4, 8), (5, 10)]
    for dv, dc in ensambles:
        threshold = find_bp_threshold_regular(dv, dc)
        print(f'({dv}, {dc}): {threshold:.04f}')
    print(f'dv11: {find_bp_threshold_irregular(lam, rho):.4f}')
