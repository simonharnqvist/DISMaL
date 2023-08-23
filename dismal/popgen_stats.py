def estimate_pi(s1, s2):

    pis = []
    for s in [s1, s2]:
        d = dict(zip([i for i in range(0, len(s))], s))

        total = 0
        count = 0
        for k, v in d.items():
            total += k * v
            count += v
        pi = total / count
        pis.append(pi)
    return (pis[0] + pis[1])/2

def estimate_dxy(s3, block_len):

    d = dict(zip([i for i in range(0, len(s3))], s3))
    total = 0
    count = 0
    for k, v in d.items():
        total += k * v
        count += v
    n_differences = total / count
    dxy = n_differences/block_len

    return dxy