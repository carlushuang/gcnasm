"""Analyze partA.csv / partB.csv from nps_interleave (pure Python, no numpy)."""
import csv
import statistics
import sys
from collections import Counter

CPN = float(sys.argv[1]) if len(sys.argv) > 1 else 2.3926


def load(fn):
    with open(fn) as f:
        r = csv.reader(f)
        next(r)
        return [(int(row[0]), [int(v) / CPN for v in row[1:]]) for row in r]


A, B = load('partA.csv'), load('partB.csv')


def pearson(a, b):
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    return cov / (va * vb) ** 0.5


# 1) XCC grouping from correlation of per-line latency vectors
cols = [[lat[x] for _, lat in A] for x in range(8)]
print('XCC latency correlation over part A lines (1 MiB, 128 B granularity):')
corr = [[pearson(cols[i], cols[j]) for j in range(8)] for i in range(8)]
for i in range(8):
    print('  x%d ' % i + ' '.join('%+.2f' % c for c in corr[i]))
g0 = sorted(x for x in range(8) if corr[0][x] > 0.5)
g1 = sorted(x for x in range(8) if x not in g0)
print('groups: G0 =', g0, ' G1 =', g1)


def classify(lat):
    m0 = statistics.fmean(lat[x] for x in g0)
    m1 = statistics.fmean(lat[x] for x in g1)
    return (0 if m0 < m1 else 1), abs(m0 - m1), m0, m1


# 2) latency levels
near, far = [], []
for _, lat in A + B:
    h, margin, m0, m1 = classify(lat)
    near.append(min(m0, m1))
    far.append(max(m0, m1))
q = lambda v, p: sorted(v)[int(len(v) * p)]
print('near-group mean latency: p10 %.0f  p50 %.0f  p90 %.0f ns' % (q(near, .1), q(near, .5), q(near, .9)))
print('far-group  mean latency: p10 %.0f  p50 %.0f  p90 %.0f ns' % (q(far, .1), q(far, .5), q(far, .9)))
margins = [classify(lat)[1] for _, lat in A + B]
print('margin |G0-G1|: p1 %.0f  p10 %.0f  p50 %.0f ns' % (q(margins, .01), q(margins, .1), q(margins, .5)))

# per-XCC near/far level (when its group is near vs far)
for x in range(8):
    mine = g0 if x in g0 else g1
    nv = [lat[x] for _, lat in A if classify(lat)[0] == (0 if mine is g0 else 1)]
    fv = [lat[x] for _, lat in A if classify(lat)[0] != (0 if mine is g0 else 1)]
    print('  x%d: local-home lines %.0f ns, remote-home lines %.0f ns (medians)' % (x, statistics.median(nv), statistics.median(fv)))


# 3) home sequence and run lengths
def runs(seq):
    out, cur, n = [], seq[0], 0
    for s in seq:
        if s == cur:
            n += 1
        else:
            out.append(n)
            cur, n = s, 1
    out.append(n)
    return out


homeA = [classify(lat)[0] for _, lat in A]
print('\npart A home IOD per 128 B line (first 256 lines = 32 KiB):')
s = ''.join(str(h) for h in homeA[:256])
for k in range(0, 256, 64):
    print('  +%6d B: %s' % (k * 128, s[k:k + 64]))
print('  fraction homed on G0: %.3f' % (homeA.count(0) / len(homeA)))
print('  run lengths (in 128 B lines):', Counter(runs(homeA)).most_common(8))

homeB = [classify(lat)[0] for _, lat in B]
print('\npart B home IOD per 4 KiB block (offset 0), 256 MiB:')
print('  fraction homed on G0: %.3f' % (homeB.count(0) / len(homeB)))
print('  run lengths (in 4 KiB blocks):', Counter(runs(homeB)).most_common(8))
print('  first 128 blocks: %s' % ''.join(str(h) for h in homeB[:128]))

# 4) GF(2) fit: home = XOR of offset bits in S (xor const)
def gf2_fit(samples, bits):
    # samples: list of (offset, home); returns (mask, const, violations) using elimination on all samples
    nb = len(bits)
    basis = {}  # pivot -> (row_mask, rhs) ; row_mask over nb+1 vars (last = const)
    inconsistent = 0
    for off, h in samples:
        row = 0
        for i, b in enumerate(bits):
            if (off >> b) & 1:
                row |= 1 << i
        row |= 1 << nb
        rhs = h
        for p in sorted(basis, reverse=True):
            if (row >> p) & 1:
                br, bh = basis[p]
                row ^= br
                rhs ^= bh
        if row == 0:
            if rhs:
                inconsistent += 1
            continue
        p = row.bit_length() - 1
        basis[p] = (row, rhs)
    # back-substitute (free vars = 0)
    sol = 0
    for p in sorted(basis):
        row, rhs = basis[p]
        val = rhs
        r = row & ~(1 << p)
        i = 0
        while r:
            if r & 1 and (sol >> i) & 1:
                val ^= 1
            r >>= 1
            i += 1
        if val:
            sol |= 1 << p
    mask = 0
    for i, b in enumerate(bits):
        if (sol >> i) & 1:
            mask |= 1 << b
    const = (sol >> nb) & 1
    viol = sum(1 for off, h in samples if (bin(off & mask).count('1') + const) % 2 != h)
    return mask, const, inconsistent, viol


confident = [(off, classify(lat)[0]) for off, lat in A + B if classify(lat)[1] > 40]
print('\nGF(2) fit over %d confident samples (margin > 40 ns):' % len(confident))
for hi in (21, 28):
    bits = list(range(7, hi))
    mask, const, inc, viol = gf2_fit(confident, bits)
    used = [b for b in bits if (mask >> b) & 1]
    print('  bits 7..%d: home = parity(offset & 0x%x) ^ %d  -> uses bits %s; inconsistent rows %d; violations %d/%d'
          % (hi - 1, mask, const, used, inc, viol, len(confident)))

# 5) page-to-page comparison of the 4 KiB-level pattern
pages = [homeB[p * 512:(p + 1) * 512] for p in range(len(homeB) // 512)]
same = sum(1 for p in pages if p == pages[0])
comp = sum(1 for p in pages if all(a != b for a, b in zip(p, pages[0])))
print('\n2 MiB pages: %d total; identical to page 0: %d; complement of page 0: %d; other: %d'
      % (len(pages), same, comp, len(pages) - same - comp))
