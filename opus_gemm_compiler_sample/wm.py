import re
import sys
import collections
import os

OP = re.compile(r"v\[(\d+):\d+\](?:\s*/\*v\[(\d+):\d+\]\*/)?")


def real(m):
    return int(m[1]) if m[1] else int(m[0])


def scan(f):
    rows = []
    for l in open(f):
        if "v_wmma" not in l:
            continue
        body = l.split("//")[0]
        ops = OP.findall(body)
        if len(ops) >= 4:
            rows.append([real(o) for o in ops[:4]])
    return rows


for f in sys.argv[1:]:
    r = scan(f)
    n = len(r)
    name = os.path.basename(f).replace(".dis", "")
    dst = collections.Counter(x[0] for x in r)
    s0 = collections.Counter(x[1] for x in r)
    in_acc = sum(v for k, v in dst.items() if 256 <= k < 512)
    al_d = sum(v for k, v in dst.items() if k % 8 == 0)
    in_b = sum(v for k, v in s0.items() if 512 <= k < 768)
    al_s = sum(v for k, v in s0.items() if k % 8 == 0)
    print(f"--- {name}  (wmma={n}) ---")
    print(f"  dst : {len(dst):>3} 个不同起点  v{min(dst)}-v{max(dst)}"
          f"  在v256-511={in_acc}  8对齐={al_d}")
    print(f"  src0: {len(s0):>3} 个不同起点  v{min(s0)}-v{max(s0)}"
          f"  在v512-767={in_b}  8对齐={al_s}")
    w = collections.Counter(len({o // 256 for o in x}) for x in r)
    print(f"  单条wmma跨越的256窗口数: {dict(sorted(w.items()))}")
