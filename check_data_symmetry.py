import csv, numpy as np

csv_file = f"simPEG_data/013/137_FWrsp.dat"

d = {}
with open(csv_file) as f:
    r = csv.reader(f)
    next(r); next(r); next(r)
    for row in r:
        d.setdefault((float(row[0]), float(row[1])), {})[row[2]] = np.array(row[3:], float)

gmax = {l: max(np.abs(v[l]).max() for v in d.values()) for l in next(iter(d.values()))}

def err(l1, p1, l2, p2, s):
    return np.abs(d[p1][l1] - s * d[p2][l2]).max() / gmax[l1]

bad = []
for (x, y) in d:
    for suf in ("R", "I"):
        tests = [  # (partner site, label, partner label, sign)
            ((x, -y), "ZXX", "ZXX", -1), ((x, -y), "ZYY", "ZYY", -1),
            ((x, -y), "ZXY", "ZXY", +1), ((x, -y), "ZYX", "ZYX", +1),
            ((x, -y), "TZX", "TZX", +1), ((x, -y), "TZY", "TZY", -1),
            ((-x, y), "ZXX", "ZXX", -1), ((-x, y), "ZYY", "ZYY", -1),
            ((-x, y), "ZXY", "ZXY", +1), ((-x, y), "ZYX", "ZYX", +1),
            ((-x, y), "TZX", "TZX", -1), ((-x, y), "TZY", "TZY", +1),
            ((y, x),  "ZXX", "ZYY", -1), ((y, x),  "ZXY", "ZYX", -1),
            ((y, x),  "TZX", "TZY", +1), ((y, x),  "TZY", "TZX", +1),
        ]
        for p2, l1, l2, s in tests:
            if p2 in d:
                e = err(l1 + suf, (x, y), l2 + suf, p2, s)
                if e > 2e-3:
                    bad.append((x, y, p2, l1 + suf, e))
    for l1, l2 in [("rhoXX", "rhoYY"), ("rhoXY", "rhoYX")]:
        if (y, x) in d and err(l1, (x, y), l2, (y, x), +1) > 2e-3:
            bad.append((x, y, (y, x), l1, err(l1, (x, y), l2, (y, x), +1)))

print(len(bad), "violations")