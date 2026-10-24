"""Figure: scripted-curl reachability map + training/evaluation regions + hardware placements."""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
R = Path(__file__).resolve().parents[3]
res = json.load(open(R / "docs/knockdown/reach_map.json"))
sys.path.insert(0, str(R / "src"))
from spirob_mjlab.knockdown.env_cfg import TRAIN_ABS_U, TRAIN_V, EVAL_ABS_U, EVAL_V
PL = {"P01": (.055, .07), "P02": (.095, .07), "P03": (.075, .115), "P04": (.095, .115), "P05": (.055, .16), "P06": (.095, .16)}
PL.update({f"P{int(k[1:])+6:02d}": (-u, v) for k, (u, v) in PL.items()})
cells = {}
for r in res:
    k = (r["u"], r["v"]); s = cells.setdefault(k, set()); s.add(r["status"])
style = {"knocked": dict(c="#2a78d6", m="o", label="Knocked off (either curl direction)"),
         "touched": dict(c="#eb6834", m="^", label="Touched, stayed on pedestal"),
         "missed": dict(c="#c3c2b7", m=".", label="Not reached"),
         "invalid_overlap": dict(c="#52514e", m="x", label="Overlaps robot at rest (invalid)")}
fig, ax = plt.subplots(figsize=(7.2, 6.4), facecolor="#fcfcfb"); ax.set_facecolor("#fcfcfb")
for key, st in style.items():
    pts = [k for k, s in cells.items() if (key in s if key != "knocked" else "knocked" in s) and
           (key == "knocked" or ("knocked" not in s and (key == "invalid_overlap" or "invalid_overlap" not in s) and
            (key != "missed" or "touched" not in s)))]
    if pts:
        ax.scatter([p[0]*100 for p in pts], [p[1]*100 for p in pts], c=st["c"], marker=st["m"],
                   s=46 if st["m"] != "." else 30, label=st["label"], zorder=3, linewidths=1.5)
for side in (-1, 1):
    a0, a1 = TRAIN_ABS_U; x0 = a0*100 if side > 0 else -a1*100
    ax.add_patch(Rectangle((x0, TRAIN_V[0]*100), (a1-a0)*100, (TRAIN_V[1]-TRAIN_V[0])*100, fill=False, ls="--", lw=1.2, ec="#0b0b0b", zorder=2,
                           label="Training region" if side > 0 else None))
    a0, a1 = EVAL_ABS_U; x0 = a0*100 if side > 0 else -a1*100
    ax.add_patch(Rectangle((x0, EVAL_V[0]*100), (a1-a0)*100, (EVAL_V[1]-EVAL_V[0])*100, fill=False, lw=1.2, ec="#52514e", zorder=2,
                           label="Held-out evaluation region" if side > 0 else None))
for k, (u, v) in PL.items():
    ax.plot(u*100, v*100, marker="+", ms=11, mew=1.8, color="#0b0b0b", zorder=4)
    ax.annotate(k, (u*100, v*100), xytext=(4, 4), textcoords="offset points", fontsize=7, color="#52514e")
ax.plot([0, 0], [0, 21], color="#52514e", lw=6, alpha=.25, solid_capstyle="round", zorder=1)
ax.annotate("robot at rest (joint 2 → tip)", (0, 21), xytext=(3, 0), textcoords="offset points", fontsize=7, color="#52514e")
ax.set_xlabel("u [cm]  lateral, + = cable_0 curl side"); ax.set_ylabel("v [cm]  along rest backbone from joint 2")
ax.set_xlim(-21, 21); ax.set_ylim(-1, 24); ax.set_aspect("equal"); ax.grid(color="#e6e5e0", lw=.6); ax.set_axisbelow(True)
for s in ax.spines.values(): s.set_color("#c3c2b7")
ax.set_title("Where a full 20 mm/s curl knocks the egg off (Lab model, MuJoCo 3.8.1)", fontsize=10, color="#0b0b0b")
ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False)
fig.tight_layout(); fig.savefig(R / "docs/knockdown/reach_map.png", dpi=160)
plt.rcParams["svg.fonttype"] = "none"; fig.savefig(R / "docs/knockdown/reach_map.svg")
print("saved")
