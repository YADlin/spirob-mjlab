"""Compact SVG of the reach map (for Notion / docs)."""
import json
from pathlib import Path
R = Path(__file__).resolve().parents[3]
res = json.load(open(R / "docs/knockdown/reach_map.json"))
cells = {}
for r in res: cells.setdefault((r["u"], r["v"]), set()).add(r["status"])
S = 20  # px per cm
X = lambda u: 60 + (u * 100 + 21) * S
Y = lambda v: 30 + (24 - v * 100) * S
W, H = int(X(0.21)) + 20, int(Y(-0.01)) + 70
o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Helvetica,Arial,sans-serif" font-size="11">',
     f'<rect width="{W}" height="{H}" fill="#fcfcfb"/>',
     f'<rect x="{X(-0.0035)}" y="{Y(0.21)}" width="{0.7*S}" height="{21*S}" rx="6" fill="#c3c2b7" opacity=".45"/>',
     f'<text x="{X(0.004)}" y="{Y(0.215)}" fill="#52514e" font-size="10">robot at rest (joint 2 → tip)</text>']
for t in range(-20, 21, 5):
    o.append(f'<line x1="{X(t/100)}" y1="{Y(0.24)}" x2="{X(t/100)}" y2="{Y(-0.01)}" stroke="#e6e5e0"/>'
             f'<text x="{X(t/100)}" y="{Y(-0.01)+14}" text-anchor="middle" fill="#52514e">{t}</text>')
for t in range(0, 25, 5):
    o.append(f'<line x1="{X(-0.21)}" y1="{Y(t/100)}" x2="{X(0.21)}" y2="{Y(t/100)}" stroke="#e6e5e0"/>'
             f'<text x="{X(-0.21)-6}" y="{Y(t/100)+4}" text-anchor="end" fill="#52514e">{t}</text>')
for (u, v), s in cells.items():
    x, y = X(u), Y(v)
    if "knocked" in s: o.append(f'<circle cx="{x}" cy="{y}" r="6" fill="#2a78d6"/>')
    elif "invalid_overlap" in s: o.append(f'<path d="M{x-5} {y-5}L{x+5} {y+5}M{x-5} {y+5}L{x+5} {y-5}" stroke="#52514e" stroke-width="2"/>')
    elif "touched" in s: o.append(f'<path d="M{x} {y-7}L{x+6} {y+5}L{x-6} {y+5}Z" fill="#eb6834"/>')
    else: o.append(f'<circle cx="{x}" cy="{y}" r="2.5" fill="#c3c2b7"/>')
for side in (-1, 1):
    for (a0, a1, v0, v1, dash, col) in ((0.045, 0.105, 0.055, 0.175, "6 4", "#0b0b0b"), (0.05, 0.10, 0.06, 0.17, "", "#52514e")):
        x0 = X(a0 if side > 0 else -a1)
        o.append(f'<rect x="{x0}" y="{Y(v1)}" width="{(a1-a0)*100*S}" height="{(v1-v0)*100*S}" fill="none" stroke="{col}" stroke-width="1.5" stroke-dasharray="{dash}"/>')
P = {"P01": (.055, .07), "P02": (.095, .07), "P03": (.075, .115), "P04": (.095, .115), "P05": (.055, .16), "P06": (.095, .16)}
P.update({f"P{int(k[1:])+6:02d}": (-u, v) for k, (u, v) in list(P.items())})
for k, (u, v) in P.items():
    x, y = X(u), Y(v)
    o.append(f'<path d="M{x-7} {y}H{x+7}M{x} {y-7}V{y+7}" stroke="#0b0b0b" stroke-width="2.2"/><text x="{x+5}" y="{y-5}" font-size="9" fill="#0b0b0b">{k}</text>')
o.append(f'<text x="{W/2}" y="18" text-anchor="middle" font-size="13" fill="#0b0b0b">Where a full 20 mm/s curl knocks the egg off (Lab model, MuJoCo 3.8.1)</text>')
o.append(f'<text x="{W/2}" y="{Y(-0.01)+30}" text-anchor="middle" fill="#0b0b0b">u [cm] lateral (+ = cable_0 curl side)</text>')
o.append(f'<text transform="translate(16 {Y(0.12)}) rotate(-90)" text-anchor="middle" fill="#0b0b0b">v [cm] along rest backbone from joint 2</text>')
ly = Y(-0.01) + 52; lx = 70
for sym, label in (('<circle cx="{x}" cy="{y}" r="6" fill="#2a78d6"/>', "knocked off"), ('<path d="M{x} {y0}L{x6} {y5}L{xm6} {y5}Z" fill="#eb6834"/>', "touched, stayed on"),
                   ('<circle cx="{x}" cy="{y}" r="2.5" fill="#c3c2b7"/>', "not reached"), ('<path d="M{xm5} {ym5}L{x5} {y5}M{xm5} {y5}L{x5} {ym5}" stroke="#52514e" stroke-width="2"/>', "overlaps robot"),
                   ('<rect x="{xm7}" y="{ym5}" width="14" height="10" fill="none" stroke="#0b0b0b" stroke-dasharray="4 3"/>', "training region"),
                   ('<rect x="{xm7}" y="{ym5}" width="14" height="10" fill="none" stroke="#52514e"/>', "evaluation region"),
                   ('<path d="M{xm7} {y}H{x7}M{x} {ym7}V{y7}" stroke="#0b0b0b" stroke-width="2"/>', "hardware placement")):
    o.append(sym.format(x=lx, y=ly, y0=ly-7, x6=lx+6, y5=ly+5, xm6=lx-6, xm5=lx-5, ym5=ly-5, x5=lx+5, xm7=lx-7, x7=lx+7, ym7=ly-7, y7=ly+7))
    o.append(f'<text x="{lx+11}" y="{ly+4}" fill="#52514e">{label}</text>'); lx += 118
o.append("</svg>")
svg = "".join(o)
(R / "docs/knockdown/reach_map.svg").write_text(svg); print(len(svg), W, H)
