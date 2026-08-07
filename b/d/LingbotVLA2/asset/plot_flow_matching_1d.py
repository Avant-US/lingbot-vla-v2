#!/usr/bin/env python3
"""1D flow-matching intuition: x_t = t*noise + (1-t)*action, u_t = noise - action."""

from pathlib import Path

OUT = Path(__file__).with_name("flow_matching_1d.svg")


def main() -> None:
    # Fixed toy scalars for a clean illustration
    a, eps = 0.2, 1.0  # action, noise
    # x(t) = t*eps + (1-t)*a = a + t*(eps-a)
    # u = eps - a = 0.8 (constant)

    w, h = 720, 420
    ml, mr, mt, mb = 70, 40, 50, 60
    pw, ph = w - ml - mr, h - mt - mb

    def sx(t: float) -> float:
        return ml + t * pw

    def sy(x: float) -> float:
        # map x in [-0.1, 1.2] to plot
        lo, hi = -0.1, 1.2
        return mt + (1 - (x - lo) / (hi - lo)) * ph

    # sample polyline
    n = 40
    pts = []
    for i in range(n + 1):
        t = i / n
        x = t * eps + (1 - t) * a
        pts.append(f"{sx(t):.1f},{sy(x):.1f}")

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#FAFAFA"/>',
        '<text x="360" y="28" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" '
        'font-size="16" font-weight="700" fill="#222">Flow Matching (1D intuition)</text>',
        # axes
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="#333" stroke-width="1.5"/>',
        f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="#333" stroke-width="1.5"/>',
        f'<text x="{ml+pw/2}" y="{h-18}" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" '
        'font-size="13" fill="#333">t (noise ← → action)</text>',
        f'<text x="18" y="{mt+ph/2}" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" '
        'font-size="13" fill="#333" transform="rotate(-90 18 '
        f'{mt+ph/2})">x_t</text>',
        # path
        f'<polyline points="{" ".join(pts)}" fill="none" stroke="#4C78A8" stroke-width="3"/>',
        # endpoints
        f'<circle cx="{sx(0)}" cy="{sy(a)}" r="7" fill="#54A24B"/>',
        f'<circle cx="{sx(1)}" cy="{sy(eps)}" r="7" fill="#E45756"/>',
        f'<text x="{sx(0)+10}" y="{sy(a)-8}" font-family="DejaVu Sans,Arial,sans-serif" '
        f'font-size="12" fill="#54A24B">action a={a}</text>',
        f'<text x="{sx(1)-120}" y="{sy(eps)-10}" font-family="DejaVu Sans,Arial,sans-serif" '
        f'font-size="12" fill="#E45756">noise ε={eps}</text>',
        # velocity arrows at a few t
    ]

    u = eps - a
    for t in (0.25, 0.5, 0.75):
        x = t * eps + (1 - t) * a
        # arrow along increasing t, proportional to u (vertical component of path)
        x2 = (t + 0.08) * eps + (1 - (t + 0.08)) * a
        parts.append(
            f'<line x1="{sx(t):.1f}" y1="{sy(x):.1f}" x2="{sx(t+0.08):.1f}" y2="{sy(x2):.1f}" '
            f'stroke="#F58518" stroke-width="2.5" marker-end="url(#arrow)"/>'
        )

    parts.insert(
        3,
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">'
        '<path d="M0,0 L6,3 L0,6 Z" fill="#F58518"/></marker></defs>',
    )

    # formulas box
    parts.append(
        f'<rect x="{ml+20}" y="{mt+20}" width="300" height="78" fill="#fff" stroke="#ddd" rx="6"/>'
    )
    parts.append(
        f'<text x="{ml+35}" y="{mt+42}" font-family="DejaVu Sans,Arial,sans-serif" font-size="12" fill="#222">'
        "x_t = t·ε + (1−t)·a</text>"
    )
    parts.append(
        f'<text x="{ml+35}" y="{mt+62}" font-family="DejaVu Sans,Arial,sans-serif" font-size="12" fill="#222">'
        f"u_t = ε − a = {u:.1f} (train target)</text>"
    )
    parts.append(
        f'<text x="{ml+35}" y="{mt+82}" font-family="DejaVu Sans,Arial,sans-serif" font-size="12" fill="#666">'
        "Inference: Euler from t=1→0, x ← x + dt·v_θ</text>"
    )

    # tick labels
    for t, lab in ((0, "0 (clean)"), (1, "1 (noise)")):
        parts.append(
            f'<text x="{sx(t)}" y="{mt+ph+22}" text-anchor="middle" '
            f'font-family="DejaVu Sans,Arial,sans-serif" font-size="11" fill="#444">{lab}</text>'
        )

    parts.append("</svg>")
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
