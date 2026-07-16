#!/usr/bin/env python3
"""Token-level MoE: shared expert + top-k routed experts (stdlib SVG)."""

from pathlib import Path

OUT = Path(__file__).with_name("moe_routing.svg")


def box(x, y, w, h, fill, label, sub="", stroke="#333"):
    lines = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="1.5" rx="8"/>',
        f'<text x="{x+w/2}" y="{y+h/2-2}" text-anchor="middle" '
        f'font-family="DejaVu Sans,Arial,sans-serif" font-size="13" font-weight="600" fill="#222">{label}</text>',
    ]
    if sub:
        lines.append(
            f'<text x="{x+w/2}" y="{y+h/2+16}" text-anchor="middle" '
            f'font-family="DejaVu Sans,Arial,sans-serif" font-size="11" fill="#555">{sub}</text>'
        )
    return "\n".join(lines)


def arrow(x1, y1, x2, y2, color="#555"):
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2" '
        f'marker-end="url(#arr)"/>'
    )


def main() -> None:
    w, h = 920, 520
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#FAFAFA"/>',
        '<defs><marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">'
        '<path d="M0,0 L6,3 L0,6 Z" fill="#555"/></marker></defs>',
        '<text x="460" y="32" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" '
        'font-size="17" font-weight="700" fill="#222">Token-level Sparse MoE in Action Expert</text>',
        '<text x="460" y="54" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" '
        'font-size="12" fill="#666">robotwin.yaml: N_r=32, K=4, sigmoid router, λ=4.0, shared expert always on</text>',
        box(40, 100, 140, 70, "#D6EAF8", "token u", "hidden state"),
        box(240, 100, 160, 70, "#FCF3CF", "Router (FP32)", "sigmoid + bias"),
        box(460, 80, 200, 50, "#D5F5E3", "Shared Expert E^(s)", "all tokens"),
    ]

    # routed experts grid
    ex_w, ex_h = 70, 44
    start_x, start_y = 460, 160
    selected = {0, 3, 7, 11}  # top-k illustration among first 12 shown
    for i in range(12):
        r, c = divmod(i, 6)
        x = start_x + c * (ex_w + 12)
        y = start_y + r * (ex_h + 14)
        fill = "#FADBD8" if i in selected else "#EAECEE"
        stroke = "#E74C3C" if i in selected else "#ABB2B9"
        label = f"E{i}" if i < 11 else "…"
        parts.append(box(x, y, ex_w, ex_h, fill, label, "" if i < 11 else "E31", stroke=stroke))

    parts.append(
        '<text x="670" y="280" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" '
        'font-size="12" fill="#C0392B">Top-K=4 selected (red)</text>'
    )

    parts.append(box(40, 360, 200, 70, "#E8DAEF", "MoE output m(u)", "shared + λ Σ g_j E_j"))
    parts.append(arrow(180, 135, 240, 135))
    parts.append(arrow(400, 120, 460, 105))
    parts.append(arrow(400, 145, 460, 180))
    parts.append(arrow(140, 170, 140, 360))
    parts.append(arrow(560, 270, 200, 360))

    # formula panel
    parts.append(
        '<rect x="280" y="340" width="600" height="140" fill="#fff" stroke="#ddd" rx="8"/>'
    )
    formulas = [
        "s_j = Sigmoid(uᵀ e_j)     (affinity; not softmax)",
        "R(u) = TopK(s_j + b_j, K)  (selection uses bias)",
        "g_j = s_j / Σ_{k∈R} s_k     (weights ignore bias)",
        "b_j ← b_j − γ · sign(n_j − mean n)   (loss-free LB)",
        "m(u) = E^(s)(u) + λ Σ_{j∈R} g_j E_j^(r)(u)",
    ]
    for i, line in enumerate(formulas):
        parts.append(
            f'<text x="300" y="{370 + i * 22}" font-family="DejaVu Sans,Arial,sans-serif" '
            f'font-size="13" fill="#222">{line}</text>'
        )

    parts.append("</svg>")
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
