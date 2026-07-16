#!/usr/bin/env python3
"""Draw the unified 55-dim action/state slot layout as SVG (stdlib only)."""

from pathlib import Path

OUT = Path(__file__).with_name("action_space_55dim.svg")

# (label, dim, color)
SLOTS = [
    ("arm.position", 14, "#4C78A8"),
    ("end.position", 14, "#F58518"),
    ("effector.position", 2, "#E45756"),
    ("hand.position", 12, "#72B7B2"),
    ("waist.position", 4, "#54A24B"),
    ("head.position", 2, "#EECA3B"),
    ("base.position", 3, "#B279A2"),
    ("reserved", 4, "#9D755D"),
]


def main() -> None:
    w, h = 900, 280
    bar_y, bar_h = 90, 56
    margin = 40
    usable = w - 2 * margin
    unit = usable / 55

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#FAFAFA"/>',
        '<text x="450" y="36" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" '
        'font-size="18" font-weight="700" fill="#222">Unified 55-dim Action / State Space</text>',
        '<text x="450" y="58" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" '
        'font-size="12" fill="#666">Missing body parts are zero-padded; joint_mask marks valid dims</text>',
    ]

    x = margin
    for label, dim, color in SLOTS:
        bw = dim * unit
        parts.append(
            f'<rect x="{x:.1f}" y="{bar_y}" width="{bw:.1f}" height="{bar_h}" '
            f'fill="{color}" stroke="#fff" stroke-width="1.5" rx="3"/>'
        )
        cx = x + bw / 2
        short = label.split(".")[0] if "." in label else label
        parts.append(
            f'<text x="{cx:.1f}" y="{bar_y + bar_h / 2 + 5:.1f}" text-anchor="middle" '
            f'font-family="DejaVu Sans,Arial,sans-serif" font-size="11" font-weight="600" fill="#fff">'
            f"{short} ({dim})</text>"
        )
        x += bw

    # axis ticks
    parts.append(
        f'<line x1="{margin}" y1="{bar_y + bar_h + 12}" x2="{margin + usable}" '
        f'y2="{bar_y + bar_h + 12}" stroke="#888" stroke-width="1"/>'
    )
    for i in (0, 14, 28, 30, 42, 46, 48, 51, 55):
        tx = margin + i * unit
        parts.append(
            f'<line x1="{tx:.1f}" y1="{bar_y + bar_h + 8}" x2="{tx:.1f}" '
            f'y2="{bar_y + bar_h + 16}" stroke="#888"/>'
        )
        parts.append(
            f'<text x="{tx:.1f}" y="{bar_y + bar_h + 32}" text-anchor="middle" '
            f'font-family="DejaVu Sans,Arial,sans-serif" font-size="11" fill="#444">{i}</text>'
        )

    # legend
    lx, ly = margin, 220
    for i, (label, dim, color) in enumerate(SLOTS):
        col = i % 4
        row = i // 4
        xx = lx + col * 210
        yy = ly + row * 22
        parts.append(f'<rect x="{xx}" y="{yy}" width="12" height="12" fill="{color}" rx="2"/>')
        parts.append(
            f'<text x="{xx + 18}" y="{yy + 11}" font-family="DejaVu Sans,Arial,sans-serif" '
            f'font-size="11" fill="#333">{label} = {dim}</text>'
        )

    parts.append("</svg>")
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
