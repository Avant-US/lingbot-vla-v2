#!/usr/bin/env python3
"""Draw the 55-dim unified space with RoboTwin hanging_mug valid dims highlighted."""

from pathlib import Path

OUT = Path(__file__).with_name("hanging_mug_55dim.svg")

# Canonical 55-dim layout from the paper / robotwin training joints.
# RoboTwin maps 12 arm joints + 2 grippers; remaining slots are zero-padded.
SLOTS = [
    ("arm.position", 14, "#4C78A8", 12),  # 6+6 joints, pad 2
    ("end.position", 14, "#F58518", 0),
    ("effector.position", 2, "#E45756", 2),
    ("hand.position", 12, "#B0B0B0", 0),
    ("waist.position", 4, "#B0B0B0", 0),
    ("head.position", 2, "#B0B0B0", 0),
    ("base.position", 3, "#B0B0B0", 0),
    ("reserved", 4, "#D0D0D0", 0),
]


def main() -> None:
    w, h = 980, 320
    bar_y, bar_h = 92, 58
    margin = 40
    usable = w - 2 * margin
    unit = usable / 55
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#FAFAFA"/>',
        '<text x="490" y="34" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" '
        'font-size="18" font-weight="700" fill="#222">Unified 55-dim space for hanging_mug (RoboTwin Aloha-AgileX)</text>',
        '<text x="490" y="56" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" '
        'font-size="12" fill="#666">Colored sub-bars = joint_mask=1 (14 physical DoF). Gray = padded zeros, excluded from L1 loss.</text>',
    ]
    x = margin
    for label, dim, color, used in SLOTS:
        bw = dim * unit
        parts.append(
            f'<rect x="{x:.1f}" y="{bar_y}" width="{bw:.1f}" height="{bar_h}" '
            f'fill="#E8E8E8" stroke="#fff" stroke-width="1.5" rx="3"/>'
        )
        if used > 0:
            uw = used * unit
            parts.append(
                f'<rect x="{x:.1f}" y="{bar_y}" width="{uw:.1f}" height="{bar_h}" '
                f'fill="{color}" stroke="#fff" stroke-width="1.5" rx="3"/>'
            )
        cx = x + bw / 2
        short = label.split(".")[0]
        fill = "#222" if used == 0 else "#fff"
        parts.append(
            f'<text x="{cx:.1f}" y="{bar_y + bar_h / 2 + 5:.1f}" text-anchor="middle" '
            f'font-family="DejaVu Sans,Arial,sans-serif" font-size="11" font-weight="600" fill="{fill}">'
            f"{short} ({used}/{dim})</text>"
        )
        x += bw

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

    legend = [
        (40, 250, "#4C78A8", "arm joints used: 12 = L[0:6)+R[7:13)"),
        (40, 272, "#E45756", "grippers used: 2 = L[6:7)+R[13:14)"),
        (520, 250, "#E8E8E8", "padded slots (joint_mask=0, no loss)"),
        (520, 272, "#F58518", "end.position declared but unused on RoboTwin"),
    ]
    for xx, yy, color, text in legend:
        parts.append(f'<rect x="{xx}" y="{yy}" width="12" height="12" fill="{color}" rx="2"/>')
        parts.append(
            f'<text x="{xx + 18}" y="{yy + 11}" font-family="DejaVu Sans,Arial,sans-serif" '
            f'font-size="12" fill="#333">{text}</text>'
        )
    parts.append("</svg>")
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
