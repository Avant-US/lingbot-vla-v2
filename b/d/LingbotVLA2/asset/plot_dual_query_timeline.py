#!/usr/bin/env python3
"""Dual-query distillation timeline: Q_t and Q_{t+T} aligned with action chunk."""

from pathlib import Path

OUT = Path(__file__).with_name("dual_query_timeline.svg")


def main() -> None:
    w, h = 900, 420
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#FAFAFA"/>',
        '<text x="450" y="30" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" '
        'font-size="17" font-weight="700" fill="#222">Dual-Query Distillation Timeline</text>',
        '<text x="450" y="52" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" '
        'font-size="12" fill="#666">Horizon T = action chunk size; teachers frozen; queries learnable</text>',
    ]

    # time axis
    y_axis = 200
    parts.append(f'<line x1="80" y1="{y_axis}" x2="820" y2="{y_axis}" stroke="#333" stroke-width="2"/>')
    for x, lab in ((120, "t (now)"), (520, "t+T (future)"), (780, "time →")):
        parts.append(f'<circle cx="{x}" cy="{y_axis}" r="5" fill="#333"/>')
        parts.append(
            f'<text x="{x}" y="{y_axis + 28}" text-anchor="middle" '
            f'font-family="DejaVu Sans,Arial,sans-serif" font-size="13" fill="#333">{lab}</text>'
        )

    # action chunk bar
    parts.append(
        f'<rect x="120" y="{y_axis - 18}" width="400" height="16" fill="#AED6F1" stroke="#2980B9" rx="4"/>'
    )
    parts.append(
        f'<text x="320" y="{y_axis - 28}" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" '
        'font-size="12" fill="#1A5276">action chunk (flow-matching targets)</text>'
    )

    # Q_t box
    parts.append(
        '<rect x="60" y="70" width="160" height="90" fill="#D5F5E3" stroke="#1E8449" stroke-width="2" rx="8"/>'
    )
    parts.append(
        '<text x="140" y="100" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" '
        'font-size="14" font-weight="700" fill="#1E8449">Q_t</text>'
    )
    parts.append(
        '<text x="140" y="122" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" '
        'font-size="11" fill="#333">current depth</text>'
    )
    parts.append(
        '<text x="140" y="140" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" '
        'font-size="11" fill="#333">(+ current DINO)</text>'
    )
    parts.append(f'<line x1="140" y1="160" x2="120" y2="{y_axis - 5}" stroke="#1E8449" stroke-width="2"/>')

    # Q_{t+T} box
    parts.append(
        '<rect x="440" y="70" width="180" height="90" fill="#FCF3CF" stroke="#B7950B" stroke-width="2" rx="8"/>'
    )
    parts.append(
        '<text x="530" y="100" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" '
        'font-size="14" font-weight="700" fill="#B7950B">Q_{t+T}</text>'
    )
    parts.append(
        '<text x="530" y="122" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" '
        'font-size="11" fill="#333">future depth + video</text>'
    )
    parts.append(
        '<text x="530" y="140" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" '
        'font-size="11" fill="#333">(shared query mode)</text>'
    )
    parts.append(f'<line x1="530" y1="160" x2="520" y2="{y_axis - 5}" stroke="#B7950B" stroke-width="2"/>')

    # teachers
    parts.append(
        '<rect x="680" y="80" width="180" height="50" fill="#FADBD8" stroke="#922B21" rx="6"/>'
    )
    parts.append(
        '<text x="770" y="110" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" '
        'font-size="12" fill="#922B21">LingBot-Depth (frozen)</text>'
    )
    parts.append(
        '<rect x="680" y="145" width="180" height="50" fill="#D2B4DE" stroke="#6C3483" rx="6"/>'
    )
    parts.append(
        '<text x="770" y="175" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" '
        'font-size="12" fill="#6C3483">DINO-Video (frozen)</text>'
    )

    # attention isolation note
    parts.append(
        '<rect x="80" y="280" width="740" height="110" fill="#fff" stroke="#ddd" rx="8"/>'
    )
    notes = [
        "Prefix order (utils.prefix_query_segments): language → current_depth → [future_video] → future_depth",
        "block_future_depth_to_action / block_suffix_to_future_video: action suffix cannot attend future queries",
        "→ Future info trains perception queries as auxiliary task, without leaking into action generation",
        "Loss: L_depth (L1/smooth-L1) + L_video (MSE) on Proj(Q) vs teacher features; weights ≈ 0.004",
    ]
    for i, line in enumerate(notes):
        parts.append(
            f'<text x="100" y="{310 + i * 22}" font-family="DejaVu Sans,Arial,sans-serif" '
            f'font-size="12" fill="#222">{line}</text>'
        )

    parts.append("</svg>")
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
