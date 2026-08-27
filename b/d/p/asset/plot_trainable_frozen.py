#!/usr/bin/env python3
"""Draw trainable vs frozen modules during hanging_mug post-training."""

from pathlib import Path

OUT = Path(__file__).with_name("hanging_mug_trainable_frozen.svg")


def box(x, y, w, h, fill, text, sub="", stroke="#333"):
    lines = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>',
        f'<text x="{x + w / 2:.1f}" y="{y + (22 if sub else h / 2 + 5):.1f}" text-anchor="middle" '
        f'font-family="DejaVu Sans,Arial,sans-serif" font-size="13" font-weight="700" fill="#222">{text}</text>',
    ]
    if sub:
        lines.append(
            f'<text x="{x + w / 2:.1f}" y="{y + 42:.1f}" text-anchor="middle" '
            f'font-family="DejaVu Sans,Arial,sans-serif" font-size="11" fill="#444">{sub}</text>'
        )
    return "\n".join(lines)


def main() -> None:
    w, h = 980, 430
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#FAFAFA"/>',
        '<text x="490" y="32" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" '
        'font-size="18" font-weight="700" fill="#222">hanging_mug fine-tune: what is updated vs frozen</text>',
        '<text x="490" y="54" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" '
        'font-size="12" fill="#666">post_training=true, freeze_vision_encoder=false, train_expert_only=false</text>',
        box(40, 80, 280, 70, "#C8E6C9", "Qwen3 ViT", "trainable (lr = 1e-4)"),
        box(350, 80, 280, 70, "#C8E6C9", "Qwen3-VL LM", "trainable, FSDP2 sharded"),
        box(660, 80, 280, 70, "#C8E6C9", "Query tokens + align heads", "depth/video distillation"),
        box(40, 180, 430, 80, "#A5D6A7", "Action Expert 36-layer + Sparse MoE", "trainable; routed-expert LR = 2.83e-4"),
        box(500, 180, 210, 80, "#C8E6C9", "state/action proj", "AdaRMSNorm time cond."),
        box(740, 180, 200, 80, "#C8E6C9", "Muon + AdamW", "2D/3D vs 1D params"),
        box(40, 290, 280, 80, "#FFCDD2", "MoGe (frozen)", "depth teacher, no_grad"),
        box(350, 290, 280, 80, "#FFCDD2", "LingBot-Depth / MoRGBD", "frozen teacher"),
        box(660, 290, 280, 80, "#FFCDD2", "DINO-Video", "frozen teacher"),
        '<text x="40" y="410" font-family="DejaVu Sans,Arial,sans-serif" font-size="12" fill="#333">'
        "Green = student weights loaded from robbyant/lingbot-vla-v2-6b and updated. Red = teachers, forward-only.</text>",
        "</svg>",
    ]
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
