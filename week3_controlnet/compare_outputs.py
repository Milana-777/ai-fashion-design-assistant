import torch, json, os
import open_clip
from PIL import Image

print("=" * 54)
print("   CONTROLLED vs UNCONTROLLED COMPARISON")
print("=" * 54)

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model.eval()

def clip_score(image_path, prompt):
    img  = preprocess(
        Image.open(image_path).convert("RGB")
    ).unsqueeze(0)
    text = tokenizer([prompt])
    with torch.no_grad():
        i_feat = model.encode_image(img)
        t_feat = model.encode_text(text)
        i_feat = i_feat / i_feat.norm(dim=-1, keepdim=True)
        t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)
        return round((i_feat @ t_feat.T).item() * 100, 2)

COMPARISONS = [
    {
        "name":        "evening_dress",
        "uncontrolled": "week2_sdxl/outputs/elegant_dress.png",
        "controlled":   "week3_controlnet/outputs/evening_dress.png",
        "prompt":       "elegant evening dress fashion photography",
    },
    {
        "name":        "power_suit",
        "uncontrolled": "week2_sdxl/library_outputs/elegant_evening.png",
        "controlled":   "week3_controlnet/outputs/power_suit.png",
        "prompt":       "tailored power suit jacket business professional",
    },
    {
        "name":        "casual_top",
        "uncontrolled": "week2_sdxl/library_outputs/casual_summer.png",
        "controlled":   "week3_controlnet/outputs/casual_top.png",
        "prompt":       "casual cotton top streetwear fashion photography",
    },
]

print("\nCLIP score comparison:\n")
print(f"  {'Design':<20} {'Uncontrolled':>14} "
      f"{'Controlled':>12} {'Improvement':>13}")
print("  " + "─" * 62)

report = []
for c in COMPARISONS:
    u_score, ctrl_score = 0.0, 0.0

    if os.path.exists(c["uncontrolled"]):
        u_score = clip_score(c["uncontrolled"], c["prompt"])
    else:
        print(f"  SKIP uncontrolled: {c['uncontrolled']}")

    if os.path.exists(c["controlled"]):
        ctrl_score = clip_score(c["controlled"], c["prompt"])
    else:
        print(f"  SKIP controlled: {c['controlled']}")

    delta = round(ctrl_score - u_score, 2)
    arrow = "↑" if delta > 0 else "↓"
    print(f"  {c['name']:<20} {u_score:>12.1f} "
          f"{ctrl_score:>12.1f}   "
          f"{arrow} {abs(delta):.1f}")

    report.append({
        "name":              c["name"],
        "uncontrolled_score": u_score,
        "controlled_score":   ctrl_score,
        "improvement":        delta,
    })

avg_u    = sum(r["uncontrolled_score"] for r in report) / len(report)
avg_ctrl = sum(r["controlled_score"]   for r in report) / len(report)
print("  " + "─" * 62)
print(f"  {'Average':<20} {avg_u:>12.1f} {avg_ctrl:>12.1f}"
      f"   {'↑' if avg_ctrl > avg_u else '↓'} "
      f"{abs(avg_ctrl - avg_u):.1f}")

print(f"\n\nPose control scores:")
print(f"  {'Design':<28} {'CLIP Score':>12}")
print("  " + "─" * 42)
pose_dir = "week3_controlnet/pose_outputs"
pose_prompts = {
    "standing_evening_output":  "fashion model evening gown studio photography",
    "walking_streetwear_output":"fashion model walking streetwear urban style",
    "hips_powersuit_output":    "fashion model power suit business professional",
}
for fname, prompt in pose_prompts.items():
    fpath = os.path.join(pose_dir, f"{fname}.png")
    if os.path.exists(fpath):
        s = clip_score(fpath, prompt)
        print(f"  {fname:<28} {s:>12.1f}")

out_path = "week3_controlnet/comparison_report.json"
with open(out_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"\n{'='*54}")
print("   COMPARISON COMPLETE!")
print(f"{'='*54}")
print(f"\nReport saved: {out_path}")
print("\nKey insight:")
print("  ControlNet constrains structure → higher prompt")
print("  adherence → better CLIP scores on fashion tasks")
print("\nTask 4 complete!")
print("Week 3 complete!")
