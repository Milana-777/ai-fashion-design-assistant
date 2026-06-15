import cv2
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
import os, time, json

print("=" * 54)
print("   POSE CONTROL — CONTROLNET PIPELINE")
print("=" * 54)

print("\nLoading OpenPose ControlNet model...")
print("(Downloads ~1.4GB on first run...)\n")

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose",
    torch_dtype=torch.float32,
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float32,
    safety_checker=None,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config
)
pipe.enable_attention_slicing()
print("Pipeline ready!\n")

# ── Pose skeleton builder ─────────────────────────────────────
def draw_pose_skeleton(pose_name="standing", save_path=None):
    canvas = np.zeros((512, 512, 3), dtype=np.uint8)

    POSES = {
        "standing": {
            "joints": {
                "nose":       (255, 50),
                "neck":       (255, 100),
                "r_shoulder": (200, 120),
                "l_shoulder": (310, 120),
                "r_elbow":    (170, 220),
                "l_elbow":    (340, 220),
                "r_wrist":    (155, 310),
                "l_wrist":    (355, 310),
                "r_hip":      (215, 290),
                "l_hip":      (295, 290),
                "r_knee":     (210, 390),
                "l_knee":     (300, 390),
                "r_ankle":    (205, 480),
                "l_ankle":    (305, 480),
            },
            "connections": [
                ("nose","neck"),
                ("neck","r_shoulder"),("neck","l_shoulder"),
                ("r_shoulder","r_elbow"),("r_elbow","r_wrist"),
                ("l_shoulder","l_elbow"),("l_elbow","l_wrist"),
                ("neck","r_hip"),("neck","l_hip"),
                ("r_hip","l_hip"),
                ("r_hip","r_knee"),("r_knee","r_ankle"),
                ("l_hip","l_knee"),("l_knee","l_ankle"),
            ]
        },
        "walking": {
            "joints": {
                "nose":       (245, 45),
                "neck":       (248, 95),
                "r_shoulder": (190, 115),
                "l_shoulder": (308, 115),
                "r_elbow":    (150, 210),
                "l_elbow":    (350, 195),
                "r_wrist":    (175, 305),
                "l_wrist":    (320, 290),
                "r_hip":      (210, 285),
                "l_hip":      (290, 285),
                "r_knee":     (185, 375),
                "l_knee":     (315, 370),
                "r_ankle":    (160, 460),
                "l_ankle":    (330, 455),
            },
            "connections": [
                ("nose","neck"),
                ("neck","r_shoulder"),("neck","l_shoulder"),
                ("r_shoulder","r_elbow"),("r_elbow","r_wrist"),
                ("l_shoulder","l_elbow"),("l_elbow","l_wrist"),
                ("neck","r_hip"),("neck","l_hip"),
                ("r_hip","l_hip"),
                ("r_hip","r_knee"),("r_knee","r_ankle"),
                ("l_hip","l_knee"),("l_knee","l_ankle"),
            ]
        },
        "hands_on_hips": {
            "joints": {
                "nose":       (255, 48),
                "neck":       (255, 98),
                "r_shoulder": (195, 118),
                "l_shoulder": (315, 118),
                "r_elbow":    (165, 195),
                "l_elbow":    (345, 195),
                "r_wrist":    (210, 268),
                "l_wrist":    (300, 268),
                "r_hip":      (215, 285),
                "l_hip":      (295, 285),
                "r_knee":     (210, 385),
                "l_knee":     (300, 385),
                "r_ankle":    (208, 475),
                "l_ankle":    (302, 475),
            },
            "connections": [
                ("nose","neck"),
                ("neck","r_shoulder"),("neck","l_shoulder"),
                ("r_shoulder","r_elbow"),("r_elbow","r_wrist"),
                ("l_shoulder","l_elbow"),("l_elbow","l_wrist"),
                ("neck","r_hip"),("neck","l_hip"),
                ("r_hip","l_hip"),
                ("r_hip","r_knee"),("r_knee","r_ankle"),
                ("l_hip","l_knee"),("l_knee","l_ankle"),
            ]
        },
    }

    COLORS = {
        "head":   (255, 100, 100),
        "torso":  (100, 255, 100),
        "r_arm":  (255, 165,   0),
        "l_arm":  (  0, 165, 255),
        "r_leg":  (200, 100, 255),
        "l_leg":  (255, 100, 200),
    }

    def get_color(j1, j2):
        pair = (j1, j2)
        if "nose" in pair or "neck" in pair and "shoulder" not in j2:
            return COLORS["head"]
        if "r_shoulder" in pair or "r_elbow" in pair or "r_wrist" in pair:
            return COLORS["r_arm"]
        if "l_shoulder" in pair or "l_elbow" in pair or "l_wrist" in pair:
            return COLORS["l_arm"]
        if "r_hip" in pair or "r_knee" in pair or "r_ankle" in pair:
            return COLORS["r_leg"]
        if "l_hip" in pair or "l_knee" in pair or "l_ankle" in pair:
            return COLORS["l_leg"]
        return COLORS["torso"]

    pose = POSES[pose_name]
    joints = pose["joints"]

    for j1, j2 in pose["connections"]:
        pt1 = joints[j1]
        pt2 = joints[j2]
        col = get_color(j1, j2)
        cv2.line(canvas, pt1, pt2, col, 3)

    for name, pt in joints.items():
        cv2.circle(canvas, pt, 6, (255, 255, 255), -1)
        cv2.circle(canvas, pt, 4, (200, 200, 200),  1)

    img = Image.fromarray(canvas)
    if save_path:
        img.save(save_path)
    return img

# ── Pose + fashion combinations ───────────────────────────────
POSE_DESIGNS = [
    {
        "name":    "standing_evening",
        "pose":    "standing",
        "prompt":  (
            "full body fashion model wearing elegant "
            "black evening gown, silk fabric, "
            "high fashion photography, studio lighting, "
            "white background, vogue editorial, 4k"
        ),
        "negative": "blurry, ugly, deformed, extra limbs, bad anatomy",
        "cfg":   8.0,
        "cnet":  1.0,
        "steps": 20,
    },
    {
        "name":    "walking_streetwear",
        "pose":    "walking",
        "prompt":  (
            "full body fashion model walking, "
            "oversized streetwear outfit, white hoodie, "
            "black joggers, urban style, "
            "editorial fashion photography, clean background"
        ),
        "negative": "blurry, ugly, deformed, extra limbs, static",
        "cfg":   8.0,
        "cnet":  1.0,
        "steps": 20,
    },
    {
        "name":    "hips_powersuit",
        "pose":    "hands_on_hips",
        "prompt":  (
            "full body fashion model hands on hips, "
            "tailored navy power suit, "
            "structured blazer and trousers, "
            "business fashion photography, white background, "
            "sharp focus, editorial quality"
        ),
        "negative": "blurry, ugly, deformed, casual, extra limbs",
        "cfg":   8.0,
        "cnet":  1.0,
        "steps": 20,
    },
]

results = []

for design in POSE_DESIGNS:
    print(f"Processing: {design['name']}...")

    pose_path = (
        f"week3_controlnet/pose_outputs/"
        f"{design['name']}_pose.png"
    )
    pose_img = draw_pose_skeleton(design["pose"], pose_path)
    print(f"  Pose saved   : {pose_path}")

    print(f"  Generating...")
    start = time.time()
    image = pipe(
        prompt                        = design["prompt"],
        negative_prompt               = design["negative"],
        image                         = pose_img,
        num_inference_steps           = design["steps"],
        guidance_scale                = design["cfg"],
        controlnet_conditioning_scale = design["cnet"],
        height=512, width=512,
    ).images[0]

    out_path = (
        f"week3_controlnet/pose_outputs/"
        f"{design['name']}_output.png"
    )
    image.save(out_path)
    elapsed = time.time() - start

    results.append({
        "name":     design["name"],
        "pose":     design["pose"],
        "pose_img": pose_path,
        "output":   out_path,
        "time_sec": round(elapsed),
    })
    print(f"  Output saved : {out_path}  ({elapsed:.0f}s)\n")

log_path = "week3_controlnet/pose_control_log.json"
with open(log_path, "w") as f:
    json.dump(results, f, indent=2)

print("=" * 54)
print("   POSE CONTROL COMPLETE!")
print("=" * 54)
print(f"\nGenerated : {len(results)} pose-controlled designs")
print(f"Log saved : {log_path}")
print(f"Outputs   : week3_controlnet/pose_outputs/")
print("\nTask 3 complete!")
