import os, json, time, torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from peft import LoraConfig, get_peft_model

print("=" * 54)
print("   LoRA LITE TRAINER — RAM SAFE VERSION")
print("=" * 54)

BRAND       = "luxury"
DATA_DIR    = f"week4_lora/datasets/{BRAND}"
IMG_DIR     = os.path.join(DATA_DIR, "images")
ADAPTER_DIR = f"week4_lora/adapters/{BRAND}"
os.makedirs(ADAPTER_DIR, exist_ok=True)

TRAIN_STEPS = 50
LR          = 1e-4
IMG_SIZE    = 256
RANK        = 2
MODEL_ID    = "runwayml/stable-diffusion-v1-5"

print(f"\n  Brand   : {BRAND}")
print(f"  Steps   : {TRAIN_STEPS}")
print(f"  ImgSize : {IMG_SIZE}x{IMG_SIZE} (RAM-safe)")
print(f"  Rank    : {RANK} (minimal footprint)\n")

# ── Minimal dataset ───────────────────────────────────────────
class BrandDataset(Dataset):
    def __init__(self, img_dir, cap_file, size=256):
        self.img_dir = img_dir
        self.size    = size
        with open(cap_file) as f:
            caps = json.load(f)
        self.items = [
            (k, v) for k, v in caps.items()
            if os.path.exists(os.path.join(img_dir, k))
        ][:10]

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        fname, cap = self.items[idx]
        img = Image.open(
            os.path.join(self.img_dir, fname)
        ).convert("RGB").resize((self.size, self.size))
        arr = np.array(img).astype(np.float32)/127.5 - 1.0
        return {
            "pixel_values": torch.from_numpy(arr).permute(2,0,1),
            "caption": cap,
        }

print("Loading dataset (10 samples)...")
ds     = BrandDataset(IMG_DIR, f"{DATA_DIR}/captions.json")
loader = DataLoader(ds, batch_size=1, shuffle=True)
print(f"  {len(ds)} samples ready\n")

# ── Load ONLY the UNet (not full pipeline) ────────────────────
print("Loading UNet only (saves ~3GB RAM)...")
unet = UNet2DConditionModel.from_pretrained(
    MODEL_ID, subfolder="unet",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)
tokenizer = CLIPTokenizer.from_pretrained(
    MODEL_ID, subfolder="tokenizer"
)
text_enc = CLIPTextModel.from_pretrained(
    MODEL_ID, subfolder="text_encoder",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)
vae = AutoencoderKL.from_pretrained(
    MODEL_ID, subfolder="vae",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)
scheduler = DDPMScheduler.from_pretrained(
    MODEL_ID, subfolder="scheduler"
)

vae.eval();      text_enc.eval()
for p in vae.parameters():      p.requires_grad_(False)
for p in text_enc.parameters(): p.requires_grad_(False)
print("  Components loaded!\n")

# ── Apply LoRA ────────────────────────────────────────────────
print("Applying LoRA (rank=2)...")
lora_cfg = LoraConfig(
    r=RANK, lora_alpha=RANK*2,
    target_modules=["to_q","to_v"],
    lora_dropout=0.0, bias="none",
)
unet = get_peft_model(unet, lora_cfg)
trainable = sum(
    p.numel() for p in unet.parameters() if p.requires_grad
)
print(f"  Trainable params: {trainable:,}\n")

optimizer = torch.optim.AdamW(
    [p for p in unet.parameters() if p.requires_grad], lr=LR
)

# ── Training loop ─────────────────────────────────────────────
print("Training...\n")
unet.train()
losses    = []
start     = time.time()
data_iter = iter(loader)

for step in range(1, TRAIN_STEPS + 1):
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(loader)
        batch     = next(data_iter)

    pixels   = batch["pixel_values"]
    captions = batch["caption"]

    with torch.no_grad():
        latents = vae.encode(pixels).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        tok = tokenizer(
            captions, padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        )
        enc = text_enc(tok.input_ids)[0]

    noise = torch.randn_like(latents)
    t     = torch.randint(
        0, scheduler.config.num_train_timesteps,
        (1,)
    ).long()
    noisy = scheduler.add_noise(latents, noise, t)

    pred  = unet(noisy, t, enc).sample
    loss  = torch.nn.functional.mse_loss(pred, noise)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        unet.parameters(), 1.0
    )
    optimizer.step()

    losses.append(loss.item())

    if step % 10 == 0:
        avg = sum(losses[-10:]) / 10
        pct = step / TRAIN_STEPS
        bar = "█" * int(pct*20) + "░" * (20-int(pct*20))
        print(
            f"  Step {step:>3}/{TRAIN_STEPS} "
            f"[{bar}] loss={avg:.4f} "
            f"({time.time()-start:.0f}s)"
        )

# ── Save adapter ──────────────────────────────────────────────
print(f"\nSaving LoRA adapter...")
unet.save_pretrained(ADAPTER_DIR)

meta = {
    "brand":      BRAND,
    "token":      f"<{BRAND}-style>",
    "steps":      TRAIN_STEPS,
    "rank":       RANK,
    "final_loss": round(losses[-1], 4),
    "avg_loss":   round(sum(losses)/len(losses), 4),
}
with open(f"{ADAPTER_DIR}/adapter_meta.json","w") as f:
    json.dump(meta, f, indent=2)

print(f"  Saved: {ADAPTER_DIR}")
print(f"\n{'='*54}")
print(f"   LORA TRAINING COMPLETE!")
print(f"{'='*54}")
print(f"\n  Final loss : {losses[-1]:.4f}")
print(f"  Avg loss   : {sum(losses)/len(losses):.4f}")
print(f"  Time       : {time.time()-start:.0f}s")
print(f"\nTask 2 complete!")
