"""
finetune_sd_lora.py
-------------------
One script that:
  1. downloads a ready-made dataset (5 photos of one dog, from Hugging Face: diffusers/dog-example)
  2. downloads the official diffusers DreamBooth-LoRA training script (matching your diffusers version)
  3. fine-tunes Stable Diffusion 1.5 with LoRA, sized for a 6 GB GPU (e.g. RTX 3050 Laptop)
  4. generates a comparison grid: rows = training steps, columns = prompts

Usage (inside the venv created by setup.bat):
    python finetune_sd_lora.py                 # everything
    python finetune_sd_lora.py --skip-train    # only regenerate the grid from saved checkpoints
    python finetune_sd_lora.py --steps 400     # shorter training
"""

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------- settings
BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DATASET_REPO = "diffusers/dog-example"
RARE_TOKEN = "sks"          # made-up word that becomes the dog's "name"
CLASS_NAME = "dog"
INSTANCE_PROMPT = f"a photo of {RARE_TOKEN} {CLASS_NAME}"

TEST_PROMPTS = [
    f"a photo of {RARE_TOKEN} dog",                 # can it reproduce the subject?
    f"a photo of {RARE_TOKEN} dog on a beach",      # new setting
    f"a pencil sketch of {RARE_TOKEN} dog",         # new style
    "a photo of a dog",                             # has it forgotten generic dogs? (overfitting)
    "a photo of a cat",                             # does everything turn into the dog? (overfitting)
]
SEED = 42

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "dog"
DOWNLOAD_DIR = ROOT / "data" / "_download"
OUT_DIR = ROOT / "output" / "dog-lora"
SCRIPTS_DIR = ROOT / "scripts"
RESULTS_DIR = ROOT / "results"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


# ---------------------------------------------------------------- 1. dataset
def download_dataset():
    if DATA_DIR.exists() and any(p.suffix.lower() in IMAGE_EXTS for p in DATA_DIR.iterdir()):
        print(f"[data] already present: {DATA_DIR}")
        return
    from huggingface_hub import snapshot_download

    print(f"[data] downloading {DATASET_REPO} ...")
    snapshot_download(repo_id=DATASET_REPO, repo_type="dataset", local_dir=DOWNLOAD_DIR)

    # The training script opens EVERY file in the folder as an image, so copy
    # only the images (no README, no .cache folder) into the training folder.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in DOWNLOAD_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and ".cache" not in p.parts:
            shutil.copy2(p, DATA_DIR / p.name)
            n += 1
    shutil.rmtree(DOWNLOAD_DIR, ignore_errors=True)
    print(f"[data] {n} training images in {DATA_DIR}")


# ---------------------------------------------------------------- 2. training script
def get_train_script() -> Path:
    import diffusers

    version = diffusers.__version__
    ref = "main" if "dev" in version else f"v{version}"
    target = SCRIPTS_DIR / f"train_dreambooth_lora_{ref}.py"
    if not target.exists():
        url = (f"https://raw.githubusercontent.com/huggingface/diffusers/{ref}"
               "/examples/dreambooth/train_dreambooth_lora.py")
        print(f"[script] downloading training script for diffusers {version}")
        SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, target)
    return target


# ---------------------------------------------------------------- 3. train
def train(steps: int, checkpoint_every: int, lr: float, rank: int, use_8bit: bool):
    script = get_train_script()
    cmd = [
        sys.executable, str(script),
        "--pretrained_model_name_or_path", BASE_MODEL,
        "--instance_data_dir", str(DATA_DIR),
        "--instance_prompt", INSTANCE_PROMPT,
        "--output_dir", str(OUT_DIR),
        "--resolution", "512",
        "--train_batch_size", "1",
        "--gradient_accumulation_steps", "1",
        "--gradient_checkpointing",            # saves VRAM
        "--mixed_precision", "fp16",           # saves VRAM
        "--learning_rate", str(lr),            # LoRA uses ~1e-4 (full fine-tune uses ~2e-6)
        "--lr_scheduler", "constant",
        "--lr_warmup_steps", "0",
        "--rank", str(rank),                   # LoRA size
        "--max_train_steps", str(steps),
        "--checkpointing_steps", str(checkpoint_every),
        "--seed", str(SEED),
        "--report_to", "tensorboard",
    ]
    if use_8bit:
        cmd.append("--use_8bit_adam")          # saves VRAM (needs bitsandbytes)

    print("[train] starting. Expect roughly 20-60 min on a laptop RTX 3050.")
    print("[train] command:\n  " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[train] done. LoRA saved to {OUT_DIR}")


# ---------------------------------------------------------------- 4. generate grid
def find_stages():
    """Return [(label, lora_file_or_None), ...] ordered by training step."""
    stages = [("base model\n(no LoRA)", None)]
    ckpts = []
    if OUT_DIR.exists():
        for d in OUT_DIR.glob("checkpoint-*"):
            f = d / "pytorch_lora_weights.safetensors"
            if f.exists():
                ckpts.append((int(d.name.split("-")[1]), f))
    for step, f in sorted(ckpts):
        stages.append((f"step {step}", f))
    final = OUT_DIR / "pytorch_lora_weights.safetensors"
    if not ckpts and final.exists():
        stages.append(("final", final))
    return stages


def generate():
    import torch
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    stages = find_stages()
    if len(stages) == 1:
        print("[gen] no trained LoRA found - run without --skip-train first.")
        return

    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, lora in stages:
        pipe.unload_lora_weights()
        if lora is not None:
            pipe.load_lora_weights(str(lora.parent), weight_name=lora.name)
        row = []
        for i, prompt in enumerate(TEST_PROMPTS):
            print(f"[gen] {label.replace(chr(10), ' ')} | {prompt}")
            g = torch.Generator("cuda").manual_seed(SEED)
            img = pipe(prompt, num_inference_steps=25, guidance_scale=7.5, generator=g).images[0]
            tag = label.split("\n")[0].replace(" ", "_")
            img.save(RESULTS_DIR / f"{tag}_p{i}.png")
            row.append(img)
        rows.append((label, row))

    out = RESULTS_DIR / "comparison_grid.png"
    make_grid(rows, TEST_PROMPTS).save(out)
    print(f"[gen] grid saved: {out}")


def make_grid(rows, prompts, cell=256, label_w=150, header_h=60):
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.load_default(size=16)
    except TypeError:                      # Pillow < 10.1
        font = ImageFont.load_default()

    W = label_w + cell * len(prompts)
    H = header_h + cell * len(rows)
    grid = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(grid)

    for c, p in enumerate(prompts):
        words, lines, line = p.split(), [], ""
        for w in words:
            if len(line + " " + w) > 28:
                lines.append(line)
                line = w
            else:
                line = (line + " " + w).strip()
        lines.append(line)
        draw.multiline_text((label_w + c * cell + 6, 6), "\n".join(lines[:3]), fill="black", font=font)

    for r, (label, imgs) in enumerate(rows):
        y = header_h + r * cell
        draw.multiline_text((8, y + cell // 2 - 16), label, fill="black", font=font)
        for c, img in enumerate(imgs):
            grid.paste(img.resize((cell, cell)), (label_w + c * cell, y))
    return grid


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Fine-tune SD 1.5 with DreamBooth-LoRA on a small GPU")
    ap.add_argument("--steps", type=int, default=800, help="total training steps")
    ap.add_argument("--checkpoint-every", type=int, default=200, help="save a LoRA every N steps")
    ap.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    ap.add_argument("--rank", type=int, default=4, help="LoRA rank (4-16 is typical)")
    ap.add_argument("--no-8bit", action="store_true", help="disable 8-bit Adam if bitsandbytes fails")
    ap.add_argument("--skip-train", action="store_true", help="only generate the comparison grid")
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        sys.exit("CUDA GPU not found. Install the CUDA build of PyTorch (see setup.bat).")
    print(f"[gpu] {torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    download_dataset()
    if not args.skip_train:
        train(args.steps, args.checkpoint_every, args.lr, args.rank, not args.no_8bit)
    generate()


if __name__ == "__main__":
    main()
