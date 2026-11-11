"""
upload_to_s3.py
---------------
Uploads the training images and the trained LoRA model to S3:

    data/dog/            ->  s3://<bucket>/datasets/dog/
    output/dog-lora/     ->  s3://<bucket>/models/dog-lora/
        (final LoRA, every checkpoint-N folder incl. optimizer state, and training logs)

Files already in S3 with the same size are skipped, so it is safe to run again.
The key layout matches app.py, which reads models/dog-lora/<checkpoint>/...

Usage (inside the venv, from the folder that contains data/ and output/):
    python upload_to_s3.py --bucket diffusion-sujal
    python upload_to_s3.py --bucket diffusion-sujal --dry-run    # only show what would upload
"""

import argparse
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parent
UPLOADS = [                                  # (local folder, S3 prefix)
    (ROOT / "data" / "dog", "datasets/dog"),
    (ROOT / "output" / "dog-lora", "models/dog-lora"),
]
CONTENT_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                 ".webp": "image/webp", ".json": "application/json", ".yml": "text/yaml"}


def remote_size(s3, bucket, key):
    try:
        return s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return None
        raise


def main():
    ap = argparse.ArgumentParser(description="Upload training images and LoRA model to S3")
    ap.add_argument("--bucket", default=os.environ.get("S3_BUCKET"), help="S3 bucket name")
    ap.add_argument("--dry-run", action="store_true", help="list files without uploading")
    args = ap.parse_args()
    if not args.bucket:
        sys.exit("Give a bucket: --bucket diffusion-sujal  (or set S3_BUCKET)")

    s3 = boto3.client("s3")
    s3.head_bucket(Bucket=args.bucket)       # fails fast on wrong name / credentials

    uploaded = skipped = total_bytes = 0
    for folder, prefix in UPLOADS:
        if not folder.exists():
            print(f"[skip] {folder} does not exist")
            continue
        for path in sorted(p for p in folder.rglob("*") if p.is_file()):
            key = f"{prefix}/{path.relative_to(folder).as_posix()}"
            size = path.stat().st_size
            if remote_size(s3, args.bucket, key) == size:
                skipped += 1
                continue
            print(f"[up] {size / 1024**2:7.1f} MB  s3://{args.bucket}/{key}")
            if not args.dry_run:
                extra = {"ContentType": CONTENT_TYPES[path.suffix.lower()]} \
                    if path.suffix.lower() in CONTENT_TYPES else None
                s3.upload_file(str(path), args.bucket, key, ExtraArgs=extra)
            uploaded += 1
            total_bytes += size

    verb = "would upload" if args.dry_run else "uploaded"
    print(f"\n{verb} {uploaded} files ({total_bytes / 1024**2:.1f} MB), "
          f"skipped {skipped} already in S3")


if __name__ == "__main__":
    main()
