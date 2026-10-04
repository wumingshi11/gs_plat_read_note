#!/usr/bin/env python3
"""Shrink the heavy PDFs with Ghostscript.

Text stays vector (searchable, crisp at any zoom); only the embedded raster
images are downsampled/re-encoded, which is where all the weight lives.
The original is moved to papers/_originals/ (use --keep to leave it in place,
--restore to put the originals back).
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PAPERS = os.path.join(ROOT, "papers")
ORIG = os.path.join(PAPERS, "_originals")
DPI = 150
JPEG_Q = 88


def gs_compress(src, dst):
    cmd = [
        "gs", "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER",
        "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.7",
        # keep text/vector untouched, drop duplicate images
        "-dDetectDuplicateImages=true", "-dCompressFonts=true", "-dSubsetFonts=true",
        "-dAutoRotatePages=/None", "-dPreserveAnnots=true",
        # images only
        "-dDownsampleColorImages=true", "-dColorImageDownsampleType=/Bicubic",
        f"-dColorImageResolution={DPI}", "-dColorImageDownsampleThreshold=1.0",
        "-dDownsampleGrayImages=true", "-dGrayImageDownsampleType=/Bicubic",
        f"-dGrayImageResolution={DPI}", "-dGrayImageDownsampleThreshold=1.0",
        "-dDownsampleMonoImages=true", "-dMonoImageDownsampleType=/Subsample",
        "-dMonoImageResolution=600",
        "-dAutoFilterColorImages=false", "-dColorImageFilter=/DCTEncode",
        "-dAutoFilterGrayImages=false", "-dGrayImageFilter=/DCTEncode",
        f"-dJPEGQ={JPEG_Q}",
        f"-sOutputFile={dst}", src,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:300])


def npages(path):
    out = subprocess.run(["pdfinfo", path], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("Pages:"):
            return int(line.split()[1])
    return -1


def txtlen(path):
    """Characters recoverable from the text layer (0 => text got rasterized)."""
    r = subprocess.run(["pdftotext", path, "-"], capture_output=True, text=True)
    return len(r.stdout.strip())


def hits(min_mb):
    out = []
    for p in sorted(glob.glob(os.path.join(PAPERS, "*", "*.pdf"))):
        if os.path.basename(os.path.dirname(p)) == "_originals":
            continue
        sz = os.path.getsize(p) / 1048576
        if sz >= min_mb:
            out.append((p, sz))
    return out


def restore():
    n = 0
    for p in sorted(glob.glob(os.path.join(ORIG, "*.pdf"))):
        dst = os.path.join(PAPERS, p.split(os.sep)[-2] if False else "", os.path.basename(p))
        # recover the category from the stored mapping file
        tag = os.path.basename(p) + ".dir"
        catfile = os.path.join(ORIG, tag)
        cat = open(catfile).read().strip() if os.path.exists(catfile) else ""
        dst = os.path.join(PAPERS, cat, os.path.basename(p))
        shutil.move(p, dst)
        if os.path.exists(catfile):
            os.remove(catfile)
        n += 1
    print(f"restored {n} originals")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-mb", type=float, default=8.0)
    ap.add_argument("--archive", action="store_true",
                    help="move the original into papers/_originals/ instead of deleting it")
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.restore:
        return restore()

    os.makedirs(ORIG, exist_ok=True)
    todo = hits(a.min_mb)
    if not todo:
        print(f"nothing >= {a.min_mb} MB")
        return
    print(f"{len(todo)} PDFs >= {a.min_mb} MB\n")
    total_before = total_after = 0.0
    for path, mb in todo:
        cat = os.path.basename(os.path.dirname(path))
        tmp = path + ".small"
        if a.dry_run:
            print(f"  would compress {os.path.basename(path):<50} {mb:>6.1f} MB")
            continue
        try:
            gs_compress(path, tmp)
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] {os.path.basename(path)}: {e}")
            if os.path.exists(tmp):
                os.remove(tmp)
            continue
        new_mb = os.path.getsize(tmp) / 1048576
        # sanity: same page count, still has a text layer
        if npages(tmp) != npages(path):
            print(f"  [REJECT] {os.path.basename(path)} page count changed")
            os.remove(tmp)
            continue
        if not txtlen(tmp):
            print(f"  [REJECT] {os.path.basename(path)} lost its text layer")
            os.remove(tmp)
            continue
        if a.archive:
            shutil.move(path, os.path.join(ORIG, os.path.basename(path)))
            with open(os.path.join(ORIG, os.path.basename(path) + ".dir"), "w") as f:
                f.write(cat)
        os.replace(tmp, path)
        total_before += mb
        total_after += new_mb
        print(f"  {os.path.basename(path):<50} {mb:>6.1f} MB -> {new_mb:>5.2f} MB")
    if not a.dry_run:
        print(f"\ntotal {total_before:.1f} MB -> {total_after:.1f} MB")


if __name__ == "__main__":
    main()
