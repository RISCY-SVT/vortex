#!/usr/bin/env python3
import argparse
import csv
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


def now_ts():
    return time.strftime("%Y%m%d_%H%M%S")


def run_cmd(cmd, log_path):
    start = time.monotonic()
    proc = subprocess.run(["bash", "-lc", cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    elapsed = time.monotonic() - start
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout)
    return proc.returncode, elapsed


def build_blackbox_cmd(build_dir, driver, app, args):
    cmd = f"cd {shlex.quote(build_dir)} && source ./ci/toolchain_env.sh && ./ci/blackbox.sh --driver={shlex.quote(driver)} --app={shlex.quote(app)}"
    if args:
        cmd += f" --args={shlex.quote(args)}"
    return cmd


def find_artifacts(outdir, prefix):
    if not prefix:
        return []
    p = Path(outdir)
    return sorted(str(x) for x in p.glob(f"{prefix}_*.ppm"))


def write_tables(outdir, rows):
    outdir = Path(outdir)
    md_path = outdir / "results.md"
    csv_path = outdir / "results.csv"

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["app", "variant", "driver", "args", "exit_code", "elapsed_s", "log", "artifacts"])
        for r in rows:
            writer.writerow([r["app"], r["variant"], r["driver"], r["args"], r["exit_code"], f"{r['elapsed']:.2f}", r["log"], "|".join(r["artifacts"])])

    with md_path.open("w") as f:
        f.write("# Video Suite Results\n\n")
        f.write("| app | variant | driver | args | exit | elapsed(s) | log | artifacts |\n")
        f.write("|---|---|---|---|---:|---:|---|---|\n")
        for r in rows:
            f.write(
                f"| {r['app']} | {r['variant']} | {r['driver']} | {r['args']} | {r['exit_code']} | {r['elapsed']:.2f} | {r['log']} | {';'.join(Path(a).name for a in r['artifacts'])} |\n"
            )


def generate_collage(ppm_files, out_path, tile=256, cols=4):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as e:
        print(f"Pillow not available: {e}")
        return False

    if not ppm_files:
        print("No PPM files found for collage")
        return False

    images = []
    labels = []
    for p in ppm_files:
        img = Image.open(p).convert("RGB")
        scale = min(tile / img.width, tile / img.height)
        new_w = max(1, int(img.width * scale))
        new_h = max(1, int(img.height * scale))
        resized = img.resize((new_w, new_h), Image.NEAREST)
        tile_img = Image.new("RGB", (tile, tile), (0, 0, 0))
        tile_img.paste(resized, ((tile - new_w) // 2, (tile - new_h) // 2))
        images.append(tile_img)
        labels.append(Path(p).stem)

    cols = max(1, min(cols, len(images)))
    rows = math.ceil(len(images) / cols)
    pad = 4
    out_w = cols * tile + (cols - 1) * pad
    out_h = rows * tile + (rows - 1) * pad
    canvas = Image.new("RGB", (out_w, out_h), (16, 16, 16))

    font = ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)

    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        x = c * (tile + pad)
        y = r * (tile + pad)
        canvas.paste(img, (x, y))
        label = labels[idx]
        # simple shadow for readability
        draw.text((x + 4, y + 4), label, font=font, fill=(0, 0, 0))
        draw.text((x + 3, y + 3), label, font=font, fill=(255, 255, 255))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=90)
    return True


def files_identical(path_a, path_b):
    try:
        with open(path_a, "rb") as fa, open(path_b, "rb") as fb:
            if os.fstat(fa.fileno()).st_size != os.fstat(fb.fileno()).st_size:
                return False
            while True:
                ba = fa.read(1 << 20)
                bb = fb.read(1 << 20)
                if not ba and not bb:
                    return True
                if ba != bb:
                    return False
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Run Vortex video compute suite (simx) and generate reports.")
    parser.add_argument("--build-dir", default="/data/vortex/build", help="Build directory (default: /data/vortex/build)")
    parser.add_argument("--driver", default="simx", help="Driver (default: simx)")
    parser.add_argument("--out-dir", default=None, help="Output directory for logs/artifacts")
    parser.add_argument("--mode", choices=["quick", "full"], default="full", help="quick=defaults, full=pretty+tail+padding")
    args = parser.parse_args()

    build_dir = Path(args.build_dir).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else (build_dir / "artifacts" / "video_suite" / now_ts())
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    apps = ["test_pattern", "yuv2rgb", "scaler", "convolution", "alpha_blend", "scanout_sim"]

    def add_run(app, variant, cmd_args, prefix=None):
        runs.append({
            "app": app,
            "variant": variant,
            "driver": args.driver,
            "args": cmd_args,
            "prefix": prefix,
        })

    if args.mode == "quick":
        defaults = {
            "test_pattern": "-w 63 -h 65 -mode 0",
            "yuv2rgb": "-w 62 -h 66",
            "scaler": "-w 63 -h 65 -ow 96 -oh 96",
            "convolution": "-w 63 -h 65",
            "alpha_blend": "-w 63 -h 65",
            "scanout_sim": "-w 63 -h 65",
        }
        for app in apps:
            base = defaults[app]
            prefix = f"quick_{app}"
            add_run(app, "quick", f"{base} --dump --outdir {out_dir} --prefix {prefix}", prefix)
    else:
        for app in apps:
            if app == "scaler":
                add_run(app, "pretty", f"-w 191 -h 193 -ow 320 -oh 320 --dump --outdir {out_dir} --prefix pretty_{app}", f"pretty_{app}")
            elif app == "scanout_sim":
                add_run(app, "pretty", f"-w 320 -h 320 -stride 1344 --dump --outdir {out_dir} --prefix pretty_{app}", f"pretty_{app}")
            else:
                add_run(app, "pretty", f"-w 320 -h 320 --dump --outdir {out_dir} --prefix pretty_{app}", f"pretty_{app}")

            if app == "yuv2rgb":
                prefix = f"tail_{app}"
                add_run(app, "tail", f"-w 318 -h 322 --outdir {out_dir} --prefix {prefix}", prefix)
            elif app == "scaler":
                prefix = f"tail_{app}"
                add_run(app, "tail", f"-w 197 -h 199 -ow 319 -oh 321 --outdir {out_dir} --prefix {prefix}", prefix)
            else:
                prefix = f"tail_{app}"
                add_run(app, "tail", f"-w 319 -h 321 --outdir {out_dir} --prefix {prefix}", prefix)

            if app == "scanout_sim":
                add_run(app, "padding", f"-w 63 -h 65 --dump --outdir {out_dir} --prefix padding_{app}", f"padding_{app}")

    rows = []
    any_fail = False
    log_dir = out_dir / "logs"

    for r in runs:
        log_path = log_dir / f"{r['app']}_{r['variant']}.log"
        cmd = build_blackbox_cmd(str(build_dir), args.driver, r["app"], r["args"])
        rc, elapsed = run_cmd(cmd, log_path)
        if r["app"] == "scaler" and r["variant"] == "pretty":
            near = out_dir / f"{r['prefix']}_output_nearest.ppm"
            bil = out_dir / f"{r['prefix']}_output_bilinear.ppm"
            if not near.exists() or not bil.exists():
                msg = "ERROR: scaler pretty outputs missing; cannot verify nearest vs bilinear difference"
                with log_path.open("a") as f:
                    f.write(f"\n{msg}\n")
                print(msg)
                rc = 1
            elif files_identical(near, bil):
                msg = "ERROR: scaler outputs identical; scaling coverage insufficient"
                with log_path.open("a") as f:
                    f.write(f"\n{msg}\n")
                print(msg)
                rc = 1
        if rc != 0:
            any_fail = True
        artifacts = find_artifacts(out_dir, r.get("prefix")) if r.get("prefix") else []
        rows.append({
            "app": r["app"],
            "variant": r["variant"],
            "driver": args.driver,
            "args": r["args"],
            "exit_code": rc,
            "elapsed": elapsed,
            "log": str(log_path),
            "artifacts": artifacts,
        })
        print(f"{r['app']}:{r['variant']} exit={rc} elapsed={elapsed:.2f}s")

    write_tables(out_dir, rows)

    ppm_files = sorted(str(p) for p in Path(out_dir).glob("*.ppm"))
    collage_path = out_dir / "collage.jpg"
    if not generate_collage(ppm_files, collage_path):
        print("Failed to generate collage.jpg (see above)")
        any_fail = True
    else:
        print(f"Collage written: {collage_path}")

    if any_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
