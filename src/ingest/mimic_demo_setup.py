"""
Download MIMIC-IV Demo dataset (public, no credentials required) as a
fallback when MIMIC-III PhysioNet access is pending.

MIMIC-IV Demo contains ~100 patients with full structure matching MIMIC-III tables.
Tables are renamed to match the MIMIC-III naming convention used in batch_loader.py.

Usage:
  python src/ingest/mimic_demo_setup.py [--out-dir /data/mimic-iii]
"""

import os
import argparse
import zipfile
import shutil
import urllib.request
from pathlib import Path

DEMO_URL = "https://physionet.org/static/published-projects/mimic-iv-demo/mimic-iv-clinical-database-demo-2.2.zip"

# Map MIMIC-IV demo table → MIMIC-III filename expected by batch_loader.py
TABLE_MAP = {
    "hosp/patients.csv":           "PATIENTS.csv",
    "hosp/admissions.csv":         "ADMISSIONS.csv",
    "hosp/diagnoses_icd.csv":      "DIAGNOSES_ICD.csv",
    "hosp/procedures_icd.csv":     "PROCEDURES_ICD.csv",
    "hosp/prescriptions.csv":      "PRESCRIPTIONS.csv",
    "hosp/labevents.csv":          "LABEVENTS.csv",
    "hosp/d_icd_diagnoses.csv":    "D_ICD_DIAGNOSES.csv",
    "hosp/d_icd_procedures.csv":   "D_ICD_PROCEDURES.csv",
    "hosp/d_labitems.csv":         "D_LABITEMS.csv",
    "note/discharge.csv":          "NOTEEVENTS.csv",   # closest equivalent
}


def download_with_progress(url: str, dest: Path):
    print(f"Downloading MIMIC-IV demo (~55 MB) ...")
    def _hook(count, block_size, total):
        pct = min(count * block_size / total * 100, 100)
        print(f"\r  {pct:.1f}%", end="", flush=True)
    urllib.request.urlretrieve(url, dest, reporthook=_hook)
    print()


def extract_and_rename(zip_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "_tmp_extract"
    tmp.mkdir(exist_ok=True)

    print("Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp)

    # Find root inside zip (varies by version)
    roots = [p for p in tmp.iterdir() if p.is_dir()]
    root = roots[0] if roots else tmp

    copied = 0
    for src_rel, dest_name in TABLE_MAP.items():
        src = root / src_rel
        if not src.exists():
            # Try alternate paths
            candidates = list(root.rglob(src_rel.split("/")[-1]))
            if candidates:
                src = candidates[0]
            else:
                print(f"  WARN: {src_rel} not found, skipping")
                continue
        dest = out_dir / dest_name
        shutil.copy2(src, dest)
        print(f"  {src_rel} → {dest_name}")
        copied += 1

    shutil.rmtree(tmp)
    print(f"\nDone. {copied}/{len(TABLE_MAP)} tables copied to {out_dir}")
    print("NOTE: NOTEEVENTS.csv is sourced from discharge notes (MIMIC-IV format)")
    print("      Column names may differ slightly — batch_loader.py will handle missing cols.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=os.getenv("MIMIC_DATA_DIR", "/data/mimic-iii"))
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download if zip already exists")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    zip_path = out_dir.parent / "mimic-iv-demo.zip"

    if not args.skip_download or not zip_path.exists():
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        download_with_progress(DEMO_URL, zip_path)
    else:
        print(f"Using existing {zip_path}")

    extract_and_rename(zip_path, out_dir)


if __name__ == "__main__":
    main()
