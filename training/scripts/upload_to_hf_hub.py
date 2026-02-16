"""
Upload hierarchical ONNX models + semantic gate to HuggingFace Hub.

Builds a staging directory with the correct structure, then uploads everything.

Usage:
    python -m training.scripts.upload_to_hf_hub
    python -m training.scripts.upload_to_hf_hub --dry-run

Requirements:
    pip install huggingface_hub
    huggingface-cli login
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def build_hierarchy_metadata(onnx_dir: Path) -> dict:
    """Build hierarchy_metadata.json from the exported ONNX directory structure."""
    contexts = ["professional", "learning", "social", "psychological", "personal"]

    metadata = {
        "version": "2.0",
        "description": "Hierarchical 4-level intent classification (ONNX INT8 quantized)",
        "levels": {
            "routing": {"path": "routing/", "type": "single_label"},
            "contexts": {"path": "contexts/", "type": "single_label"},
            "entities": {"type": "single_label", "models": {}},
            "sub_entities": {"type": "multi_label", "models": {}},
        },
    }

    for ctx in contexts:
        entity_dir = onnx_dir / ctx / "entities"
        if entity_dir.exists() and (entity_dir / "config.json").exists():
            with open(entity_dir / "config.json") as f:
                config = json.load(f)
            metadata["levels"]["entities"]["models"][ctx] = {
                "path": f"{ctx}/entities/",
                "labels": list(config.get("id2label", {}).values()),
            }

        ctx_dir = onnx_dir / ctx
        if not ctx_dir.exists():
            continue
        for sub_dir in sorted(ctx_dir.iterdir()):
            if sub_dir.name == "entities" or not sub_dir.is_dir():
                continue
            if not (sub_dir / "model_quantized.onnx").exists():
                continue
            labels, threshold = [], 0.5
            if (sub_dir / "label_mappings.json").exists():
                with open(sub_dir / "label_mappings.json") as f:
                    lc = json.load(f)
                lm = lc.get("label_mappings", lc.get("id2label", lc))
                labels = list(lm.values()) if isinstance(lm, dict) else lm
                threshold = lc.get("threshold", 0.5)
            if ctx not in metadata["levels"]["sub_entities"]["models"]:
                metadata["levels"]["sub_entities"]["models"][ctx] = {}
            metadata["levels"]["sub_entities"]["models"][ctx][sub_dir.name] = {
                "path": f"{ctx}/{sub_dir.name}/",
                "labels": labels,
                "threshold": threshold,
            }

    return metadata


def build_staging(staging_dir: Path, onnx_dir: Path, semantic_gate_dir: Path, tuning_json: Path):
    """Build staging directory with correct HF Hub structure."""
    if staging_dir.exists():
        shutil.rmtree(str(staging_dir))
    staging_dir.mkdir(parents=True)

    copied = 0
    skipped_fp32 = 0

    # 1. Copy hierarchical models (skip FP32)
    if onnx_dir.exists():
        dest_hier = staging_dir / "hierarchical"
        for src_file in onnx_dir.rglob("*"):
            if src_file.is_dir():
                continue
            if src_file.name == "model.onnx":
                skipped_fp32 += 1
                continue
            rel = src_file.relative_to(onnx_dir)
            dst = dest_hier / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src_file), str(dst))
            copied += 1

        # Generate hierarchy_metadata.json
        metadata = build_hierarchy_metadata(onnx_dir)
        with open(dest_hier / "hierarchy_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        copied += 1

    # 2. Copy semantic gate files
    if semantic_gate_dir.exists():
        dest_sg = staging_dir / "semantic_gate"
        for src_file in semantic_gate_dir.rglob("*"):
            if src_file.is_dir():
                continue
            if src_file.name == "model.onnx":
                skipped_fp32 += 1
                continue
            rel = src_file.relative_to(semantic_gate_dir)
            dst = dest_sg / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src_file), str(dst))
            copied += 1

    # 3. Copy tuning JSON to semantic_gate/
    if tuning_json.exists():
        dest_sg = staging_dir / "semantic_gate"
        dest_sg.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(tuning_json), str(dest_sg / "semantic_gate_hierarchical_tuning.json"))
        copied += 1

    return copied, skipped_fp32


def main():
    parser = argparse.ArgumentParser(description="Upload ONNX models to HuggingFace Hub")
    parser.add_argument("--repo-id", type=str, default="cfa0819/pop-skills-onnx")
    parser.add_argument("--onnx-dir", type=str, default="training/models/full_onnx")
    parser.add_argument("--semantic-gate-dir", type=str, default="training/models/onnx/semantic_gate")
    parser.add_argument("--tuning-json", type=str, default="training/results/semantic_gate_hierarchical_tuning.json")
    parser.add_argument("--staging-dir", type=str, default="training/models/_hf_staging")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    onnx_dir = Path(args.onnx_dir)
    semantic_gate_dir = Path(args.semantic_gate_dir)
    tuning_json = Path(args.tuning_json)
    staging_dir = Path(args.staging_dir)

    print("=" * 60)
    print("UPLOAD TO HUGGINGFACE HUB")
    print("=" * 60)
    print(f"  Repo:          {args.repo_id}")
    print(f"  Hierarchical:  {onnx_dir}")
    print(f"  Semantic gate: {semantic_gate_dir}")
    print(f"  Tuning JSON:   {tuning_json}")

    if not onnx_dir.exists():
        print(f"\n[ERROR] ONNX directory not found: {onnx_dir}")
        sys.exit(1)

    # Build staging directory
    print(f"\n[1/3] Building staging directory...")
    copied, skipped = build_staging(staging_dir, onnx_dir, semantic_gate_dir, tuning_json)

    # Calculate size
    total_size = sum(f.stat().st_size for f in staging_dir.rglob("*") if f.is_file())
    total_files = sum(1 for f in staging_dir.rglob("*") if f.is_file())
    print(f"  Files staged: {total_files} ({total_size / 1024 / 1024:.0f} MB)")
    print(f"  FP32 skipped: {skipped}")

    # Show structure
    print(f"\n  Structure:")
    for d in sorted(set(f.parent.relative_to(staging_dir) for f in staging_dir.rglob("*") if f.is_file())):
        n_files = sum(1 for f in (staging_dir / d).iterdir() if f.is_file())
        print(f"    {d}/ ({n_files} files)")

    if args.dry_run:
        print(f"\n[DRY RUN] Would upload {total_files} files ({total_size / 1024 / 1024:.0f} MB)")
        print(f"  Cleaning up staging dir...")
        shutil.rmtree(str(staging_dir), ignore_errors=True)
        return

    # Create repo
    print(f"\n[2/3] Creating repo (if needed)...")
    create_repo(args.repo_id, repo_type="model", exist_ok=True)

    # Upload
    print(f"\n[3/3] Uploading {total_files} files ({total_size / 1024 / 1024:.0f} MB)...")
    print(f"  Using upload_large_folder for reliable upload...")
    api = HfApi()
    api.upload_large_folder(
        folder_path=str(staging_dir),
        repo_id=args.repo_id,
        repo_type="model",
    )

    # Cleanup staging
    print(f"\n  Cleaning up staging dir...")
    shutil.rmtree(str(staging_dir), ignore_errors=True)

    print(f"\n{'='*60}")
    print(f"DONE! https://huggingface.co/{args.repo_id}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
