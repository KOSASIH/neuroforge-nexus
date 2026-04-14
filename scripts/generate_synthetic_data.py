#!/usr/bin/env python3
"""
NeuroForge Nexus — Synthetic Data Generator
============================================
Generate labeled EEG datasets for BCI development and model training.

Usage:
    python scripts/generate_synthetic_data.py
    python scripts/generate_synthetic_data.py --subjects 20 --duration 120 --channels 64
    python scripts/generate_synthetic_data.py --paradigm p300 --subjects 5
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from neuroforge.simulation.brain_simulator import BrainSimulator
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="NeuroForge Synthetic Data Generator")
    parser.add_argument("--subjects", type=int, default=5, help="Number of subjects")
    parser.add_argument("--duration", type=float, default=60.0, help="Session duration (s)")
    parser.add_argument("--channels", type=int, default=64, help="EEG channels")
    parser.add_argument("--sample-rate", type=int, default=256, help="Sample rate (Hz)")
    parser.add_argument(
        "--paradigm",
        choices=["motor_imagery", "p300", "ssvep", "resting", "all"],
        default="all",
    )
    parser.add_argument("--output-dir", default="data/synthetic", help="Output directory")
    parser.add_argument("--snr", type=float, default=15.0, help="Signal-to-noise ratio (dB)")
    parser.add_argument("--format", choices=["npz", "npy"], default="npz")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    paradigms = (
        ["motor_imagery", "p300", "ssvep", "resting"]
        if args.paradigm == "all"
        else [args.paradigm]
    )

    start = time.time()
    manifest = {"sessions": []}

    for sub_idx in range(args.subjects):
        subject_id = f"subject_{sub_idx:03d}"
        logger.info(f"Generating {subject_id} | {len(paradigms)} paradigms")

        sim = BrainSimulator(
            n_channels=args.channels,
            sample_rate=args.sample_rate,
            subject_id=subject_id,
        )

        for paradigm in paradigms:
            logger.info(f"  → {paradigm}")

            if paradigm == "motor_imagery":
                session = sim.simulate_motor_imagery(
                    n_trials=max(10, int(args.duration / 6)),
                    snr_db=args.snr,
                )
            elif paradigm == "p300":
                session = sim.simulate_p300(
                    n_trials=max(50, int(args.duration / 0.8)),
                    snr_db=args.snr,
                )
            elif paradigm == "ssvep":
                session = sim.simulate_ssvep(
                    n_trials=max(20, int(args.duration / 3)),
                    snr_db=args.snr,
                )
            else:  # resting
                session = sim.simulate_resting_state(
                    duration_s=args.duration,
                    snr_db=args.snr,
                )

            # Save
            fname = f"{subject_id}_{paradigm}"
            if args.format == "npz":
                out_path = os.path.join(args.output_dir, fname + ".npz")
                np.savez_compressed(
                    out_path,
                    data=session.data.astype(np.float32),
                    labels=session.labels,
                    sample_rate=session.sample_rate,
                )
            else:
                out_path = os.path.join(args.output_dir, fname + ".npy")
                np.save(out_path, session.data.astype(np.float32))

            # Metadata
            meta = {
                "session_id": session.session_id,
                "subject_id": session.subject_id,
                "paradigm": session.paradigm,
                "n_channels": session.data.shape[0],
                "n_samples": session.data.shape[1],
                "duration_s": session.duration_s,
                "sample_rate": session.sample_rate,
                "n_events": len(session.events),
                "snr_db": session.snr_db,
                "file": out_path,
            }
            manifest["sessions"].append(meta)

    # Save manifest
    manifest["total_sessions"] = len(manifest["sessions"])
    manifest["generated_in_s"] = round(time.time() - start, 2)
    manifest["subjects"] = args.subjects
    manifest["paradigms"] = paradigms

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.success(
        f"\n✅ Generated {manifest['total_sessions']} sessions in {manifest['generated_in_s']}s"
        f"\n   Output: {args.output_dir}"
        f"\n   Manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()
