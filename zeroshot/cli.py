"""Reproduce reviewed runs without the web interface; writes only on explicit invocation."""
import argparse
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from .demo import demo_bundle
from .engine import run
from .schema import RunConfig
from .web import export_bundle


def main():
    parser = argparse.ArgumentParser(description="ZeroShot reproducible local runner")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Write synthetic inputs for inspection; does not run predictions")
    demo.add_argument("--out", required=True)
    predict = commands.add_parser("run", help="Run reviewed evidence against CSV context/test rows")
    predict.add_argument("--context", required=True)
    predict.add_argument("--test")
    predict.add_argument("--evidence", required=True)
    predict.add_argument("--config", required=True)
    predict.add_argument("--out", required=True)
    args = parser.parse_args()
    destination = Path(args.out)
    if destination.exists():
        parser.error("Output path already exists. Choose a new path to preserve previous work.")
    if args.command == "demo":
        destination.mkdir(parents=True)
        frame, bundle = demo_bundle()
        frame.to_csv(destination / "context.csv", index=False)
        (destination / "evidence.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
        config = RunConfig(method="direct", allow_synthetic=True).model_dump()
        (destination / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        print("Synthetic inputs written. Review model.reviewed and config.assumptions_approved before running.")
        return
    context = pd.read_csv(args.context)
    test = pd.read_csv(args.test) if args.test else None
    bundle = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result = run(context, test, bundle, config, lambda text, fraction: print(text, flush=True))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("predictions.csv", result["predictions"].to_csv(index=False))
        archive.writestr("audit.json", json.dumps(result["audit"], indent=2, allow_nan=False))
        archive.writestr("evidence.json", json.dumps(export_bundle(bundle), indent=2, allow_nan=False))
        samples = io.BytesIO(); np.savez_compressed(samples, **result["arrays"])
        archive.writestr("predictive_distributions.npz", samples.getvalue())
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()

