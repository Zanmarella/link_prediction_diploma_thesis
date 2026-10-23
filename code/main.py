"""Entry point for the synthetic baseline experiment."""

import argparse
import json
import subprocess
from pathlib import Path

from experiment import run


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/baseline.json")
    parser.add_argument("--output", type=Path, default=root)
    parser.add_argument("--build-pdfs", action="store_true", help="Compile article and one-slide PDF with latexmk")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    run(config, args.output)
    from reporting import export
    export(args.output)
    from plot_results import plot_all
    plot_all(args.output)
    if args.build_pdfs:
        for folder in ("paper", "slides"):
            subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                           cwd=args.output / folder, check=True)


if __name__ == "__main__":
    main()
