import argparse
import pathlib
import statistics
from typing import List


def parse_returns_file(path: pathlib.Path) -> List[float]:
    """Parse a returns_XXX file into a list of floats."""
    vals: List[float] = []
    if not path.exists():
        return vals

    with path.open("r") as f:
        for line in f:
            clean = (
                line.replace(",", " ")
                    .replace("[", " ")
                    .replace("]", " ")
                    .strip()
            )
            if not clean:
                continue
            for tok in clean.split():
                try:
                    vals.append(float(tok))
                except ValueError:
                    continue
    return vals


def summarize_run(run_dir: pathlib.Path) -> None:
    """Look under runs/<run_name>/eval_*/returns_* and print basic stats."""
    if not run_dir.exists():
        print(f"[WARN] Run dir not found: {run_dir}")
        return

    eval_dirs = sorted(d for d in run_dir.glob("eval_*") if d.is_dir())
    if not eval_dirs:
        print(f"[WARN] No eval_* directories under {run_dir}")
        return

    all_returns: List[float] = []
    for ed in eval_dirs:
        for ret_file in ed.glob("returns_*"):
            vals = parse_returns_file(ret_file)
            all_returns.extend(vals)

    if not all_returns:
        print(f"[WARN] No returns found under {run_dir}")
        return

    mean_ret = statistics.mean(all_returns)
    median_ret = statistics.median(all_returns)
    min_ret = min(all_returns)
    max_ret = max(all_returns)

    print(f"=== {run_dir.name} ===")
    print(f"  Num episodes: {len(all_returns)}")
    print(f"  Mean return : {mean_ret:.3f}")
    print(f"  Median      : {median_ret:.3f}")
    print(f"  Min / Max   : {min_ret:.3f} / {max_ret:.3f}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Summarize evaluation returns from RecSim runs."
    )
    parser.add_argument(
        "run_names",
        nargs="+",
        help="Names under ./runs/ to analyze (e.g. smoke_test_lambda0).",
    )
    args = parser.parse_args()

    root = pathlib.Path(__file__).parent.resolve()
    runs_root = root / "runs"

    for name in args.run_names:
        summarize_run(runs_root / name)


if __name__ == "__main__":
    main()
