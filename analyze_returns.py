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
    """Look under runs/<run_name>/eval_*/returns_* and diversity_* and print basic stats."""
    if not run_dir.exists():
        print(f"[WARN] Run dir not found: {run_dir}")
        return

    eval_dirs = sorted(d for d in run_dir.glob("eval_*") if d.is_dir())
    if not eval_dirs:
        print(f"[WARN] No eval_* directories under {run_dir}")
        return

    all_returns: List[float] = []
    all_diversity: List[float] = []

    for ed in eval_dirs:
        for ret_file in ed.glob("returns_*"):
            vals = parse_returns_file(ret_file)
            all_returns.extend(vals)
        for div_file in ed.glob("diversity_*"):
            vals = parse_returns_file(div_file)
            all_diversity.extend(vals)

    print(f"=== {run_dir.name} ===")

    if all_returns:
        mean_ret = statistics.mean(all_returns)
        median_ret = statistics.median(all_returns)
        min_ret = min(all_returns)
        max_ret = max(all_returns)

        print(f"  [Returns]")
        print(f"    Num episodes: {len(all_returns)}")
        print(f"    Mean return : {mean_ret:.3f}")
        print(f"    Median      : {median_ret:.3f}")
        print(f"    Min / Max   : {min_ret:.3f} / {max_ret:.3f}")
    else:
        print(f"  [Returns] No returns found.")

    if all_diversity:
        mean_div = statistics.mean(all_diversity)
        median_div = statistics.median(all_diversity)
        min_div = min(all_diversity)
        max_div = max(all_diversity)

        print(f"  [Diversity]")
        print(f"    Num episodes: {len(all_diversity)}")
        print(f"    Mean diversity: {mean_div:.3f}")
        print(f"    Median        : {median_div:.3f}")
        print(f"    Min / Max     : {min_div:.3f} / {max_div:.3f}")
    
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
