import os
import subprocess
import shlex
import pathlib
import sys
import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-name",
        type=str,
        default="ie_penalized",
        help="Subdirectory under ./runs/ where outputs will be saved.",
    )
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=100,
        help="Max steps per episode for the RecSim runner.",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=5,
        help="Number of TrainRunner iterations.",
    )
    parser.add_argument(
        "--max-training-steps",
        type=int,
        default=200,
        help="Max training steps per TrainRunner iteration.",
    )
    parser.add_argument(
        "--max-eval-episodes",
        type=int,
        default=5,
        help="Number of eval episodes.",
    )
    parser.add_argument(
        "--rep-penalty",
        type=float,
        default=0.0,
        help="Repetition penalty lambda for the environment.",
    )

    args = parser.parse_args()

    if not args.run_name:
        args.run_name = "ie_penalized"

    return args


def main(args):
    root = pathlib.Path(__file__).parent.resolve()
    os.environ["PYTHONPATH"] = f"{root}/overrides:" + os.environ.get("PYTHONPATH", "")

    os.environ["IE_REP_PENALTY"] = str(args.rep_penalty)

    outdir = root / "runs" / args.run_name
    outdir.mkdir(parents=True, exist_ok=True)

    gin_bindings = [
        f"simulator.runner_lib.Runner.max_steps_per_episode={args.max_steps_per_episode}",
        f"simulator.runner_lib.TrainRunner.num_iterations={args.num_iterations}",
        f"simulator.runner_lib.TrainRunner.max_training_steps={args.max_training_steps}",
        f"simulator.runner_lib.EvalRunner.max_eval_episodes={args.max_eval_episodes}",
    ]

    cmd = [
        "python",
        "-m",
        "recsim.main",
        "--logtostderr",
        f'--base_dir={outdir}',
        "--agent_name=full_slate_q",
        "--environment_name=interest_exploration",
        "--episode_log_file=episode_logs.tfrecord",
    ]

    for b in gin_bindings:
        cmd.append(f"--gin_bindings={b}")

    print("Running:", " ".join(cmd))
    sys.exit(subprocess.call(cmd, env=os.environ.copy()))


if __name__ == "__main__":
    args = get_args()
    main(args)
