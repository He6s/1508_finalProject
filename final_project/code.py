import os, subprocess, shlex, pathlib, sys

def main():
    root = pathlib.Path(__file__).parent.resolve()
    # Make our override take precedence over installed recsim
    os.environ["PYTHONPATH"] = f"{root}/overrides:" + os.environ.get("PYTHONPATH", "")
    outdir = root / "runs" / "ie_penalized"
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = f"""
    python -m recsim.main --logtostderr
      --base_dir="{outdir}"
      --agent_name=full_slate_q
      --environment_name=interest_exploration
      --episode_log_file="episode_logs.tfrecord"
      --gin_bindings=simulator.runner_lib.Runner.max_steps_per_episode=100
      --gin_bindings=simulator.runner_lib.TrainRunner.num_iterations=5
      --gin_bindings=simulator.runner_lib.TrainRunner.max_training_steps=200
      --gin_bindings=simulator.runner_lib.EvalRunner.max_eval_episodes=5
    """
    print("Running:", cmd)
    sys.exit(subprocess.call(shlex.split(cmd), env=os.environ.copy()))

if __name__ == "__main__":
    main()

