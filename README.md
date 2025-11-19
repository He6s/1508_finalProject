# RecSim Project – Interest Exploration + Repetition Penalty

This repo runs Google **RecSim**’s `interest_exploration` environment and adds a small **environment modification**:  
a **repetition penalty** in the user choice model.

- Custom choice model: `overrides/recsim/environments/interest_exploration/choice_model.py`
- Run outputs: `./runs/<run-name>/...`

---

## 1. Environment setup (do this once)

We only support this stack (because RecSim + Dopamine + TF1 are cursed):

- **Python**: 3.7
- **Conda env**: `recsim37`
- **OS**: Windows 10/11 (CPU only is fine)

In **Anaconda Prompt**:

```bash
# Clone the repo
git clone https://github.com/He6s/1508_finalProject.git
cd 1508_finalProject

# Create + activate Python 3.7 env
conda create -n recsim37 python=3.7 -y
conda activate recsim37
python -V   # should show 3.7.x

# Install the exact packages we use
pip install --upgrade pip

pip install \
  "tensorflow==1.15.0" \
  "dopamine-rl==2.0.5" \
  "recsim==0.2.4" \
  "numpy==1.18.5" \
  "protobuf==3.20.3" \
  "gin-config==0.1.1" \
  "atari-py==0.2.6" \
  "absl-py"


## Project status

### What’s done

- **Environment & tooling**
  - Working Conda env `recsim37` on Windows with TensorFlow 1.15, Dopamine, and RecSim.
  - Patched `gin.tf` / TensorFlow summary issues so `python -m recsim.main` runs without crashing.

- **Core experiments**
  - Novelty implemented: **repetition penalty** in  
    `overrides/recsim/environments/interest_exploration/choice_model.py`.
  - RL training wrapper: `code.py` calls `recsim.main` with:
    - `--run-name`, `--rep-penalty`, `--max-steps-per-episode`,  
      `--num-iterations`, `--max-training-steps`, `--max-eval-episodes`.
  - Completed runs (results under `runs/<run-name>/...`):
    - `random_baseline` – random policy.
    - `smoke_test_lambda0` – short sanity check, λ = 0.0.
    - `lambda_0`, `lambda_0_long` – RL, λ = 0.0.
    - `lambda_0_2`, `lambda_0_2_long` – RL, λ = 0.2.

- **Evaluation tools**
  - `run_random_baseline.py` – runs the random policy and writes  
    `runs/random_baseline/eval_random/returns_random`.
  - `analyze_returns.py` – reads `runs/*/eval_*/returns_*` and prints
    num episodes, mean, median, min, max for each run.

- **Repo hygiene**
  - `.gitignore` ignores heavy training logs: `runs/*/train/` and `runs/*/eval_*/`.
  - Only small summary files (`runs/.../returns_*`) are tracked in git.

### What’s left to do

- **Baselines**
  - Finish and run `run_bandit_baseline.py` (simple heuristic / bandit baseline).
  - Compare: random vs. bandit vs. RL (λ = 0.0 vs. λ > 0.0) using `analyze_returns.py`.

- **Analysis & plots**
  - Add a small plotting script (e.g. `plot_returns.py`) to visualize
    mean returns with variability (error bars or per-episode curves).
  - Write a short interpretation of results:
    - RL vs. random.
    - Effect of the repetition penalty (λ = 0 vs. λ = 0.2).

- **Report / documentation**
  - Integrate final setup, baselines, and results into the course report.
  - Make sure README has a “How to reproduce” section:
    - How to run RL experiments via `code.py ...`.
    - How to run baselines (`run_random_baseline.py`, `run_bandit_baseline.py`).
    - How to summarize results with `analyze_returns.py`.

