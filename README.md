# RecSim Project – Interest Exploration with Repetition Penalty

This project uses Google **RecSim**’s `interest_exploration` environment with a
custom **repetition penalty** in the user choice model. We treat this as a
sequential recommendation problem and compare four policies:

- **Random baseline**
- **Bandit baseline** (epsilon-greedy over slates)
- **RL agent, λ = 0.0** (no repetition penalty)
- **RL agent, λ = 0.2** (with repetition penalty)

The environment is based on **synthetic user profiles** (RecSim’s built-in
`interest_exploration` model). We modify the user choice model to penalize
repeated recommendations and study the impact on long-term reward and slate
diversity.

Final runs and plots are already in the repo; the instructions below are only
for reproducing them if needed.

---

## 1. Repository structure (where things are)

Only folders/scripts that matter for the report are listed here.

- **Environment modification**
  - `overrides/recsim/environments/interest_exploration/choice_model.py`  
    → custom choice model with **repetition penalty** parameter `rep_penalty` (λ).

- **RL training wrapper**
  - `code.py`  
    → calls `recsim.main` with flags such as `--run-name`, `--rep-penalty`,
      `--max-steps-per-episode`, etc.

- **Baselines (demo scripts)**
  - `run_random_baseline.py`  
    → random recommender; logs returns and a simple slate-diversity metric.
  - `run_bandit_baseline.py`  
    → epsilon-greedy bandit over slates; logs returns and diversity.

- **Evaluation / plotting**
  - `analyze_returns.py`  
    → reads `runs/*/eval_*/returns_*`, prints mean/median/min/max per run.
  - `plot_results.py`  
    → generates the bar plots used in the report.

- **Final run folders used in the report**
  - `runs/random_baseline_long`
  - `runs/bandit_baseline_long`
  - `runs/lambda_0_long`      (RL, λ = 0.0)
  - `runs/lambda_0_2_long`    (RL, λ = 0.2)

  Each contains:
  - `eval_* / returns_*` – per-episode returns  
  - (Baselines only) `eval_random/diversity_random`,
    `eval_bandit/diversity_bandit` – per-episode diversity values

- **Results / figures**
  - `results/final_analyze_returns.txt`  
    → summary table (num episodes, mean, median, min, max) for the four final runs.
  - `results/final_diversity.txt`  
    → mean diversity for `random_baseline_long` and `bandit_baseline_long`.
  - `results/mean_return_bar.png`  
    → bar chart of mean return (Random, Bandit, RL λ=0, RL λ=0.2).
  - `results/diversity_bar.png`  
    → bar chart of mean diversity (Random vs Bandit).

These files are enough to write the report and slides without re-running
anything.

---

## 2. Setup

### 2.1 Clone the repository

```bash
git clone https://github.com/He6s/1508_finalProject.git
cd 1508_finalProject
