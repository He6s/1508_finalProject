# RecSim Minimal Project

This repo runs Google **RecSim**’s pre-implemented `interest_exploration` environment and adds a tiny **environment modification** (our required novelty): a **repetition penalty** in the user choice model (see `overrides/recsim/environments/interest_exploration/choice_model.py`).  
Outputs are saved under `./runs/ie_penalized/`.

---

## 1) Prerequisites
- **Python 3.8** (use Conda so everyone has the same version)
- macOS / Linux / Windows (Anaconda Prompt)

---

## 2) Quickstart (copy these commands)

```bash
# Clone
git clone https://github.com/He6s/1508_finalProject.git recsim-project
cd recsim-project

# Create & activate Python 3.8 environment (Conda)
conda create -n recsim38 python=3.8 -y
conda activate recsim38
python -V     # should print 3.8.x

# Install dependencies using the env’s pip
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Run
python code.py
# (or) ./scripts/run_ie.sh
