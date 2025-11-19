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
