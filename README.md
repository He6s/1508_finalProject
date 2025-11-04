# RecSim Minimal Project
Uses RecSim's prebuilt **interest_exploration** environment. Our novelty is a
small repetition penalty in the choice model (see `overrides/.../choice_model.py`).

## Quickstart (Python 3.8 via conda recommended)
```bash
conda create -n recsim38 python=3.8 -y
conda activate recsim38
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python code.py    # or: ./scripts/run_ie.sh

