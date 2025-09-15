# Switchable & Dynamic Quantization for GPT-2 (SQuAD)

This project implements **switchable and dynamic quantization** on GPT-2 with LoRA adapters, including:
- Per-layer configurable bit-widths (weights & activations)
- Runtime profile switching (`uniform8`, `mixed8648`)
- Joint (switchable) training across profiles
- Optional cyclic precision training
- Random-precision robustness evaluation

---

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -U pip
pip install -r requirements.txt

# 3. Data Preparation
python scripts/01_download_squad.py

# 4. Step 1: Quantization Integration (Sanity Check)
export PYTHONPATH=$PWD
python scripts/10_step1_sanity.py

# 5. Step 2: Multi-LoRA Integration (Sanity Check)
export PYTHONPATH=$PWD
python scripts/11_step2_sanity.py

# 6. Step 3: Switchable Training
# Default: 1000 steps (see configs/training.yaml)
# Saves checkpoint to ./checkpoints/last.pt
export PYTHONPATH=$PWD
bash scripts/02_train_switchable.sh

# 7. Step 4: Evaluation
export PYTHONPATH=$PWD
export EVAL_DEVICE=cpu  
python scripts/03_eval_ppl.py --max_eval 1000 --batch_size 4 --calib_steps 50

# 8. Step 5: Cyclic Precision Training
export PYTHONPATH=$PWD
bash scripts/04_cyclic_precision.sh

# 9. Step 6: Robustness Evaluation
export PYTHONPATH=$PWD
bash scripts/05_robustness.sh

