source .venv/bin/activate
accelerate launch -m src.training.loop_switchable \
  --base_cfg configs/gpt2_base.yaml \
  --train_cfg configs/training.yaml