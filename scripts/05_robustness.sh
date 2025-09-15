source .venv/bin/activate
export PYTHONPATH=$PWD
export EVAL_DEVICE=cpu
python -m src.eval.robustness \
  --ckpt ./checkpoints/last_cyclic.pt \
  --profiles configs/bitwidth_uniform.yaml configs/bitwidth_mixed_small.yaml \
  --max_eval 500 \
  --batch_size 4 \
  --attack_strength 2 \
  --seed 123
