source .venv/bin/activate
export PYTHONPATH=$PWD
export USE_LORA=1            
export SUBSET_FRAC=0.3      
accelerate launch -m src.training.loop_cyclic \
  --base_cfg configs/gpt2_base.yaml \
  --train_cfg configs/training.yaml \
  --verbose 1
