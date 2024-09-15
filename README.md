# Adaptive Rank Selection for Low-Rank Approximation of Language Models
This repository contains the implementation (unofficial) of the paper "Adaptive Rank Selections for Low-Rank Approximation of Language Models". The method introduces learnable neural networks to predict optimal decomposition ranks. This repo only implements the rank selection step - i.e it only implements [Algorithm 1: Adaptive Rank Selection](https://aclanthology.org/2024.naacl-long.13.pdf). Fine-tuning after rank selection is not implemented

<p align="center">
<img src="outline.png" alt="Outline Image" width="50%" />
  <p style="font-size: 14px; color: gray;">
    <a href="https://aclanthology.org/2024.naacl-long.13.pdf">Figure 1</a>
  </p>
</p>

## Implementation Caveats:
* Implementation of SVD: ASVD and Fisher SVD is implemented, doesn't implement IWSVD. IWSVD is utilized in final paper.

## Setup
* `conda create --name svd python=3.9; conda activate svd`
* `pip install -r requirements.txt`
	* install may fail of [eval harness](https://github.com/EleutherAI/lm-evaluation-harness). In that case, install from source as mentioned in their README

## Training 

```
LR=1e-3;
MODEL=meta-llama/Llama-2-7b-hf;

EXP_NAME=llama7b_llc_adaptive_90
COMP=0.90 # target compression

NUM_TRAIN_SAMPLES=70000
MAX_LEN=256
BATCH_SIZE=4

LTYPE=adaptive
GAMMA=0.001
LAMBDA=2.

python train_adaptive.py --model=$MODEL --target_param_ratio=0.90 --eval_full --batch_size=$BATCH_SIZE --lr=$LR --num_train_samples=$NUM_TRAIN_SAMPLES --exp_name=$EXP_NAME --max_length=$MAX_LEN --cache_dir=cache_dir --load_act_cache --eval_freq_steps=500 --eval_batch_size=4 --alpha=0.5 --lambda=$LAMBDA --gamma=$GAMMA

```
