# Adaptive Rank Selection for Low-Rank Approximation of Language Models
This repository contains the implementation two differentiable low-rank compression methods:
* Adaptive Rank Selections for Low-Rank Approximation of Language Models: [paper](https://aclanthology.org/2024.naacl-long.13.pdf)
* Learning to Low-Rank Compress: [paper](https://openreview.net/pdf?id=960Ny6IjEr)

<p align="center">
<img src="outline.png" alt="Outline Image" width="50%" />
  <p style="font-size: 14px; color: gray;">
    <a href="https://aclanthology.org/2024.naacl-long.13.pdf">Figure 1</a>
  </p>
</p>

## Implementation Details
* **Adaptive Rank Selections for Low-Rank Approximation of Language Models:**
  * Unofficial implementation of the paper *"Adaptive Rank Selections for Low-Rank Approximation of Language Models"*. The method introduces learnable neural networks to predict optimal decomposition ranks. This repository only implements the rank selection step—i.e., it only implements [Algorithm 1: Adaptive Rank Selection](https://aclanthology.org/2024.naacl-long.13.pdf). Fine-tuning after rank selection is not implemented.
  * **Caveats:**
    * In the main branch, the rank selection layer differs from the original work, assigning one GRU to each layer. Refer to the `fix_hypernet` branch for the exact implementation, where one GRU is used overall, and only linear projection layers are assigned per layer for mask prediction.
    * Implementation of SVD: ASVD and Fisher SVD are implemented here, while IWSVD is not. IWSVD is used in the final paper.
  * **Rank Selection Layer:** [Module](https://github.com/sidhantls/adaptive-rank-selection-svd/blob/a2be7e398a6fa2a78fc5c049dde36fba6a20b258/utils/adaptive_rank_selection.py#L59)

* **Learning to Low-Rank Compress:**
  * This includes an implementation of *Learning to Low-Rank Compress*. There are some simplifications to make the codebase more uniform across both implementations—for example, the distillation objective and total variation loss from the original work are not included. However, as noted in the Appendix, using a pre-training loss provides similar performance (albeit slightly lower).
  * **Rank Selection Layer:** [Module](https://github.com/sidhantls/adaptive-rank-selection-svd/blob/a2be7e398a6fa2a78fc5c049dde36fba6a20b258/utils/adaptive_rank_selection.py#L196)


## Setup
* `conda create --name svd python=3.9; conda activate svd`
* `pip install -r requirements.txt`
	* install may fail of [eval harness](https://github.com/EleutherAI/lm-evaluation-harness). In that case, install from source as mentioned in their README

## Training
### Training Script for Adaptive Rank Selection

```
# constants
NUM_TRAIN_SAMPLES=50000
MAX_LEN=256
BETA=1.
ACT_AWARE=activation
COMP_VALUES=(0.90 0.85 0.80)
EVAL_BS=8
BATCH_SIZE=4
LTYPE=adaptive
R_LOSS=default
LR=1e-3

MODEL=meta-llama/Llama-2-7b-hf
CACHE_DIR=cache_train_llama2
LAMBDA=16.
GAMMA=1.

#MODEL=meta-llama/Meta-Llama-3-8B
#CACHE_DIR=cache_train_llama
#LAMBDA=8.
#GAMMA=2.

#MODEL=google/gemma-7b
#CACHE_DIR=cache_train_gemma
#LAMBDA=8.
#GAMMA=2.

# Loop over the COMP values
for i in ${!COMP_VALUES[@]}; do
    COMP=${COMP_VALUES[$i]}
    EXP_NAME="${MODEL#*/}_${LTYPE}_${COMP}_fixmse_${GAMMA}_${LAMBDA}"
    p_param=0.4
    # Check if it's the first iteration
    if [ $i -eq 0 ]; then
        # Command for the first iteration without extra arguments
        python train_adaptive.py --model=$MODEL --target_param_ratio=$COMP --eval_full --batch_size=$BATCH_SIZE --lr=$LR --num_train_samples=$NUM_TRAIN_SAMPLES --exp_name=$EXP_NAME --max_length=$MAX_LEN --cache_dir=$CACHE_DIR --eval_freq_steps=500 --eval_batch_size=$EVAL_BS --alpha=0.5 --lambda=$LAMBDA --gamma=$GAMMA --act_aware=$ACT_AWARE  --layer_type=$LTYPE --beta_scale=$BETA --r_loss=$R_LOSS --tau=0.4 --p_param=$p_param
    else
        python train_adaptive.py --model=$MODEL --target_param_ratio=$COMP --eval_full --batch_size=$BATCH_SIZE --lr=$LR --num_train_samples=$NUM_TRAIN_SAMPLES --exp_name=$EXP_NAME --max_length=$MAX_LEN --cache_dir=$CACHE_DIR --eval_freq_steps=500 --eval_batch_size=$EVAL_BS --alpha=0.5 --lambda=$LAMBDA --gamma=$GAMMA --act_aware=$ACT_AWARE --layer_type=$LTYPE --beta_scale=$BETA --r_loss=$R_LOSS --tau=0.4 --p_param=$p_param --load_act_cache
    fi
done
```

### Training Script for Learning to Low-Rank Compress
For this, we can use `layer_type="simple"`

```
LTYPE=simple
R_LOSS=default
LR=1e-2
gamma_scale=0.  # there's no allignment loss, set scale to 0 
lambda_scale=1. # compression scale 
beta_scale=0.5  # pre-training scale
```
