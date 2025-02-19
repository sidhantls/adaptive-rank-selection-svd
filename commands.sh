# Define variables
MODEL="meta-llama/Llama-2-7b-hf"
BS=2
lr=1e-2
TRAIN_SAMPLES=70000
cache_dir="train_cache"
RATIO=0.85
lambda_scale=0.1 # scale to train over more timesteps
gamma_scale=0 # allignment loss
beta_scale=1.0
layer_type="struct_pruning"
r_loss="struct_pruning"

# Construct exp_name
exp_name="${MODEL##*/}_ratio${RATIO}_lambda${lambda_scale}_gamma${gamma_scale}_beta${beta_scale}_${layer_type}"

# python train_adaptive.py --model_name=$MODEL --batch_size=$BS --num_train_samples=$TRAIN_SAMPLES --max_length=256 --lr=$lr --alpha=0.5 --eval_freq_steps=2000 --exp_name=$exp_name --cache_dir=$cache_dir --eval_full --act_aware="activation" --target_param_ratio=$RATIO --lambda_scale=$lambda_scale --gamma_scale=$gamma_scale --layer_type=$layer_type --r_loss=$r_loss --beta_scale=$beta_scale

# Define arrays for loops
MODELS=("meta-llama/Llama-2-7b-hf" "meta-llama/Meta-Llama-3-8B" "google/gemma-7b")
MODELS=("google/gemma-7b")
RATIOS=(0.90 0.85 0.80)

# Nested loops for MODEL and RATIO
for MODEL in "${MODELS[@]}"; do
    for RATIO in "${RATIOS[@]}"; do
        # Construct exp_name
        exp_name="${MODEL##*/}_ratio${RATIO}_lambda${lambda_scale}_gamma${gamma_scale}_beta${beta_scale}_${layer_type}"
        
        echo "Running experiment with MODEL=$MODEL and RATIO=$RATIO"
        
        python train_adaptive.py --model_name=$MODEL --batch_size=$BS --num_train_samples=$TRAIN_SAMPLES --max_length=256 --lr=$lr --eval_freq_steps=2000 --exp_name=$exp_name --cache_dir=$cache_dir --eval_full --act_aware="activation" --target_param_ratio=$RATIO --lambda_scale=$lambda_scale --gamma_scale=$gamma_scale --layer_type=$layer_type --r_loss=$r_loss --beta_scale=$beta_scale
    done
done
