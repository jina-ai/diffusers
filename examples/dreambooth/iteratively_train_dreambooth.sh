max_train_steps=200

output_dir_base="/home/joschka/hf_dreambooth_output"
experiment_name="iteratively"
trained_model_path="/shared/ml_models/diffuser/stable-diffusion-v1-4/"

declare -a objects=(
  "briscoe4"
  "vexx4"
  "lnlwntrt"
)

declare -a instance_data_dirs=(
  "/home/joschka/workspace/diffusers/examples/dreambooth/data/briscoe4-png-512-cropped/"
  "/home/joschka/workspace/ColossalAI/examples/images/diffusion/data/vexx4-val-jpg-only/"
  "/home/joschka/workspace/ColossalAI/examples/images/diffusion/data/lnlwntrt-4-images-jpg/"
)

declare -a instance_prompts=(
  "a brc dog"
  "a sks painting"
  "a lnl painting"
)

for i in ${!instance_data_dirs[*]}; do
    experiment_name="${experiment_name}-${objects[$i]}_no_prior_8bit_${max_train_steps}steps"
    output_dir="${output_dir_base}/${experiment_name}/"
    accelerate launch train_dreambooth.py \
      --pretrained_model_name_or_path=${trained_model_path} \
      --output_dir=$output_dir \
      --instance_data_dir="${instance_data_dirs[$i]}" \
      --instance_prompt="${instance_prompts[$i]}" \
      --resolution=512 \
      --train_batch_size=1 \
      --learning_rate=5e-6 --lr_scheduler="constant" --lr_warmup_steps=0 \
      --max_train_steps=$max_train_steps \
      --gradient_accumulation_steps=2 --gradient_checkpointing --use_8bit_adam
    trained_model_path=${output_dir}
done
