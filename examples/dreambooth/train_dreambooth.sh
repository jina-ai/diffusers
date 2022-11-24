max_train_steps=800

output_dir="/home/joschka/hf_dreambooth_output/briscoe4-vexx4-lnlw-no_prior-8bit-"$max_train_steps"steps/"


accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path="/shared/ml_models/diffuser/stable-diffusion-v1-4/" \
  --output_dir=$output_dir \
  --instance_data_dir="/home/joschka/workspace/diffusers/examples/dreambooth/data/briscoe4-png-512-cropped/,/home/joschka/workspace/ColossalAI/examples/images/diffusion/data/vexx4-val-jpg-only/,/home/joschka/workspace/ColossalAI/examples/images/diffusion/data/lnlwntrt-4-images-jpg/" \
  --instance_prompt="a brc dog,a sks painting,a lnl painting" \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=5e-6 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --max_train_steps=$max_train_steps \
  --gradient_accumulation_steps=2 --gradient_checkpointing --use_8bit_adam



#  --instance_data_dir="/home/joschka/workspace/ColossalAI/examples/images/diffusion/data/vexx-val-jpg-only/,/home/joschka/workspace/ColossalAI/examples/images/diffusion/data/lnlwntrt-4-images-jpg/" \
#  --instance_prompt="a sks painting,a lnl painting" \


# --instance_data_dir="/home/joschka/workspace/ColossalAI/examples/images/diffusion/data/vexx-val-jpg-only/,/home/joschka/workspace/ColossalAI/examples/images/diffusion/data/lnlwntrt-4-images-jpg/" \
# --instance_data_dir="/home/joschka/workspace/ColossalAI/examples/images/diffusion/data/lnlwntrt-4-images-jpg/" \

# normal optimization
# #  --gradient_accumulation_steps=1


# for 8bit optimization -> < 16 GB VRAM training
# #  --gradient_accumulation_steps=2 --gradient_checkpointing --use_8bit_adam


# deepspeed -> < 8 GB VRAM training
# #  --gradient_accumulation_steps=1 --gradient_checkpointing \
# #  --mixed_precision=fp16
