path_model="/home/joschka/hf_dreambooth_output/vexx_no_prior_8bit/"

CUDA_VISIBLE_DEVICES=2 python3 inference_dreambooth.py \
  --path_model $path_model \
  --prompt "$1"