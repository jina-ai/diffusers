folder="iteratively-briscoe4_no_prior_8bit_200steps"

GPU=1

path_model="/home/joschka/hf_dreambooth_output/$folder/"

declare -a prompts=(
                "a photo of an astronaut riding a horse on mars" "A fantasy landscape, trending on artstation"
                "a dog" "a golden retriever dog"
                "a painting" "a picasso painting" "a matisse painting" "a banksy painting"
                "a brc dog" "a picasso painting of a brc dog" "a brc dog painted by picasso"
                )

for prompt in "${prompts[@]}"
do
  CUDA_VISIBLE_DEVICES=$GPU python3 inference_dreambooth.py \
  --path_model $path_model \
  --prompt "$prompt" \
  --sub_dir $folder
done
echo "evaluated: ${folder}"




folder="iteratively-briscoe4_no_prior_8bit_200steps-vexx4_no_prior_8bit_200steps"

GPU=1

path_model="/home/joschka/hf_dreambooth_output/$folder/"

declare -a prompts=(
                "a photo of an astronaut riding a horse on mars" "A fantasy landscape, trending on artstation"
                "a dog" "a golden retriever dog"
                "a painting" "a picasso painting" "a matisse painting" "a banksy painting"
                "a sks painting" "a sks painting of a dog" "a sks painting of a cat flying" "a sks painting of a car"
                "a brc dog" "a picasso painting of a brc dog" "a brc dog painted by picasso"
                "a sks painting of a brc dog" "a brc dog painted by sks"
                )


for prompt in "${prompts[@]}"
do
  CUDA_VISIBLE_DEVICES=$GPU python3 inference_dreambooth.py \
  --path_model $path_model \
  --prompt "$prompt" \
  --sub_dir $folder
done
echo "evaluated: ${folder}"





folder="iteratively-briscoe4_no_prior_8bit_200steps-vexx4_no_prior_8bit_200steps-lnlwntrt_no_prior_8bit_200steps"

GPU=1

path_model="/home/joschka/hf_dreambooth_output/$folder/"

declare -a prompts=(
                "a photo of an astronaut riding a horse on mars" "A fantasy landscape, trending on artstation"
                "a dog" "a golden retriever dog"
                "a painting" "a picasso painting" "a matisse painting" "a banksy painting"
                "a sks painting" "a sks painting of a dog" "a sks painting of a cat flying" "a sks painting of a car"
                "a lnl painting" "a lnl painting of a dog" "a lnl painting of a cat flying" "a lnl painting of a car"
                "a brc dog" "a picasso painting of a brc dog" "a brc dog painted by picasso"
                "a sks painting of a brc dog" "a lnl painting of a brc dog" "a brc dog painted by sks" "a brc dog painted by lnl"
                )


for prompt in "${prompts[@]}"
do
  CUDA_VISIBLE_DEVICES=$GPU python3 inference_dreambooth.py \
  --path_model $path_model \
  --prompt "$prompt" \
  --sub_dir $folder
done
echo "evaluated: ${folder}"





