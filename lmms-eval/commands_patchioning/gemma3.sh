export HF_HOME="~/.cache/huggingface"
# pip3 install transformers==4.57.1 (Qwen3VL models)
# pip3 install ".[qwen]" (for Qwen's dependencies)

# Exmaple with Qwen3-VL-4B-Instruct: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct 

export HF_TOKEN="xxx"  # replace xxx with your huggingface token

# Trace crop evaluation
CUDA_VISIBLE_DEVICES=0,2,4,5 accelerate launch --num_processes=4 --main_process_port=12346 -m lmms_eval \
    --model=gemma3 \
    --model_args=pretrained=google/gemma-3-4b-it \
    --tasks "coco_karpathy_test_trace_crop" \
    --batch_size 1 \
    --predict_only \
    --output_path results_patchioning

# Trace Visual Prompting evaluation
CUDA_VISIBLE_DEVICES=0,2,4,5 accelerate launch --num_processes=4 --main_process_port=12347 -m lmms_eval \
    --model=gemma3 \
    --model_args=pretrained=google/gemma-3-4b-it \
    --tasks "coco_karpathy_test_trace_vp" \
    --batch_size 1 \
    --predict_only \
    --output_path results_patchioning

# Region-set crop evaluation
CUDA_VISIBLE_DEVICES=0,2,4,5 accelerate launch --num_processes=4 --main_process_port=12348 -m lmms_eval \
    --model=gemma3 \
    --model_args=pretrained=google/gemma-3-4b-it \
    --tasks "coco_karpathy_test_region-set_crop" \
    --batch_size 1 \
    --predict_only \
    --output_path results_patchioning

# Region-set Visual Prompting evaluation
CUDA_VISIBLE_DEVICES=0,2,4,5 accelerate launch --num_processes=4 --main_process_port=12349 -m lmms_eval \
    --model=gemma3 \
    --model_args=pretrained=google/gemma-3-4b-it \
    --tasks "coco_karpathy_test_region-set_vp" \
    --batch_size 1 \
    --predict_only \
    --output_path results_patchioning


# Dense Crop evaluation
CUDA_VISIBLE_DEVICES=0,2,4,5 accelerate launch --num_processes=4 --main_process_port=12350 -m lmms_eval \
    --model=gemma3 \
    --model_args=pretrained=google/gemma-3-4b-it \
    --tasks "coco_karpathy_test_dense_crop" \
    --batch_size 1 \
    --predict_only \
    --output_path results_patchioning 


# Dense Visual Prompting evaluation
CUDA_VISIBLE_DEVICES=0,2,4,5 accelerate launch --num_processes=4 --main_process_port=12351 -m lmms_eval \
    --model=gemma3 \
    --model_args=pretrained=google/gemma-3-4b-it \
    --tasks "coco_karpathy_test_dense_vp" \
    --batch_size 1 \
    --predict_only \
    --output_path results_patchioning 


# Image Captioning evaluation
CUDA_VISIBLE_DEVICES=0,2,4,5 accelerate launch --num_processes=4 --main_process_port=12352 -m lmms_eval \
    --model=gemma3 \
    --model_args=pretrained=google/gemma-3-4b-it \
    --tasks "coco_karpathy_test" \
    --batch_size 1 \
    --predict_only \
    --output_path results_patchioning 
