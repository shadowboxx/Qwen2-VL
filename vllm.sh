export CUDA_LAUNCH_BLOCKING=1

python -m vllm.entrypoints.openai.api_server --port 8080 \
--served-model-name Qwen2-VL-7B-Instruct \
--model /home/huyunliu/models/Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4 \

# --quantization gptq \