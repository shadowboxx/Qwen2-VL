export CUDA_LAUNCH_BLOCKING=1

python web_demo_mm.py --server-name 0.0.0.0 --server-port 8080 \
--checkpoint-path /home/huyunliu/models/Qwen/Qwen2-VL-7B-Instruct-AWQ
