# # Dataset Preparation
from huggingface_hub import snapshot_download
# snapshot_download(repo_id="liuhaotian/llava-v1.5-7b", local_dir='/home/bscho333/Workspace/data/llava')
snapshot_download(repo_id="openai/clip-vit-large-patch14-336", local_dir='/home/bscho333/data/clip-vit-large-patch14-336')