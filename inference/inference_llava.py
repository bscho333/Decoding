import os
import sys
import json
import random
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/experiments')
print(sys.path)
return 0

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, Conversation, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, KeywordsStoppingCriteria

from utils import dist_util
from utils.logger import create_logger
from glob import glob

import re
import requests
from io import BytesIO
from PIL import Image
from torchvision.transforms import v2

from coco_loader import COCODataset

# import kornia
from ritual_utils.ritual_sample import evolve_ritual_sampling
from ritual_utils.vcd_add_noise import add_diffusion_noise

evolve_ritual_sampling()
torch.multiprocessing.set_sharing_strategy('file_system')

import warnings
warnings.filterwarnings(action='ignore')

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def image_parser(args):
    data_list = [json.loads(file) for file in open(os.path.expanduser(args.image_list), "r")]
    imglst = []
    cfglst = []
    for data in data_list:
        image_file = data["image"]
        image = load_image(image_file, args.data_path)
        imglst.append(image)
        cfgdict = {}
        cfgdict["image_id"] = image_file
        # Add more configurations here
        cfglst.append(cfgdict)
    return imglst, cfglst


def load_image(image_file, image_folder=None):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    elif image_file.isdigit():
        image = Image.open(image_folder + "COCO_val2014_000000" + image_file + ".jpg").convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def parse_args():
    parser = argparse.ArgumentParser(description="Inference via LLaVA")
    parser.add_argument("--model_path", type=str, help="model")
    parser.add_argument("--model_base", type=str, default="llava")

    parser.add_argument("--conv_mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)

    parser.add_argument("--data_path", type=str, default="/mnt/server17_hard1/sangmin/data/coco/val2014/", help="data path")
    parser.add_argument("--anno_path", type=str, default="/mnt/server17_hard1/sangmin/data/coco/annotations/instances_val2014.json")
    parser.add_argument("--log_path", type=str, default="/mnt/server16_hard0/sangmin/code/neurips2024/logs/chair")
    parser.add_argument("--out_path", type=str, default="/mnt/server16_hard0/sangmin/code/neurips2024/chair_results/llava", help="output path")

    parser.add_argument("--image_list", type=str, default="/home/bscho333/Workspace/experiment/image_list.json")
    parser.add_argument("--question", type=str, default="Please describe this image in detail.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="num workers")

    parser.add_argument("--use_ritual", type=str2bool, default=False)

    parser.add_argument("--use_vcd", type=str2bool, default=False)
    parser.add_argument("--noise_step", type=int, default=500)
    
    parser.add_argument("--use_m3id", type=str2bool, default=False)
    
    parser.add_argument("--ritual_alpha_pos", type=float, default=3)
    parser.add_argument("--ritual_alpha_neg", type=float, default=1)
    parser.add_argument("--ritual_beta", type=float, default=0.1)

    parser.add_argument("--num_eval_samples", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--experiment_index", type=int, default=0)

    args = parser.parse_known_args()[0]
    return args


def main():
    args = parse_args()
    # Setup DDP:
    dist_util.setup_dist(args)
    device = dist_util.device()

    # Setup an experiment folder:
    if dist.get_rank() == 0:
        os.makedirs(
            args.log_path, exist_ok=True
        )  # Make results folder (holds all experiment subfolders)
        model_string_name = args.model_path.split("/")[-1]
        # experiment_index = len(glob(f"{args.log_path}/{model_string_name}/*")) + args.experiment_index
        experiment_index = args.experiment_index
        experiment_dir = f"{args.log_path}/{model_string_name}/{experiment_index}"  # Create an experiment folder
        os.makedirs(experiment_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # ========================================
    #             Model & Dataset
    # ========================================
    logger.info('Initializing Model')

    #### for ritual
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

    # chair_dataset = COCODataset(
    #     data_path=args.data_path,
    #     anno_path=args.anno_path,
    #     trans=image_processor,
    #     model=args.model_base
    # )
    # chair_loader = DataLoader(
    #     chair_dataset, 
    #     batch_size=args.batch_size, 
    #     shuffle=False, 
    #     num_workers=args.num_workers,
    #     drop_last=False
    # )

    os.makedirs(
        args.out_path, exist_ok=True
    )

    # ==============================================
    #               Augmentations
    # ==============================================
    aug_dict = {
    'horizontal flip':v2.RandomHorizontalFlip(p=1),
    'vertical flip':v2.RandomVerticalFlip(p=1),
    'rotation':v2.RandomRotation(degrees=180),
    'color jitter':v2.ColorJitter(brightness=1, contrast=1, saturation=1, hue=0.5),
    'gaussian blur':v2.GaussianBlur(kernel_size=13, sigma=(1.5, 2.0)),
    'crop':v2.RandomResizedCrop(size=336),
    }

    # For statistics
    pos_aug_counter = {k:0 for k in aug_dict}
    pos_aug_counter.update({None: 0})

    # ========================================
    #            Start Generation
    # ========================================
    logger.info("Start eval...")
    
    qs = args.question
    conv = conv_templates[args.conv_mode].copy()
    qu_out = DEFAULT_IMAGE_TOKEN + '\n' + qs
    conv.append_message(conv.roles[0], qu_out)
    conv.append_message(conv.roles[1], None)
    prompt_out = conv.get_prompt()
    input_ids = (
        tokenizer_image_token(prompt_out, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    images, cfglst = image_parser(args)
    image_sizes = [image.size for image in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)
    
    image_pos = None
    image_neg = None
    
    with torch.inference_mode():
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                images=images_tensor
                images_pos=(image_pos.unsqueeze(0).half().cuda() if image_pos is not None else None),
                images_neg=(image_neg.unsqueeze(0).half().cuda() if image_neg is not None else None),
                image_sizes=image_sizes,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                use_ritual=args.use_ritual,
                use_vcd=args.use_vcd,
                use_m3id=args.use_m3id,
                ritual_alpha_pos=args.ritual_alpha_pos,
                ritual_alpha_neg=args.ritual_alpha_neg,
                ritual_beta=args.ritual_beta,
            )
    # RITUAL CODE
    input_token_len = input_ids.shape[1]
    print("input_token_len", input_token_len)
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()

    logger.info(f"[VQA for ritual]")
    logger.info(f"V: {cfglst[0]["image_id"]}")
    logger.info(f"Q: {qs}")
    logger.info(f"A: {outputs}")
    logger.info(f"="*50)

    img_save = {}
    img_save["image_id"] = img_id.item()
    img_save["caption"] = outputs

    # dump metric file
    with open(os.path.join(args.out_path, f"exp_{experiment_index:03d}.jsonl"), "a") as f:
        json.dump(img_save, f)
        f.write('\n')
        



    
    
    
    
    
    
    
    logger.info(vars(args))

    if args.use_ritual:
        logger.info(f"RITUAL Transformation: {pos_aug_counter}")

if __name__ == "__main__":
    main()