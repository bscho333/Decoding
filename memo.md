# Config
## load_pretrained_model
### tokenizer
```
tokenizer: LlamaTokenizer(name_or_path='/root/data/llava-v1.5-7b', vocab_size=32000, model_max_length=2048, is_fast=False, padding_side='right', truncation_side='right', special_tokens={'bos_token': AddedToken("<s>", rstrip=False, lstrip=False, single_word=False, normalized=False), 'eos_token': AddedToken("</s>", rstrip=False, lstrip=False, single_word=False, normalized=False), 'unk_token': AddedToken("<unk>", rstrip=False, lstrip=False, single_word=False, normalized=False), 'pad_token': '<unk>'}, clean_up_tokenization_spaces=False)
```
### model
```
model_name: llava-v1.5-7b
```
```
model_path: /home/bscho333/data/llava-v1.5-7b
```
```
model.config: LlavaConfig {
  "_name_or_path": "/home/bscho333/data/llava-v1.5-7b",
  "architectures": [
    "LlavaLlamaForCausalLM"
  ],
  "bos_token_id": 1,
  "eos_token_id": 2,
  "freeze_mm_mlp_adapter": false,
  "freeze_mm_vision_resampler": false,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "image_aspect_ratio": "pad",
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_length": 4096,
  "max_position_embeddings": 4096,
  "mm_hidden_size": 1024,
  "mm_projector_type": "mlp2x_gelu",
  "mm_resampler_type": null,
  "mm_use_im_patch_token": false,
  "mm_use_im_start_end": false,
  "mm_vision_select_feature": "patch",
  "mm_vision_select_layer": -2,
  "mm_vision_tower": "openai/clip-vit-large-patch14-336",
  "model_type": "llava",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 32,
  "pad_token_id": 0,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.31.0",
  "tune_mm_mlp_adapter": false,
  "tune_mm_vision_resampler": false,
  "unfreeze_mm_vision_tower": false,
  "use_cache": true,
  "use_mm_proj": true,
  "vocab_size": 32000
}
```
```
model: LlavaLlamaForCausalLM(
  (model): LlavaLlamaModel(
    (embed_tokens): Embedding(32000, 4096, padding_idx=0)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
    (vision_tower): CLIPVisionTower(
      (vision_tower): CLIPVisionModel(
        (vision_model): CLIPVisionTransformer(
          (embeddings): CLIPVisionEmbeddings(
            (patch_embedding): Conv2d(3, 1024, kernel_size=(14, 14), stride=(14, 14), bias=False)
            (position_embedding): Embedding(577, 1024)
          )
          (pre_layrnorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          (encoder): CLIPEncoder(
            (layers): ModuleList(
              (0-23): 24 x CLIPEncoderLayer(
                (self_attn): CLIPAttention(
                  (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
                  (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
                  (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
                  (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
                )
                (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (mlp): CLIPMLP(
                  (activation_fn): QuickGELUActivation()
                  (fc1): Linear(in_features=1024, out_features=4096, bias=True)
                  (fc2): Linear(in_features=4096, out_features=1024, bias=True)
                )
                (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              )
            )
          )
          (post_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        )
      )
    )
    (mm_projector): Sequential(
      (0): Linear(in_features=1024, out_features=4096, bias=True)
      (1): GELU(approximate='none')
      (2): Linear(in_features=4096, out_features=4096, bias=True)
    )
  )
  (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
)
```
### image_processor
```
image_processor: CLIPImageProcessor {
  "crop_size": {
    "height": 336,
    "width": 336
  },
  "do_center_crop": true,
  "do_convert_rgb": true,
  "do_normalize": true,
  "do_rescale": true,
  "do_resize": true,
  "feature_extractor_type": "CLIPFeatureExtractor",
  "image_mean": [
    0.48145466,
    0.4578275,
    0.40821073
  ],
  "image_processor_type": "CLIPImageProcessor",
  "image_std": [
    0.26862954,
    0.26130258,
    0.27577711
  ],
  "resample": 3,
  "rescale_factor": 0.00392156862745098,
  "size": {
    "shortest_edge": 336
  }
}
```
### context_len
```
context_len: 2048
```

# Inference
## Tokenization
### qs
```
qs: 'Please describe this image in detail.'
```
### conv
```
conv: Conversation(system="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.", roles=('USER', 'ASSISTANT'), messages=[], offset=0, sep_style=<SeparatorStyle.TWO: 2>, sep=' ', sep2='</s>', version='v1', skip_next=False)
```
### qu_out
`qu_out = DEFAULT_IMAGE_TOKEN + '\n' + qs`
```
qu_out: '<image>\nPlease describe this image in detail.'
```
### conv.append
`conv.append_message(conv.roles[0], qu_out)`\
`conv.append_message(conv.roles[1], None)`
```
conv: Conversation(system="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.", roles=('USER', 'ASSISTANT'), messages=[['USER', '<image>\nPlease describe this image in detail.'], ['ASSISTANT', None]], offset=0, sep_style=<SeparatorStyle.TWO: 2>, sep=' ', sep2='</s>', version='v1', skip_next=False)
```
### prompt_out
```
prompt_out: "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nPlease describe this image in detail. ASSISTANT:"
```
### tokenizer_image_token
```
IMAGE_TOKEN_INDEX: -200
tokenizer_image_token(prompt_out, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    : tensor([    1,   319, 13563,  1546,   263, 12758,  5199,   322,   385, 23116,
            21082, 20255, 29889,   450, 20255,  4076,  8444, 29892, 13173, 29892,
              322,  1248,   568,  6089,   304,   278,  5199, 29915, 29879,  5155,
            29889,  3148,  1001, 29901, 29871,  -200, 29871,    13, 12148,  8453,
              445,  1967,   297,  9493, 29889,   319,  1799,  9047, 13566, 29901])
    .shape: torch.Size([50])
tokenizer_image_token(prompt_out, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
    : tensor([[    1,   319, 13563,  1546,   263, 12758,  5199,   322,   385, 23116,
             21082, 20255, 29889,   450, 20255,  4076,  8444, 29892, 13173, 29892,
               322,  1248,   568,  6089,   304,   278,  5199, 29915, 29879,  5155,
             29889,  3148,  1001, 29901, 29871,  -200, 29871,    13, 12148,  8453,
               445,  1967,   297,  9493, 29889,   319,  1799,  9047, 13566, 29901]])
    .shape: torch.Size([1, 50])
```
```
def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids
```
### input_ids
`input_ids = (
    tokenizer_image_token(prompt_out, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    .unsqueeze(0)
    .cuda()
)`
```
input_ids: torch.Size([1, 50])
    tensor([[    1,   319, 13563,  1546,   263, 12758,  5199,   322,   385, 23116,
             21082, 20255, 29889,   450, 20255,  4076,  8444, 29892, 13173, 29892,
               322,  1248,   568,  6089,   304,   278,  5199, 29915, 29879,  5155,
             29889,  3148,  1001, 29901, 29871,  -200, 29871,    13, 12148,  8453,
               445,  1967,   297,  9493, 29889,   319,  1799,  9047, 13566, 29901]],
           device='cuda:0')
"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: "
-> 35 token (1 + 34)
"<image>"
-> 1 token
"\nPlease describe this image in detail. ASSISTANT:"
-> 15 token (1 + 14)

=> 1(bos_token, 1) + 34(A chat..., 319 ~ 29871) + 1(image_token, -200) + 14(\nPlease..., 29871 ~ 29901) = 50 token
```
### stop_str
`stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2`
```
stop_str: '</s>'
```



## image
### images
`images, cfglst = image_parser(args)`
```
images: [<PIL.Image.Image image mode=RGB size=500x334>]
```
### image_sizes
`image_sizes = [image.size for image in images]`
```
image_sizes: [(500, 334)]
```
### process_images
```
process_images: <function llava.mm_utils.process_images(images, image_processor, model_cfg)>

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
        : 'pad'
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images
```
### images_tensor
```
images_tensor = process_images(
    images,
    image_processor,
    model.config
).to(model.device, dtype=torch.float16)

: torch.Size([1, 3, 336, 336])
```

## model.generate
### image encode
`encoded_img = self.get_model().get_vision_tower()(images)`
```
encoded_img: torch.Size([1,  576,        1024])
                        (bs, patch_size, vision_hidden_size)

tensor([[[ 0.7373,  0.2524,  0.4878,  ...,  0.0934,  0.7534,  1.0000],
          [ 0.5596, -0.8633,  0.5576,  ...,  0.7891,  0.4727,  0.9751],
          [ 2.0176,  1.1387, -0.3218,  ...,  2.9395,  0.2117,  2.2344],
          ...,
          [ 2.0449,  1.1309, -0.0826,  ...,  2.6816,  0.4646,  2.0977],
          [ 0.9775, -0.4370,  0.5537,  ...,  0.0339,  0.6714,  0.7505],
          [ 0.6289, -0.3303,  0.9082,  ...,  0.4570,  0.9321,  0.7837]]],
        dtype=torch.float16),
```

### image embed
`image_features = self.get_model().mm_projector(image_features)`
```
image_features: torch.Size([1,  576,        4096])
                            (bs, patch_size, language_hidden_size)

tensor([[[-0.1654,  0.3359, -0.4175,  ...,  0.0538,  0.5742,  0.0063],
          [-0.2148,  0.2190, -0.1888,  ..., -0.2893,  0.3076, -0.3088],
          [ 0.9995,  0.0498, -0.9438,  ...,  0.2761,  0.5044,  0.4685],
          ...,
          [ 0.7090,  0.1019, -0.9595,  ...,  0.1685,  0.4329,  0.2478],
          [-0.0754,  0.4036, -0.4861,  ..., -0.2462,  0.6230, -0.2517],
          [-0.2542,  0.2216, -0.3010,  ..., -0.5015,  0.6348, -0.4343]]],
        dtype=torch.float16, grad_fn=<ToCopyBackward0>)
```
## output
### output_ids
```
output_ids: torch.Size([1, 114])
    tensor([[    1,   319, 13563,  1546,   263, 12758,  5199,   322,   385, 23116,
             21082, 20255, 29889,   450, 20255,  4076,  8444, 29892, 13173, 29892,
               322,  1248,   568,  6089,   304,   278,  5199, 29915, 29879,  5155,
             29889,  3148,  1001, 29901, 29871,  -200, 29871,    13, 12148,  8453,
               445,  1967,   297,  9493, 29889,   319,  1799,  9047, 13566, 29901,
               450,  1967,  5680,   263,   767,   411,   263,  2071,   403,  3377,
               322,   263, 11203, 29892,  2845, 22049,  1623,   278, 11952,  4208,
               470, 24067,   278, 11952,  1550, 13587,  6567,   411,   278, 11203,
               373,   263,   454,  1161, 29889,   450,   767,  5692,   304,   367,
              2071,   403,  3377,   292,   411,   278, 11203,  3802,   491, 29892,
              4969,   263,  2090,   322, 19780, 25005, 29889, 29871,    13,    13,
              8439,   526,   263,  2846]], device='cuda:0')
```
### decode
`outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)`
```
['The image features a man with a skateboard and a dog, either walking down the street together or crossing the street while holding hands with the dog on a leash. The man appears to be skateboarding with the dog close by, creating a fun and friendly atmosphere. \n\nThere are a few']
```

### SampleDecoderOnlyOutput
```
sequences: torch.LongTensor = None
scores: Optional[Tuple[torch.FloatTensor]] = None
attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
```

# pdb `model.generate()`
## GPU usage
### `nvidia-smi`
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.107.02             Driver Version: 550.107.02     CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3090        Off |   00000000:A1:00.0 Off |                  N/A |
| 38%   34C    P8             21W /  350W |   17660MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
+-----------------------------------------------------------------------------------------+
```
## func `generate()`
Goes to `/root/miniconda3/envs/RITUAL/lib/python3.10/site-packages/transformers/generation/utils.py(1161)generate()`
### gemeration_config
```
GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "max_length": 4096,
  "pad_token_id": 0,
  "transformers_version": "4.31.0"
}

after generation_config.update(**kwargs)

GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "do_sample": true,
  "eos_token_id": 2,
  "max_length": 4096,
  "max_new_tokens": 64,
  "pad_token_id": 0,
  "top_k": null,
  "transformers_version": "4.31.0"
}

after max token update

GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "do_sample": true,
  "eos_token_id": 2,
  "max_length": 114,
  "max_new_tokens": 64,
  "pad_token_id": 0,
  "top_k": null,
  "transformers_version": "4.31.0"
}
```
### model_kwargs
```
{'images': tensor([[[[-0.0113, -0.0113, -0.0113,  ..., -0.0113, -0.0113, -0.0113],
          [-0.0113, -0.0113, -0.0113,  ..., -0.0113, -0.0113, -0.0113],
          [-0.0113, -0.0113, -0.0113,  ..., -0.0113, -0.0113, -0.0113],
          ...,
          [-0.0113, -0.0113, -0.0113,  ..., -0.0113, -0.0113, -0.0113],
          [-0.0113, -0.0113, -0.0113,  ..., -0.0113, -0.0113, -0.0113],
          [-0.0113, -0.0113, -0.0113,  ..., -0.0113, -0.0113, -0.0113]],

         [[-0.0112, -0.0112, -0.0112,  ..., -0.0112, -0.0112, -0.0112],
          [-0.0112, -0.0112, -0.0112,  ..., -0.0112, -0.0112, -0.0112],
          [-0.0112, -0.0112, -0.0112,  ..., -0.0112, -0.0112, -0.0112],
          ...,
          [-0.0112, -0.0112, -0.0112,  ..., -0.0112, -0.0112, -0.0112],
          [-0.0112, -0.0112, -0.0112,  ..., -0.0112, -0.0112, -0.0112],
          [-0.0112, -0.0112, -0.0112,  ..., -0.0112, -0.0112, -0.0112]],

         [[-0.0013, -0.0013, -0.0013,  ..., -0.0013, -0.0013, -0.0013],
          [-0.0013, -0.0013, -0.0013,  ..., -0.0013, -0.0013, -0.0013],
          [-0.0013, -0.0013, -0.0013,  ..., -0.0013, -0.0013, -0.0013],
          ...,
          [-0.0013, -0.0013, -0.0013,  ..., -0.0013, -0.0013, -0.0013],
          [-0.0013, -0.0013, -0.0013,  ..., -0.0013, -0.0013, -0.0013],
          [-0.0013, -0.0013, -0.0013,  ..., -0.0013, -0.0013, -0.0013]]]],
       device='cuda:0', dtype=torch.float16), 'images_pos': None, 'images_neg': None, 'use_ritual': False, 'use_vcd': False, 'use_m3id': False, 'ritual_alpha_pos': 3, 'ritual_alpha_neg': 1, 'ritual_beta': 0.1, 'output_attentions': False, 'output_hidden_states': False, 'use_cache': True}

accepts_attention_mask=True
requires_attention_mask=True
```
### _prepate_attention_mask_for_generation
```
pad_token_id = 0
eos_token_id = 2
is_pad_token_not_equal_to_eos_token_id = True
is_pad_token_in_inputs = False

```
### etc
```
self.config.is_encoder_decoder = False
input_ids_seq_length = 50
generation_config.max_length = 114
is_constraint_gen_mode = False
is_contrastive_search_gen_mode = False
is_greedy_gen_mode = False
is_sample_gen_mode = True
is_beam_sample_gen_mode = False
is_group_beam_gen_mode = False
is_assisted_gen_mode = False
```
```
logits_processor = []
stopping_criteria = [<transformers.generation.stopping_criteria.MaxLengthCriteria object at 0x7fe1b45097e0>]
logits_warper = []
```
### End of model.generate()
```
# 13. run sample
return self.sample(
    input_ids,
    logits_processor=logits_processor,
    logits_warper=logits_warper,
    stopping_criteria=stopping_criteria,
    pad_token_id=generation_config.pad_token_id,
    eos_token_id=generation_config.eos_token_id,
    output_scores=generation_config.output_scores,
    return_dict_in_generate=generation_config.return_dict_in_generate,
    synced_gpus=synced_gpus,
    streamer=streamer,
    **model_kwargs,
)
```
## sample
Goes to `/root/Decoding/ritual_utils/ritual_sample.py(23)sample()`
### args
```
input_ids,
    tensor([[    1,   319, 13563,  1546,   263, 12758,  5199,   322,   385, 23116,
         21082, 20255, 29889,   450, 20255,  4076,  8444, 29892, 13173, 29892,
           322,  1248,   568,  6089,   304,   278,  5199, 29915, 29879,  5155,
         29889,  3148,  1001, 29901, 29871,  -200, 29871,    13, 12148,  8453,
           445,  1967,   297,  9493, 29889,   319,  1799,  9047, 13566, 29901]],
       device='cuda:0')
logits_processor=logits_processor,
    <class 'transformers.generation.logits_process.LogitsProcessorList'>
        []
logits_warper=logits_warper,
    <class 'transformers.generation.logits_process.LogitsProcessorList'>
        []
stopping_criteria=stopping_criteria,
    <class 'transformers.generation.stopping_criteria.StoppingCriteriaList'>
        [<transformers.generation.stopping_criteria.MaxLengthCriteria object at 0x7fe1b45097e0>]
pad_token_id=generation_config.pad_token_id,
    0
eos_token_id=generation_config.eos_token_id,
    2
output_scores=generation_config.output_scores,
    False
return_dict_in_generate=generation_config.return_dict_in_generate,
    False
synced_gpus=synced_gpus,
    False
streamer=streamer,
    NONE
**model_kwargs,
    {'images': tensor([[[[-0.0113, -0.0113, -0.0113,  ..., -0.0113, -0.0113, -0.0113],
          [-0.0113, -0.0113, -0.0113,  ..., -0.0113, -0.0113, -0.0113],
          [-0.0113, -0.0113, -0.0113,  ..., -0.0113, -0.0113, -0.0113],
          ...,
          [-0.0113, -0.0113, -0.0113,  ..., -0.0113, -0.0113, -0.0113],
          [-0.0113, -0.0113, -0.0113,  ..., -0.0113, -0.0113, -0.0113],
          [-0.0113, -0.0113, -0.0113,  ..., -0.0113, -0.0113, -0.0113]],

         [[-0.0112, -0.0112, -0.0112,  ..., -0.0112, -0.0112, -0.0112],
          [-0.0112, -0.0112, -0.0112,  ..., -0.0112, -0.0112, -0.0112],
          [-0.0112, -0.0112, -0.0112,  ..., -0.0112, -0.0112, -0.0112],
          ...,
          [-0.0112, -0.0112, -0.0112,  ..., -0.0112, -0.0112, -0.0112],
          [-0.0112, -0.0112, -0.0112,  ..., -0.0112, -0.0112, -0.0112],
          [-0.0112, -0.0112, -0.0112,  ..., -0.0112, -0.0112, -0.0112]],

         [[-0.0013, -0.0013, -0.0013,  ..., -0.0013, -0.0013, -0.0013],
          [-0.0013, -0.0013, -0.0013,  ..., -0.0013, -0.0013, -0.0013],
          [-0.0013, -0.0013, -0.0013,  ..., -0.0013, -0.0013, -0.0013],
          ...,
          [-0.0013, -0.0013, -0.0013,  ..., -0.0013, -0.0013, -0.0013],
          [-0.0013, -0.0013, -0.0013,  ..., -0.0013, -0.0013, -0.0013],
          [-0.0013, -0.0013, -0.0013,  ..., -0.0013, -0.0013, -0.0013]]]],
       device='cuda:0', dtype=torch.float16), 'images_pos': None, 'images_neg': None, 'use_ritual': False, 'use_vcd': False, 'use_m3id': False, 'ritual_alpha_pos': 3, 'ritual_alpha_neg': 1, 'ritual_beta': 0.1, 'use_cache': True, 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1]], device='cuda:0')}
```

### initialize
```
logits_processor = []
stopping_criteria = [<transformers.generation.stopping_criteria.MaxLengthCriteria object at 0x7fe1b45097e0>]
unfinished_sequences = tensor([1], device='cuda:0')
model_inputs:
    {'input_ids': tensor([[    1,   319, 13563,  1546,   263, 12758,  5199,   322,   385, 23116,
             21082, 20255, 29889,   450, 20255,  4076,  8444, 29892, 13173, 29892,
               322,  1248,   568,  6089,   304,   278,  5199, 29915, 29879,  5155,
             29889,  3148,  1001, 29901, 29871,  -200, 29871,    13, 12148,  8453,
               445,  1967,   297,  9493, 29889,   319,  1799,  9047, 13566, 29901]],
           device='cuda:0'), 'past_key_values': None, 'use_cache': True, 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1]], device='cuda:0'), 'images': tensor([[[[-0.0113, -0.0113, -0.0113,  ..., -0.0113, -0.0113, -0.0113],
              [-0.0113, -0.0113, -0.0113,  ..., -0.0113, -0.0113, -0.0113],
              [-0.0113, -0.0113, -0.0113,  ..., -0.0113, -0.0113, -0.0113],
              ...,
              [-0.0113, -0.0113, -0.0113,  ..., -0.0113, -0.0113, -0.0113],
              [-0.0113, -0.0113, -0.0113,  ..., -0.0113, -0.0113, -0.0113],
              [-0.0113, -0.0113, -0.0113,  ..., -0.0113, -0.0113, -0.0113]],
    
             [[-0.0112, -0.0112, -0.0112,  ..., -0.0112, -0.0112, -0.0112],
              [-0.0112, -0.0112, -0.0112,  ..., -0.0112, -0.0112, -0.0112],
              [-0.0112, -0.0112, -0.0112,  ..., -0.0112, -0.0112, -0.0112],
              ...,
              [-0.0112, -0.0112, -0.0112,  ..., -0.0112, -0.0112, -0.0112],
              [-0.0112, -0.0112, -0.0112,  ..., -0.0112, -0.0112, -0.0112],
              [-0.0112, -0.0112, -0.0112,  ..., -0.0112, -0.0112, -0.0112]],
    
             [[-0.0013, -0.0013, -0.0013,  ..., -0.0013, -0.0013, -0.0013],
              [-0.0013, -0.0013, -0.0013,  ..., -0.0013, -0.0013, -0.0013],
              [-0.0013, -0.0013, -0.0013,  ..., -0.0013, -0.0013, -0.0013],
              ...,
              [-0.0013, -0.0013, -0.0013,  ..., -0.0013, -0.0013, -0.0013],
              [-0.0013, -0.0013, -0.0013,  ..., -0.0013, -0.0013, -0.0013],
              [-0.0013, -0.0013, -0.0013,  ..., -0.0013, -0.0013, -0.0013]]]],
           device='cuda:0', dtype=torch.float16)}
```
### flow
from `/root/Decoding/ritual_utils/ritual_sample.py(115)sample()`
```
outputs = self(
    **model_inputs,
    return_dict=True,
        True
    output_attentions=output_attentions, # True?
        no...False....
    output_hidden_states=output_hidden_states
        False
)
```
-> goes to `/root/Decoding/experiments/llava/model/language_model/llava_llama.py(57)forward()`\
\
from `/root/Decoding/experiments/llava/model/language_model/llava_llama.py(87)forward()`
```
input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)
```
-> calls `/root/Decoding/experiments/llava/model/llava_arch.py(87)prepare_inputs_labels_for_multimodal()`\
\
### prepare_inputs_labels_for_multimodal()
`/root/Decoding/experiments/llava/model/llava_arch.py(87)prepare_inputs_labels_for_multimodal()`\
```
image_token_indices: tensor([35], device='cuda:0')

```
```
cur_new_input_embeds: torch.Size([625, 4096])
    tensor([[ 0.0045, -0.0038,  0.0017,  ..., -0.0088,  0.0025, -0.0025],
            [-0.0112, -0.0129, -0.0121,  ...,  0.0090,  0.0118, -0.0081],
            [ 0.0195, -0.0058,  0.0061,  ...,  0.0171, -0.0052, -0.0212],
            ...,
            [-0.0187, -0.0017,  0.0177,  ...,  0.0238,  0.0052,  0.0101],
            [ 0.0066, -0.0161,  0.0117,  ..., -0.0103,  0.0148,  0.0073],
            [ 0.0039,  0.0015,  0.0055,  ..., -0.0042,  0.0151,  0.0024]],
           device='cuda:0', dtype=torch.float16)
new_input_embeds = [cur_new_input_embeds]
new_input_embeds = torch.stack(new_input_embeds, dim=0)
new_input_embeds.shape: torch.Size([1, 625, 4096])
```

`new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)`
```
new_attn_mask_pad_left: torch.Size([1, 575])
    tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
           device='cuda:0')
```
`attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)`
```
attention_mask: torch.Size([1, 625])
    tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1]], device='cuda:0')
```
returns to `/root/Decoding/experiments/llava/model/language_model/llava_llama.py(88)forward()`
### flow2
from `/root/Decoding/experiments/llava/model/language_model/llava_llama.py(90)forward()`
```
outputs = self.model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    past_key_values=past_key_values,
    inputs_embeds=inputs_embeds,
    use_cache=use_cache,
    output_attentions=output_attentions,
    output_hidden_states=output_hidden_states,
    return_dict=return_dict
)
```
-> goes to `/root/miniconda3/envs/RITUAL/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py(599)forward()`
