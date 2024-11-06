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
### encode_image
`image_features = self.get_model().get_vision_tower()(images)`\
`encoded_img = self.get_model().mm_projector(image_features)`
```
encoded_img: torch.Size([1,  576,        1024])
                        (bs, patch_size, language_hidden_size)
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
