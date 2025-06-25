"""
Code mainly from https://github.com/vturrisi/disef/blob/main/fine-tune/src/model.py
"""

import types
import torch
import torch.nn as nn
import clip
from timm.models._manipulate import checkpoint_seq
from typing import Any, Optional, Tuple, Union
import torch.utils.checkpoint
from torch.utils.checkpoint import checkpoint

from .lora import lora_replace_attention_layers, lora_replace_attention_layers_clip, lora_replace_attention_layers_sd_CLIP, lora_replace_attention_layers_clip_text_encoder
from .lora_unet import lora_replace_unet_attention_layers
from .lora_clip_text_encoder import lora_replace_sd_attention_layers
from .util_data import SUBSET_NAMES, TEMPLATES_SMALL
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from trainers.base_trainer import BaseTrainer
from utils import utils
import math
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionXLPipeline
#from diffusers import DiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from torch.distributed.optim import DistributedOptimizer
import os
import json
from transformers import CLIPTokenizer
from torchvision.transforms import v2
import random
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from diffusers.utils.torch_utils import is_compiled_module
from accelerate import Accelerator
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers
from peft import LoraConfig, get_peft_model
from torch.distributed.rpc import RRef
import torch.distributed as dist
from torchviz import make_dot
# Initialize CLIP tokenizer
from diffusers import DiffusionPipeline
from transformers import AutoTokenizer, PretrainedConfig
def get_dataset_name_for_template(dataset):
    dataset_name = {
        "imagenet_100": "",
        "imagenet": "",
        "std10": "",
        "pets": "pet ",
        "fgvc_aircraft": "aircraft ",
        "cars": "car ",
        "eurosat": "satellite ",
        "dtd": "texture ",
        "flowers102": "flower ",
        "food101": "food ",
        "sun397": "scene ",
        "caltech101": "",
    }[dataset]
    return dataset_name

    
class CLIP(nn.Module):
    def __init__(
        self, 
        dataset,
        is_lora_image,
        is_lora_text,
        config,
        clip_download_dir="model_clip",
        clip_version="ViT-B/16",
        
    ):
        super().__init__()
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.distributed = self.config['distributed']
        self.guidance_scale = self.config['guidance_scale']
        self.lora = self.config['lora']
        self.ft_text = self.config['ft_text']
        self.label_smoothing = self.config['label_smoothing']
        self.dpm_solver = self.config['dpm_scheduler']
        self.sd_version = self.config['base']
        self.ratio = self.config['ratio']

        if self.sd_version == "2-1":
            self.gen_name = "stabilityai/stable-diffusion-2-1"
        elif self.sd_version == "2-1-base":
            self.gen_name = "stabilityai/stable-diffusion-2-1-base"
        else:
            self.gen_name = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        if self.lora:
            self.lora_rank = self.config['lora_rank']
            self.lora_alpha = self.config['lora_alpha']
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.gen_name, subfolder="tokenizer"
            )
            self.text_encoder_cls = import_model_class_from_model_name_or_path(
            self.gen_name, revision=None
            )
            self.text_encoder = self.text_encoder_cls.from_pretrained(
                self.gen_name, subfolder="text_encoder", 
            )
            
            self.vae = AutoencoderKL.from_pretrained(
                self.gen_name, subfolder="vae"
            )
            
            self.unet = UNet2DConditionModel.from_pretrained(
                self.gen_name, subfolder="unet"
            )

            # Freeze parameters of models to save memory
            self.unet.requires_grad_(False)
            self.vae.requires_grad_(False)
            self.text_encoder.requires_grad_(False)

            # Add LoRA adapters to the UNet (we assume you have an 'add_adapter' method)
        # Set correct lora layers

            #for param in self.unet.parameters():
            #    param.requires_grad_(False)
            self.unet.to(self.device, dtype=torch.float16)
            if self.vae is not None:
                self.vae.to(self.device, dtype=torch.float16)
            self.text_encoder.to(self.device, dtype=torch.float16)
            self.unet.enable_gradient_checkpointing()
            self.text_encoder.gradient_checkpointing_enable()
            
            self.unet_lora_config = LoraConfig(
                inference_mode=False,
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0","add_k_proj", "add_v_proj"],
                )            
            
            self.unet= get_peft_model(self.unet, self.unet_lora_config)
            #self.unet.add_adapter(self.unet_lora_config)
            if self.ft_text:
                self.text_lora_config = LoraConfig(
                    inference_mode=False,
                    r=16,
                    lora_alpha=16,
                    init_lora_weights="gaussian",
                    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
                )
                self.text_encoder= get_peft_model(self.text_encoder, self.text_lora_config)
                #self.text_encoder.add_adapter(self.text_lora_config)
            """
            #self.lora_layers = list(filter(lambda p: p.requires_grad, self.unet.parameters()))
            #if self.ft_text:
            #    self.lora_layers = self.lora_layers + list(
            #        filter(lambda p: p.requires_grad, self.text_encoder.parameters())
            #    )
                #print(sel f.lora_layers)
            """
            if self.dpm_solver:
                self.pipeline = DiffusionPipeline.from_pretrained(
                        self.gen_name,
                        unet=self.unet,
                        safety_checker=None,
                        text_encoder=self.text_encoder,
                        torch_dtype=torch.float16
                    )

                    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
                scheduler_args = {}

                if "variance_type" in self.pipeline.scheduler.config:
                    variance_type = self.pipeline.scheduler.config.variance_type

                    if variance_type in ["learned", "learned_range"]:
                        variance_type = "fixed_small"

                    scheduler_args["variance_type"] = variance_type

                self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    self.pipeline.scheduler.config, **scheduler_args
                )
                #self.pipeline.scheduler.set_timesteps(20)
                self.pipeline = self.pipeline.to(self.device)

            else:
                self.pipeline =   StableDiffusionPipeline(
                    self.gen_name,
                    pretrained_model_revision=None,
                    use_lora=True,
                )
            #TODO: Add pipeline
        else:
            self.pipeline = StableDiffusionPipeline.from_pretrained(self.gen_name, safety_checker = None,\
                                                                    torch_dtype = torch.float16, guidance_scale= self.guidance_scale, add_watermarker = False).to(self.device)
        self.dataset = dataset
        self.dataset_name = get_dataset_name_for_template(dataset)
        self.dataset = self.config['dataset']
        self.prompt_specific = self.config['prompt_specific']
        self.meta = self.config['meta']
        with open(self.meta, 'r') as f:
            self.class_labels = f.read().splitlines()
        self.is_lora_image = is_lora_image
        self.is_lora_text = is_lora_text
        self.clip_version = clip_version
        self.num_centroids = self.config['num_centroids']
        self.num_noise_vectors = self.config['num_noise']
        self.channel = self.config['num_channels']
        self.img_height = self.config['img_height']
        self.img_width = self.config['img_width']
        self.clip_use = self.config['clip']      
        self.num_classes = self.config['num_classes']
        if self.clip_use:
            self.img_size = (3, 224, 224)
        else:
            self.img_size = (self.channel, self.img_height, self.img_width)
        self.labels_size = (1, self.num_classes)
        self.lam_dis = self.config['lam_dis']
        self.lam_rob = self.config['lam_rob']
        self.ce = self.config['ce']
        self.fast_kmeans = self.config['fast_kmeans']
        self.cutmix = self.config['cutmix']
        self.mixup = self.config['mixup']
        self.centroids = self.initialize_centroids()
        self.noise_vectors = self.initialize_noise()
        self.batch_size = self.config['batch_size']
        self.clip_version = clip_version
        self.noise_vectors = self.initialize_noise()


        # TODO: change the number of templates
        self.templates = TEMPLATES_SMALL[:1]
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            self.clip, _ = clip.load(clip_version, device="cuda", download_root=clip_download_dir)

        # visual model
        if is_lora_image:
            if self.clip_version != "RN50":
                self.clip.visual.transformer = lora_replace_attention_layers(
                    self.clip.visual.transformer,
                    lora_r=16,
                    lora_alpha=32,
                    lora_dropout=0.1,
                    start_block=0,
                )
        #self.unet = lora_replace_sd_attention_layers(
        #    self.unet,
        #    lora_r=16,
        #    lora_alpha=32,
        #    lora_dropout=0.1
        #)
        """ 
        self.text_encoder = lora_replace_attention_layers_clip_text_encoder(
            self.text_encoder,
                lora_r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                start_block=0
        ) 
        """
        if is_lora_text:
            self.clip.transformer = lora_replace_attention_layers(
                self.clip.transformer,
                lora_r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                start_block=0,
            )
        self.register_buffer("tokenized_text", self.tokenize_text())

        # enable checkpointing for text transformer
        # datasets with more classes simply go OOM if we don't do this
        def checkpoint_forward(self, x):
            x.requires_grad = True
            x = checkpoint_seq(self.resblocks, x)
            return x

        self.clip.transformer.forward = types.MethodType(
            checkpoint_forward, self.clip.transformer)


        self.set_learnable_params()
    def initialize_centroids(self):
        """
        Initialize centroids with random noise matching the image size.
        Each centroid is a tensor representing an 'average' image.
        """
        # Initialize centroids with random values, could also use a different distribution
        centroids = torch.randn((self.num_centroids, *self.img_size), device=self.device)
        return centroids
    def initialize_noise(self):
        """
        Initialize noise vectors with random noise matching the image size.
        Each noise vector can be used to generate synthetic images.
        """
        # Initialize noise vectors with normal distribution
        noise_vectors = torch.randn((self.num_noise_vectors, 4, *tuple(int(val//8) for val in self.img_size[1:])), requires_grad = True, device=self.device)
        #print("Noise_vectors_shape", noise_vectors.shape)
        return noise_vectors
    def get_prompt(self, use_caption, class_names, image_names = None,caption_path=None,batch_size=None):
        if use_caption:
            base_prompts = [f"photo of {c}" for c in class_names]
            caption_dict = load_caption_dict(image_names, caption_path)
            caption_suffix = [caption_dict[f"{image_name.split('/')[-1]}.JPEG"] for image_name in image_names]
            prompts = [f"{base_prompts[n]}, {caption_suffix[n]}, best quality" for n in range(batch_size)]
        else:
            if self.prompt_specific:
                if self.dataset == "dtd":
                    prompts = [f"A photo of a {c} texture" for c in class_names]
                elif self.dataset == "eurosat":
                    prompts = [f"A centered satellite photo of a {c}" for c in class_names]
                elif self.dataset == "cars":
                    prompts = [f"A photo of a {c}, a type of car" for c in class_names]
                elif self.dataset == "fgvc_aircraft":
                    prompts = [f"A photo of a {c}, a type of aircraft" for c in class_names]
                elif self.dataset == "flowers102":
                    prompts = [f"A photo of a {c} flower" for c in class_names]
                elif self.dataset == "pets":
                    prompts = [f"A photo of a {c}, a type of pet" for c in class_names]
                else:
                    prompts = [f"A photo of a {c}" for c in class_names]
            else:
                prompts = [f"A photo of a {c}" for c in class_names]
        return prompts
    def generate_data(self, noise_vector,negative_prompts, class_names):
        """
        Generate synthetic data from noise vector using the pretrained model.
        """
        prompt = self.get_prompt(use_caption=False, class_names=class_names)

        transform = transforms.ToTensor()
        with torch.autocast(device_type=self.device, dtype=torch.float16):
            generated_image = self.pipeline(
                unet = self.unet,
                text_encoder = self.text_encoder,
                prompt=prompt,
                latents=noise_vector,
                negative_prompts=negative_prompts,
                output_type = 'pt'
            ).images
            #print(generated_image)

        #img_tensor_list = [transform(img).to(self.device).requires_grad_(True) for img in generated_image]

        #img_cluster = torch.stack(img_tensor_list, dim = 0)  # Stacks images into a single tensor if needed
        #
        return generated_image
    def tokenize_text(self):
        print("Tokenizing text...")

        texts = []
        for classname in SUBSET_NAMES[self.dataset]:

            class_texts = []
            for template in self.templates:
                class_texts.append(
                    template.format(self.dataset_name, classname))

            class_texts = clip.tokenize(class_texts)

            texts.append(class_texts)

        texts = torch.stack(texts)
        return texts

    def set_learnable_params(self):
        # turn off all parameters
        for p in self.clip.parameters():
            p.requires_grad = False
        for p in self.unet.parameters():
            p.requires_grad = False
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        # learnable parameters for the visual model
        if self.is_lora_image:
            if self.clip_version != "RN50":
                for name, p in self.clip.visual.named_parameters():
                    if "lora_" in name:
                        p.requires_grad = True
            else:
                for name, p in self.clip.visual.named_parameters():
                    p.requires_grad = True
                
#         elif not self.cfg.freeze_visual:
#             for p in self.clip.visual.parameters():
#                 p.requires_grad = True

        # learnable parameters for the text model
        if self.is_lora_text:
            for name, p in self.clip.transformer.named_parameters():
                if "lora_" in name:
                    p.requires_grad = True
        for name, p in self.unet.named_parameters():
            if "lora_" in name:
                p.requires_grad = True
        for name, p in self.text_encoder.text_model.named_parameters():
            if "lora_" in name:
               p.requires_grad = True
        '''
        for module in self.unet.modules():
            if hasattr(module, "transformer_blocks"):
                for block in module.transformer_blocks:
                    # Enable LoRA parameters for Self-Attention
                    if hasattr(block, "attn1"):
                        for name, param in block.attn1.named_parameters():
                            if "lora_" in name:
                                param.requires_grad = True

                    # Enable LoRA parameters for Cross-Attention
                    if hasattr(block, "attn2"):
                        for name, param in block.attn2.named_parameters():
                            if "lora_" in name:
                                param.requires_grad = True
'''

        #trainable_params, total_params, trainable_param_count = prepare_model_for_lora_training(self.unet)
        #print(f"Trainable parameters: {trainable_param_count:,}")



    def learnable_params(self):
#         return [{"name": "all", "params": [p for p in self.clip.parameters() if p.requires_grad]}]
        return [p for p in self.clip.parameters() if p.requires_grad]

    def forward_image(
        self,
        x: torch.Tensor,
    ):
        image_feats = self.clip.visual(x)
        image_feats = image_feats / image_feats.norm(dim=1, keepdim=True)
        return image_feats

    def forward_text(self, tokenized_text):
        #device = self.clip.transformer.resblocks[0].ln_1.weight.device

        n_classes, n_prompts, n_token = tokenized_text.size()
#         tokenized_text = einops.rearrange(tokenized_text, "c p d -> (c p) d")
        tokenized_text = tokenized_text.view(-1, n_token)
        #tokenized_text = tokenized_text.to(self.clip.transformer.resblocks[0].attn.in_proj_weight.device)
        #tokenized_text = tokenized_text.to(device)


        with torch.set_grad_enabled(self.is_lora_text):
            text_feats = self.clip.encode_text(tokenized_text)

        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        # average across multiple prompt templates and re-norm
#         text_feats = einops.rearrange(text_feats, "(c p) d -> c p d", c=n_classes, p=n_prompts)
        text_feats = text_feats.view(n_classes, n_prompts, -1)
        text_feats = text_feats.mean(dim=1)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        return text_feats

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        iter_index,
        state: str,
        tokenized_text: torch.Tensor = None,
        output_features: bool = False,
        **kwargs,):
        if state == 'train':
            self.batch_size = len(x)
            start_idx = iter_index * self.batch_size
            end_idx = start_idx + self.ratio * self.batch_size
            print(start_idx)
            print(end_idx)
            #print(self.noise_vectors[0])

            # Slice the noise_vectors to get the appropriate batch (128, 4, 4, 4)
            #u_A = self.noise_vectors[int(start_idx/ratio): int(end_idx/ratio)]
            u_A = self.noise_vectors[start_idx:end_idx]
            class_names = [self.class_labels[idx] for idx in labels]
            class_names = self.ratio * class_names
            negative_prompts = ["artistic, distorted, unrealistic, blurry, out of frame, cropped, deformed" for n in range(self.batch_size)]
            G_A = self.generate_data(u_A, negative_prompts, class_names)
            label_gen = labels.repeat(self.ratio)

            label_origin = labels
            label_origin = label_origin.cuda(non_blocking=True)
            if self.cutmix and self.mixup:
               p = random.random()
               if p >= 0.2:
                   pass
               else:
                   cutmix = v2.CutMix(num_classes=self.num_classes)
                   mixup = v2.MixUp(num_classes=self.num_classes)
                   cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
                   
                   x, labels = cutmix_or_mixup(x, labels)
                   G_A, label_gen = cutmix_or_mixup(G_A, label_gen)
            G_A = G_A.squeeze(1).to(torch.float16).cuda(non_blocking=True)
            label_gen = label_gen.cuda(non_blocking=True)
            image_gen_feats = self.forward_image(G_A)
            #print("generate requires grad", G_A.grad)


        if tokenized_text is None:
            tokenized_text = self.tokenized_text

        image_feats = self.forward_image(x)
        text_feats = self.forward_text(tokenized_text)
        

        logit_scale = self.clip.logit_scale.exp()

        # no instance-specific text feats
        if len(text_feats.shape) == 2:
            # cosine similarity as logits
            logits_per_image = logit_scale * image_feats @ text_feats.t()
            if state == 'train':

                gen_logits_per_image = logit_scale * image_gen_feats @ text_feats.t()
        else:
            logits_per_image = logit_scale * torch.stack(
                [image_feats[i] @ text_feats[i].t() for i in range(image_feats.shape[0])]
            )
            if state == 'train':

                gen_logits_per_image = logit_scale * torch.stack(
                    [image_gen_feats[i] @ text_feats[i].t() for i in range(image_gen_feats.shape[0])]
                )

        if output_features:
            return {
                "logits": logits_per_image,
                "image_feats": image_feats,
                "text_feats": text_feats,
            }
        if state == 'train':

            return logits_per_image, gen_logits_per_image, G_A, label_origin,label_gen
        else: 
            return logits_per_image
def clip_vit(**kwargs):
    dataset = kwargs.get('dataset')
    config = kwargs.get('config')
    model = CLIP(dataset=dataset, is_lora_image=True, config=config,is_lora_text=True, clip_version="ViT-B/16")
    return model
def clip_resnet(**kwargs):
    model = CLIP(dataset='caltech101', is_lora_image=True, is_lora_text=True, clip_version="RN50")
    return model
accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision='fp16'
    )
def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import (
            RobertaSeriesModelWithTransformation,
        )

        return RobertaSeriesModelWithTransformation
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")

