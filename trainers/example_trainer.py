# coding=utf-8
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
from diffusers import DiffusionPipeline
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
from peft import LoraConfig
import numpy as np
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from data_loader.data_processor import DataProcessor
from PIL import Image
import random
from PIL import ImageFilter, ImageOps
import torchvision.transforms.v2 as transforms_v2
from torchvision.transforms import AugMix
# Initialize CLIP tokenizer
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
def load_caption_dict(image_names,caption_path):
    class_lis = set([image_name.split("/")[0] for image_name in image_names])
    dict_lis = []
    for class_name in class_lis:
        with open(os.path.join(caption_path,f"{class_name}.json"), 'r') as file:
            dict_lis.append(json.load(file))
    caption_dict = {key: value for dictionary in dict_lis for key, value in dictionary.items()}
    return caption_dict
def squared_norm(x):
    """
    Computes and returns the squared norm of the input 2d tensor on dimension 1.
    Useful for computing euclidean distance matrix.

    Parameters
    ----------
    x : torch.Tensor of shape (n, m)

    Returns
    -------
    x_squared_norm : torch.Tensor of shape (n, )
    """
    return (x ** 2).sum(1).view(-1, 1)

class ExampleTrainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader, config, logger):
        set_seed(22)
        super(ExampleTrainer, self).__init__(model, train_loader, val_loader, config, logger)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.guidance_scale = self.config['guidance_scale']
        self.lora = self.config['lora']
        self.lora_dd = self.config['lora_dd']
        self.ft_text = self.config['ft_text']
        self.sd_version = self.config['base']
        if self.sd_version == "2-1":
            self.gen_name = "stabilityai/stable-diffusion-2-1"
        elif self.sd_version == "2-1-base":
            self.gen_name = "stabilityai/stable-diffusion-2-1-base"
        else:
            self.gen_name = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        if self.lora:
            self.lora_rank = self.config['lora_rank']
            self.lora_alpha = self.config['lora_alpha']
            self.tokenizer = CLIPTokenizer.from_pretrained(
                self.gen_name, subfolder="tokenizer"
            )
            
            self.text_encoder = CLIPTextModel.from_pretrained(
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

            for param in self.unet.parameters():
                param.requires_grad_(False)

            self.unet_lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0","add_k_proj", "add_v_proj"],
                )            
            self.unet.add_adapter(self.unet_lora_config)
            if self.ft_text:
                self.text_lora_config = LoraConfig(
                    r=16,
                    lora_alpha=32,
                    init_lora_weights="gaussian",
                    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
                )
                self.text_encoder.add_adapter(self.text_lora_config)
            
            self.lora_layers = list(filter(lambda p: p.requires_grad, self.unet.parameters()))
            if self.ft_text:
                self.lora_layers = self.lora_layers + list(
                    filter(lambda p: p.requires_grad, self.text_encoder.parameters())
                )
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                        self.gen_name,
                        unet=unwrap_model(self.unet),
                        #safety_checker =None,
                        text_encoder = unwrap_model(self.text_encoder),
                        torch_dtype=torch.float16,
                    ).to(self.device)
            #TODO: Add pipeline
        elif self.lora_dd:
            self.pipeline = StableDiffusionPipeline.from_pretrained(self.gen_name, safety_checker = None)
            self.pipeline.load_lora_weights("", weight_name = "pytorch_lora_weights.safetensors")
            self.pipeline.to(self.device)
        else:
            self.pipeline = StableDiffusionPipeline.from_pretrained(self.gen_name, safety_checker = None,\
                                                                    torch_dtype = torch.float16, guidance_scale= self.guidance_scale, add_watermarker = False).to(self.device)
        #Read class labels for cifar10, modify for other dataset
        self.meta = self.config['meta']
        with open(self.meta, 'r') as f:
            self.class_labels = f.read().splitlines()
        self.num_centroids = self.config['num_centroids']
        self.num_noise_vectors = self.config['num_noise']
        self.channel = self.config['num_channels']
        self.img_height = self.config['img_height']
        self.img_width = self.config['img_width']
        self.clip = self.config['clip']      
        self.num_classes = self.config['num_classes']
        if self.clip:
            self.img_size = (3, 256, 256)
        elif self.config['model_module_name'] == 'resnet_module':
            self.img_size = (3, 256, 256)
        else:
            self.img_size = (self.channel, self.img_height, self.img_width)
        self.lam_dis = self.config['lam_dis']
        self.lam_rob = self.config['lam_rob']
        self.ce = self.config['ce']
        self.fast_kmeans = self.config['fast_kmeans']
        self.centroids = self.initialize_centroids()
        self.noise_vectors = self.initialize_noise()
        self.batch_size = self.config['batch_size']
        self.dataset = self.config['dataset']
        self.counts = [0] * self.num_centroids
        self.ratio = self.config['ratio']
        self.cutmix = self.config['cutmix']
        self.mixup = self.config['mixup']
        self.ce_real = self.config['ce_real']
        self.ce_syn = self.config['ce_syn']
        self.noise_optim = self.config['noise_optim']
        self.create_optimization()
        self.generate = torch.zeros((self.num_noise_vectors, 3,224,224), device=self.device, dtype=torch.float16)

    def train_epoch(self):
        """
        training in a epoch
        :return: 
        """
        # Learning rate adjustment
        cur_step = -1
        self.discrepancy_loss = utils.AverageMeter()
        self.robustness_loss = utils.AverageMeter()
        self.train_losses = utils.AverageMeter()
        self.train_top1 = utils.AverageMeter()
        self.train_top5 = utils.AverageMeter()

        self.model.net.train()
        start_idx = 0
        for batch_idx, (batch_x, batch_y) in enumerate(self.train_loader):
            cur_step +=1
            self.learning_rate = self.adjust_learning_rate(optimizer=self.optimizer, epoch=self.cur_epoch, cur_step=cur_step)

            start_idx += self.ratio * len(batch_y)
            end_idx = start_idx + self.ratio * len(batch_y)


            if torch.cuda.is_available():
                batch_x, batch_y = batch_x.cuda(non_blocking=self.config['async_loading']), batch_y.cuda(non_blocking=self.config['async_loading'])
            batch_x_var, batch_y_var = Variable(batch_x), Variable(batch_y)
            self.train_step(batch_x_var, batch_y_var, centroids = self.centroids, batch_size=self.batch_size, start=start_idx - self.ratio * len(batch_y), end = end_idx -  self.ratio * len(batch_y))

            # printer
            self.logger.log_printer.iter_case_print(self.cur_epoch, self.eval_train, self.eval_validate,
                                                    len(self.train_loader), batch_idx+1, self.train_losses.avg, self.learning_rate)

            # tensorboard summary
            if self.config['is_tensorboard']:
                self.logger.summarizer.data_summarize(batch_idx, summarizer="train", summaries_dict={"lr":self.learning_rate, 'train_loss':self.train_losses.avg})

        time.sleep(1)


    def train_step(self, images, labels, centroids, batch_size, start, end):
        """
        training in a step
        :param images: 
        :param labels: 
        :return: 
        """
        # Forward pass
        with torch.autocast(device_type=self.device, dtype=torch.float16):
            if self.lora:
                self.unet.train()
            infer = self.model.net(images)

            class_names = [self.class_labels[idx] for idx in labels]
            class_names = self.ratio * class_names
            self.TS = {i: [[],[]] for i in range(self.num_centroids)}  # Empty sets for each region/centroid

            negative_prompts = ["artistic, distorted, unrealistic, blurry, out of frame, cropped, deformed" for n in range(batch_size)]

            if self.cur_epoch == 1:
                gen_imgs = self.generate_data(negative_prompts, class_names)
                #print(gen_imgs.shape)
                self.generate[start:end] = gen_imgs
            else:
                gen_imgs = self.generate[start:end]
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
                    
                    images, labels = cutmix_or_mixup(images, labels)
                    gen_imgs, label_gen = cutmix_or_mixup(gen_imgs, label_gen)
            G_infer = self.model.net(gen_imgs)


            cluster_assignment = self.assign_to_clusters(gen_imgs, centroids)
            real_cluster_assignment = self.assign_to_clusters(images, centroids)

            centroids = self.update_centroids(centroids=centroids, data=gen_imgs, cluster_assignments=cluster_assignment, eta=self.counts)

            self.TS, self.counts = self.update_TS_and_counts(real_assignment=real_cluster_assignment, generated_assignments=cluster_assignment)


            # STEP 3: Loss function modification. 
            losses, discrepancy_loss, robustness_loss = self.get_loss(pred_real=infer, label_real=labels, pred_gen=G_infer, label_gen=label_gen)

            loss = losses.item()


            # measure accuracy and record loss
            prec1, prec5 = self.compute_accuracy(infer.data, label_origin.data, topk=(1, 5))
            self.train_losses.update(loss, images.size(0))
            self.discrepancy_loss.update(discrepancy_loss, images.size(0))
            self.robustness_loss.update(robustness_loss,images.size(0))
            self.train_top1.update(prec1[0], images.size(0))
            self.train_top5.update(prec5[0], images.size(0))

            if torch.cuda.device_count() > 1 and torch.cuda.is_available():
                self.optimizer.module.zero_grad()
            else:
                self.optimizer.zero_grad()

            if self.lora:
                self.lora_optimizer.zero_grad()
            losses.backward()
            if torch.cuda.device_count() > 1 and torch.cuda.is_available():
                self.optimizer.module.step()
            else:
                self.optimizer.step()
            if self.lora:
                self.lora_optimizer.step()



    def get_loss(self, pred_real, label_real, pred_gen, label_gen):
        """
        Compute the total loss across all regions in TS as described in equation (18).

        :param pred_real: Predictions for real data S.
        :param label_real: Labels for real data S.
        :param pred_gen: Predictions for generated data G.
        :param label_gen: Labels for generated data G.
        :return: Total loss.
        """
        # Criterion (cross-entropy loss)
        criterion = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            criterion = criterion.cuda()

        # Term 1: F(S, h) - Real data loss
        loss_real = criterion(pred_real, label_real)

        # Term 2: F(G, h) - Generated data loss
        loss_gen = criterion(pred_gen, label_gen)

        # Initialize discrepancy and regularization losses
        total_discrepancy_loss = 0
        total_robustness_loss = 0

        # Total number of generated samples (g is the total count across all regions)
        g = len(label_gen)

        for region_index, region in self.TS.items():

            num_real_regions = len(self.TS[region_index][0])  # Size of the region

            #TODO: Compute only samples in region i
            if num_real_regions > 0:
                for i in self.TS[region_index][0]:
                    for j in self.TS[region_index][1]:
                        discrepancy_loss = (1 / (g * num_real_regions)) * \
                                                F.mse_loss(pred_real[i], pred_gen[j])  # MSE loss for discrepancy
                        total_discrepancy_loss += discrepancy_loss

                for i in self.TS[region_index][1]:
                    for j in self.TS[region_index][1]:
                        robust_loss = (1/g) * (1/len(self.TS[region_index][1])) * F.mse_loss(pred_gen[i], pred_gen[j])
                        total_robustness_loss += robust_loss

        total_loss = self.ce_real * (loss_real) + self.ce_syn * loss_gen + self.lam_dis * total_discrepancy_loss + self.lam_rob * total_robustness_loss
        print("Discrepancy",total_discrepancy_loss)
        print("Robustness", total_robustness_loss)
        return total_loss, total_discrepancy_loss, total_robustness_loss
    def initialize_centroids(self):
        """
        Initialize centroids with random noise matching the image size.
        Each centroid is a tensor representing an 'average' image.
        """
        # Initialize centroids with random values, could also use a different distribution
        centroids = torch.randn((self.num_centroids, 3,224,224), device=self.device)
        return centroids
    def initialize_noise(self):
        """
        Initialize noise vectors with random noise matching the image size.
        Each noise vector can be used to generate synthetic images.
        """
        # Initialize noise vectors with normal distribution
        noise_vectors = torch.randn((self.num_noise_vectors, 4, *tuple(int(val//8) for val in self.img_size[1:])), device=self.device)
        return noise_vectors
    def generate_data(self,negative_prompts, class_names):
        """
        Generate synthetic data from noise vector using the pretrained model.
        """
        CLIP_NORM_MEAN = (0.48145466, 0.4578275, 0.40821073)
        CLIP_NORM_STD = (0.26862954, 0.26130258, 0.27577711)   
        prompt = self.get_prompt(use_caption=False, class_names=class_names)
        aux_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                    )
                ],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(0.2),
            Solarization(0.2),
        ])
        transform_train = transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.RandAugment(),
            transforms.RandomResizedCrop(
                224, 
                scale=(0.25, 1.0), 
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=None,
            ),
            aux_transform,
            transforms.ToTensor(),
            transforms.Normalize(CLIP_NORM_MEAN, CLIP_NORM_STD)
        ])
     
        
        with torch.autocast(device_type=self.device, dtype=torch.float16):
            if self.lora:
                generated_image = self.pipeline(
                    height = 256,
                    width = 256,
                    unet = unwrap_model(self.unet),
                    text_encoder = unwrap_model(self.text_encoder),
                    prompt=prompt,
                    guidance_scale = self.guidance_scale,
                    negative_prompts=negative_prompts,
                    #output_type = 'pt',
                ).images
            else:
                generated_image = self.pipeline(
                    prompt=prompt,
                    guidance_scale = self.guidance_scale,
                    negative_prompts=negative_prompts,
                    #output_type = 'pt',
                ).images
        
        img_tensor_list = [transform_train(img) for img in generated_image]
        del generated_image
        return torch.stack(img_tensor_list).to(self.device)

    def get_prompt(self, use_caption, class_names, image_names = None,caption_path=None,batch_size=None):
        if use_caption:
            base_prompts = [f"photo of {c}" for c in class_names]
            caption_dict = load_caption_dict(image_names, caption_path)
            caption_suffix = [caption_dict[f"{image_name.split('/')[-1]}.JPEG"] for image_name in image_names]
            prompts = [f"{base_prompts[n]}, {caption_suffix[n]}, best quality" for n in range(batch_size)]
        elif self.config['dd_prompt']:
            if self.dataset == "dtd":
                prompts = [f"A texture photo of a {c}" for c in class_names]
            elif self.dataset == "eurosat":
                prompts = [f"A satellite photo of a {c}" for c in class_names]
            elif self.dataset == "cars":
                prompts = [f"A car photo of a {c}r" for c in class_names]
            elif self.dataset == "fgvc_aircraft":
                prompts = [f"A aircraft photo of a {c}" for c in class_names]
            elif self.dataset == "flowers102":
                prompts = [f"A flower photo of a {c}" for c in class_names]
            elif self.dataset == "pets":
                prompts = [f"A pet photo of a {c}" for c in class_names]
            elif self.dataset == "food":
                prompts = [f"A food photo of a {c}" for c in class_names]
            elif self.dataset == "sun397":
                prompts = [f"A scene photo of a {c}" for c in class_names]
            else:
                prompts = [f"A photo of a {c}" for c in class_names]
        else:
            prompts = [f"A photo of a {c}" for c in class_names]
            
        return prompts
    def assign_to_clusters(self, data, centroids):
        """
        Assign generated data points to the nearest centroid clusters.
        """
        # Placeholder for assigning points to clusters based on centroids
        batch_size = data.size(0)
        num_centroids = centroids.size(0)
        data = data.cpu()
        centroids = centroids.cpu()
        # Calculate the Euclidean distance between each data point and each centroid
        data_expanded = data.unsqueeze(1)  # (batch_size, 1, img_channel, img_height, img_width)
        centroids_expanded = centroids.unsqueeze(0)  # (1, num_centroids, img_channel, img_height, img_width)

        squared_diffs = (data_expanded - centroids_expanded) ** 2

        # Sum the squared differences over the last 3 dimensions
        sum_squared_diffs = squared_diffs.sum(dim=(-1, -2, -3))

        # Take the se root to get the Euclidean distance
        distances = torch.sqrt(sum_squared_diffs)        
        # Assign each data point to the nearest centroid
        cluster_assignment = torch.argmin(distances, dim=1)  # (batch_size)
        
        return cluster_assignment
    
    def update_Kmeans_lr(self, cluster_assignment, num_centroids):
        """
        Update the learning rate based on centroid size or distance.
        """
        # Placeholder logic for learning rate update
        counts = torch.bincount(cluster_assignment, minlength=num_centroids)
        
        # Avoid division by zero by ensuring at least 1 count per centroid
        counts = torch.clamp(counts, min=1)
        
        # Calculate the learning rate as 1 / number of points assigned to each centroid
        new_eta = 1.0 / counts.float()
        return new_eta

    def update_centroids(self, centroids, data, cluster_assignments, eta):
        """
        Update the centroids with new data assignments.
        """

        if self.fast_kmeans:
            for c in range(self.num_centroids):
                idx = torch.where(cluster_assignments == c)[0]
                self.counts[c] += len(idx)
                if len(idx) > 0:
                    lr = 1 / self.counts[c]
                    self.centroids[c] = ((1 - lr) * self.centroids[c]) + \
                                                    (lr * torch.sum(torch.index_select(data, 0, idx), dim=0))
        else: 
            for k, i in enumerate(cluster_assignments):
                G_k = data[k]
                #print("Generate samples", G_k.shape)
                #print("Eta", eta)
                self.counts[i] +=1
                eta[i] = 1/self.counts[i]
                self.centroids[i] = (1 - eta[i]) * self.centroids[i] + eta[i] * G_k


        return self.centroids

    def generate_noise(self):
        """
        Generate noise from a normal distribution for the next batch.
        """
        return torch.randn(self.noise_dim).to(self.device)

    def generate_data_with_noise(self, cluster, noise):
        """
        Generate samples using the updated noise vector.
        """
        return self.generate_data(noise_vector=cluster + noise)

    def update_TS_and_counts(self, real_assignment, generated_assignments):
        """
        Update any tracking structures like T_S and cluster counts.
        """
        for k,i in enumerate(real_assignment):
            #print(k)
            #print(i)
            #print(self.TS)
            self.TS[i.item()][0].append(k)
        for k, i in enumerate(generated_assignments):
            self.TS[i.item()][1].append(k)
            #self.counts[i] += 1
        return self.TS, self.counts
    def update_model(self, loss):
        """
        Backpropagation: Update the model parameters based on the computed loss.
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def create_optimization(self):
        """
        optimizer
        :return: 
        """
        if self.config['optimizer'] == "Adam":
            self.optimizer = torch.optim.Adam(self.model.net.parameters(),
                                          lr=self.config['learning_rate'], weight_decay=self.config['weight_decay']) #lr:1e-4
        elif self.config['optimizer'] == "SGD":
            self.optimizer = torch.optim.SGD(self.model.net.parameters(),
                                             lr=self.config['learning_rate'],weight_decay = self.config['weight_decay'], momentum = self.config['momentum'])
        elif self.config['optimizer'] == "AdamW":
            self.optimizer = torch.optim.AdamW(self.model.net.parameters(),
                                          lr=self.config['learning_rate'], weight_decay=self.config['weight_decay']) 
        if torch.cuda.device_count() > 1:
            print('optimizer device_count: ',torch.cuda.device_count())
            self.optimizer = nn.DataParallel(self.optimizer,device_ids=range(torch.cuda.device_count()))

        if self.lora:
            self.lora_optimizer = torch.optim.AdamW(self.lora_layers, lr = 0.0001, weight_decay = 0.0005)


    def adjust_learning_rate(self, optimizer, epoch, cur_step):
        """
        decay learning rate
        :param optimizer: 
        :param epoch: the first epoch is 1
        :return: 
        """

        eta_max = self.config['learning_rate']
        eta_min = self.config.get('learning_rate_min', 0)
        num_epochs = self.config['num_epochs']
        if self.config['scheduler'] == 'stepLR':
            learning_rate = self.config['learning_rate'] * (self.config['learning_rate_decay'] ** ((epoch - 1) // self.config['learning_rate_decay_epoch']))
        if self.config['scheduler'] == 'cosine_annealing':

            
            T_cur = epoch - 1
            T_max = num_epochs
            
            learning_rate = eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * T_cur / T_max))
        if self.config['scheduler'] == 'warmup_cosine_annealing':
            total_steps_per_epoch = self.config['num_classes']
            warmup_epochs = self.config.get('warmup_epochs', 3)
            start_warmup_value = self.config.get('learning_rate_min', 0)
            current_iter = (epoch - 1) * total_steps_per_epoch + cur_step
            total_iters = num_epochs * total_steps_per_epoch
            warmup_iters = warmup_epochs * total_steps_per_epoch
            if current_iter < warmup_iters:  # Linear warm-up phase
                learning_rate = start_warmup_value + current_iter / warmup_iters * (eta_max - start_warmup_value)
            else:  # Cosine annealing phase
                T_cur = current_iter - warmup_iters
                T_max = total_iters - warmup_iters
                learning_rate = eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * T_cur / T_max))

        if torch.cuda.device_count() > 1 and torch.cuda.is_available():
            for param_group in optimizer.module.param_groups:
                param_group['lr'] = learning_rate
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

        return learning_rate


    def compute_accuracy(self, output, target, topk=(1,)):
        """
        compute top-n accuracy
        :param output: 
        :param target: 
        :param topk: 
        :return: 
        """
        maxk = max(topk)
        batch_size = target.size(0)
        _, idx = output.topk(maxk, 1, True, True)
        idx = idx.t()
        correct = idx.eq(target.view(1, -1).expand_as(idx))
        acc_arr = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            acc_arr.append(correct_k.mul_(1.0 / batch_size))
        return acc_arr


    def evaluate_epoch(self):
        """
        evaluating in a epoch
        :return: 
        """
        self.eval_losses = utils.AverageMeter()
        self.eval_top1 = utils.AverageMeter()
        self.eval_top5 = utils.AverageMeter()
        # Set the model to be in testing mode (for dropout and batchnorm)
        self.model.net.eval()
        for batch_idx, (batch_x, batch_y) in enumerate(self.val_loader):
            if torch.cuda.is_available():
                batch_x, batch_y = batch_x.cuda(non_blocking=self.config['async_loading']), batch_y.cuda(non_blocking=self.config['async_loading'])
            batch_x_var, batch_y_var = Variable(batch_x), Variable(batch_y)
            self.evaluate_step(batch_x_var, batch_y_var)
            utils.view_bar(batch_idx+1, len(self.val_loader))


    def evaluate_step(self, images, labels):
        """
        evaluating in a step
        :param images: 
        :param labels: 
        :return: 
        """
        with torch.no_grad():
            with torch.autocast(device_type=self.device, dtype=torch.float16):

                image_logits = self.model.net(images)

                # Loss function
                criterion = nn.CrossEntropyLoss()
                if torch.cuda.is_available():
                    criterion = criterion.cuda()

                loss = criterion(image_logits, labels)

        # measure accuracy and record loss
        prec1, prec5 = self.compute_accuracy(image_logits.data, labels.data, topk=(1, 5))

        self.eval_losses.update(loss, images.size(0)) # loss.data[0]
        self.eval_top1.update(prec1[0], images.size(0))
        self.eval_top5.update(prec5[0], images.size(0))

accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision='fp16'
    )
def unwrap_model(model):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model
def set_seed(seed):
    """function sets the seed value
    Args:
        seed (int): seed value
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # if you are suing GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __repr__(self):
        return "{}(p={}, radius_min={}, radius_max={})".format(
            self.__class__.__name__, self.p, self.radius_min, self.radius_max
        )

    def __call__(self, img):
        if random.random() <= self.p:
            radius = random.uniform(self.radius_min, self.radius_max)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __repr__(self):
        return "{}(p={})".format(self.__class__.__name__, self.p)

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
