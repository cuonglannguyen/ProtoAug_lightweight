# ProtoAug lightweight
This repo is of ICML 2025 paper Provably Improving Generalization of Few-shot Models with Synthetic Data (the lightweight version).

Code repo is adapted from https://github.com/frotms/image_classification_pytorch

## Requirements
Python3 support only. Tested on CUDA9.0, cudnn7.

* albumentations==0.1.1
* easydict==1.8
* imgaug==0.2.6
* opencv-python==3.4.3.18
* protobuf==3.6.1
* scikit-image==0.14.0
* tensorboardX==1.4
* torch==0.4.1
* torchvision==0.2.1
* wandb
* diffusers
* transformers
* peft


### pre-trained model
you can download pretrain model with url in ($net-module.py)


### configuration
| configure                       | description                                                               |
|---------------------------------|---------------------------------------------------------------------------|
| model_module_name               | eg: vgg_module                                                            |
| model_net_name                  | net function name in module, eg:vgg16                                     |
| gpu_id                          | eg: single GPU: "0", multi-GPUs:"0,1,3,4,7"                                                           |
| async_loading                   | make an asynchronous copy to the GPU                                      |
| is_tensorboard                  | if use tensorboard for visualization                                      |
| evaluate_before_train           | evaluate accuracy before training                                         |
| shuffle                         | shuffle your training data                                                |
| data_aug                        | augment your training data                                                |
| img_height                      | input height                                                              |
| img_width                       | input width                                                               |
| num_channels                    | input channel                                                             |
| num_classes                     | output number of classes                                                  |
| batch_size                      | train batch size                                                          |
| dataloader_workers              | number of workers when loading data                                       |
| learning_rate                   | learning rate                                                             |
| learning_rate_decay             | learning rate decat rate                                                  |
| learning_rate_decay_epoch       | learning rate decay per n-epoch                                           |
| train_mode                      | eg:  "fromscratch","finetune","update"                                    |
| file_label_separator            | separator between data-name and label. eg:"----"                          |
| pretrained_path                 | pretrain model path                                                       |
| pretrained_file                 | pretrain model name. eg:"alexnet-owt-4df8aa71.pth"                        |
| pretrained_model_num_classes    | output number of classes when pretrain model trained. eg:1000 in imagenet |
| save_path                       | model path when saving                                                    |
| save_name                       | model name when saving                                                    |
| train_data_root_dir             | training data root dir                                                    |
| val_data_root_dir               | testing data root dir                                                     |
| train_data_file                 | a txt filename which has training data and label list                     |
| val_data_file                   | a txt filename which has testing data and label list                      |
| num_centroids                   | Number of generated centroids                                             |
| num_noises                      | Number of generated noises, should be enough with ratio                   |
| lam_dis                         | hyperparameter of discrepancy term                                        |
| lam_rob						  | hyperparameter of robust term											  |
| ce							  | hyperparameter of ce term, same for real and synthesis                    |
| fast_kmeans                     | implement fast and not so accurate kmeans or not. please set as false now |
| ratio                           | ratio of synthetic/real data 											  |
| lora                            | add lora to finetune stable diffusion or not. Implementing so set as false|
| ft_text						  | fine-tune text encoder of stable diffusion or not						  |
| lora_rank						  | rank of lora matrix, applied only to unet of stable diffusion			  |
| lora_alpha					  |	alpha parameter of lora													  |
| noise_optim					  | optimizing noise input of generator or not. Set as false while lora is true | 
| guidance_scale				  | guidance scale of Stable Diffusion models								  |

### Training
1.make your training &. testing data and label list with txt file:

txt file with single label index eg:

	apple.jpg----0
	k.jpg----3
	30.jpg----0
	data/2.jpg----1
	abc.jpg----1
2.configuration

3.train

	python3 train.py --config configs/clip_vit_{dataset}.json


