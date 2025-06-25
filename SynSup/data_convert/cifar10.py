import os
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def setup_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_images_and_labels(dataset, folder, filename):
    labels_file = open(filename, 'w')
    for idx, (image, label) in enumerate(dataset):
        img_filename = f"{idx}.jpg"
        img_path = os.path.join(folder, img_filename)
        # Convert tensor to PIL image for saving
        pil_image = transforms.ToPILImage()(image)
        pil_image.save(img_path)
        labels_file.write(f"{img_filename}----{label}\n")
    labels_file.close()

def main():
    # Set up the directory structure
    base_dir = 'data'
    jpg_dir = os.path.join(base_dir, 'jpg')
    setup_directory(jpg_dir)

    # Define transformations: Convert data to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to tensor
    ])

    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Save images and labels
    save_images_and_labels(trainset, jpg_dir, os.path.join(base_dir, 'trainset.txt'))
    save_images_and_labels(testset, jpg_dir, os.path.join(base_dir, 'testset.txt'))

if __name__ == '__main__':
    main()