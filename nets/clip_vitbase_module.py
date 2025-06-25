import torch
import clip.clip as clip
import os
import pickle
class ImageEncoder(torch.nn.Module):
    def __init__(self, model_name, device='cuda'):
        super().__init__()
        self.model, self.train_preprocess= clip.load(model_name, device, jit=False)
        if hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')  # Remove the language transformer

    def forward(self, images):
        return self.model.encode_image(images)

    def save(self, filename):
        print(f'Saving image encoder to {filename}')
        torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image encoder from {filename}')
        return torch_load(filename)


class ClassificationHead(torch.nn.Linear):
    def __init__(self, num_classes, input_dim, normalize=False):
        super().__init__(input_dim, num_classes)
        self.normalize = normalize

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def save(self, filename):
        print(f'Saving classification head to {filename}')
        torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading classification head from {filename}')
        return torch_load(filename)


class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_head, process_images=True):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        self.process_images = process_images

    def forward(self, inputs):
        if self.process_images:
            inputs = self.image_encoder(inputs)
        outputs = self.classification_head(inputs)
        return outputs

    def save(self, filename):
        print(f'Saving image classifier to {filename}')
        torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image classifier from {filename}')
        return torch_load(filename)


def clip_classifier_resnet( **kwargs):
    # Initialize the image encoder (ResNet-based)
    num_classes = kwargs.get('num_classes', 1000)
    model_name="RN50"
    device='cuda'
    image_encoder = ImageEncoder(model_name=model_name, device=device)

    # Initialize the classification head to output 'num_classes' logits from 64-dim embeddings
    classification_head = ClassificationHead(num_classes=num_classes, input_dim=64, normalize=True)

    # Create the final image classifier
    model = ImageClassifier(image_encoder=image_encoder, classification_head=classification_head)\
        #.load("/cm/shared/cuongnl8/Cuong-thesis/nets_pretrained/RN50.pt")
    return model
def torch_save(classifier, save_path):
    if os.path.dirname(save_path) != '':
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(classifier.cpu(), f)


def torch_load(save_path, device=None):
    with open(save_path, 'rb') as f:
        classifier = torch.jit.load(f)
    if device is not None:
        classifier = classifier.to(device)
    return classifier