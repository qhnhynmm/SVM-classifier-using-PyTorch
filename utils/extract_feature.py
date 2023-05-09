import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import AutoTokenizer, AutoFeatureExtractor

class FeatureExtractor(nn.Module):
    def __init__(self, config):
        super(FeatureExtractor, self).__init__()
        self.num_classes = config.num_classes
        self.image_W = config.image_W
        self.image_H = config.image_H
        self.image_C = config.image_C

        self.input_shape = (self.image_C, self.image_H, self.image_W)
        if config.model_extract_name == 'vgg16':
            self.cnn = models.vgg16(pretrained=True)
            self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])
        elif config.model_extract_name == 'alexnet':
            self.cnn = models.alexnet(pretrained=True)
            self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])
        elif config.model_extract_name == 'resnet34':
            self.cnn = models.resnet34(pretrained=True)
            self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])
        else:
            print(f"chưa hỗ trợ model này: {config.model_extract_name}")

    def forward(self, x):
        features = self.cnn(x)
        features = features.view(features.size(0), -1)
        return features
    
    def output_size(self):
        with torch.no_grad():
            output = self.forward(torch.zeros(1, *self.input_shape))
        return output.size(1)
    

class ViT_FeatureExtractor(nn.Module):
    def __init__(self, config):
        super(ViT_FeatureExtractor, self).__init__()
        self.num_classes = config.num_classes
        self.image_W = config.image_W
        self.image_H = config.image_H
        self.image_C = config.image_C
        self.input_shape = (self.image_C, self.image_H, self.image_W)
        self.preprocessor = AutoFeatureExtractor.from_pretrained(config.model_extract_name)

    def forward(self, x):
        features = self.preprocessor(x)
        features = features.view(features.size(0), -1)
        return features
    
    def output_size(self):
        with torch.no_grad():
            output = self.forward(torch.zeros(1, *self.input_shape))
        return output.size(1)