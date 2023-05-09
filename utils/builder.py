import torch
import torch.nn as nn
from model.cnn_model import ResNet34_Model, LeNet5, NN
from model.kmeans_model import KMeans_Model
from model.svm_model import SVM_Model
from model.transfomer_model import ViT_Model

def build_model(config):
    if config.type_model == "SVM":
        return SVM_Model(config)
    if config.type_model == 'ResNet':
        return ResNet34_Model(config)
    if config.type_model == "LeNet5":
        return LeNet5(config)
    if config.type_model == "NN":
        return NN(config)
    if config.type_model == "Transformer":
        return ViT_Model(config)
    if config.type_model == "Kmeans":
        return KMeans_Model(config)
   

def build_loss_fn(config):
    if config.type_model == "SVM":
        return nn.MultiMarginLoss()
    if config.type_model == 'ResNet':
        return nn.CrossEntropyLoss()
    if config.type_model == "LeNet5":
        return nn.CrossEntropyLoss()
    if config.type_model == "NN":
        return nn.CrossEntropyLoss()
    if config.type_model == "Transformer":
        return nn.CrossEntropyLoss()