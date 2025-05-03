import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG16_Weights
from .constants import INPUT_IMG_SIZE 
import matplotlib.pyplot as plt
from torchinfo import summary
from sklearn.metrics import confusion_matrix
import seaborn as sns


class CustomVGG(nn.Module):
    """
    Custom multi-class classification model 
    with VGG16 feature extractor, pretrained on ImageNet
    and custom classification head.
    Parameters for the first convolutional blocks are freezed.
    
    Returns class scores (logits) for both train and eval mode.
    """

    def __init__(self, n_classes=2, activation=nn.ReLU, num_neurons=128):
        super(CustomVGG, self).__init__()
        self.feature_extractor = models.vgg16(weights=VGG16_Weights.DEFAULT).features[:-1]
        self.classification_head = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(
                kernel_size=(INPUT_IMG_SIZE[0] // 2 ** 5, INPUT_IMG_SIZE[1] // 2 ** 5)
            ),
            nn.Flatten(),
            nn.Linear(512, num_neurons),
            activation(),
            nn.Linear(num_neurons, n_classes),
        )
        self._freeze_params()

    def _freeze_params(self):
        for param in self.feature_extractor[:23].parameters():
            param.requires_grad = False

    def forward(self, x):
        feature_maps = self.feature_extractor(x)
        scores = self.classification_head(feature_maps)
        return scores

    @staticmethod
    def plot_training_curves(train_loss_history, val_loss_history, train_acc_history, val_acc_history):
        """Plot training and validation curves"""
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_loss_history, label='Train Loss')
        plt.plot(val_loss_history, label='Val Loss')
        plt.title('Loss Curves')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(train_acc_history, label='Train Acc')
        plt.plot(val_acc_history, label='Val Acc')
        plt.title('Accuracy Curves')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_confusion_matrix(true_labels, predictions):
        """Plot confusion matrix"""
        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title('Confusion Matrix')
        plt.show()

    def show_summary(self, batch_size=32):
        """Show model summary"""
        summary(self, input_size=(batch_size, 3, INPUT_IMG_SIZE[0], INPUT_IMG_SIZE[1]))