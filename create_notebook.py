import nbformat as nbf

# Create a new notebook
nb = nbf.v4.new_notebook()

# Define the cells
cells = [
    nbf.v4.new_markdown_cell("# Anomaly Detection Training\n\nThis notebook implements the training and evaluation of the anomaly detection model using the CustomVGG architecture."),
    
    nbf.v4.new_code_cell("""import os
import sys

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.abspath("__file__")))

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
import itertools
import pandas as pd
import seaborn as sns
import gc

from utils.dataloader import get_train_loaders, get_test_loaders
from utils.model import CustomVGG
from utils.helper import train, evaluate, predict_localize
from utils.constants import NEG_CLASS

# Add safe globals for model loading
torch.serialization.add_safe_globals([torch.nn.modules.activation.ReLU])"""),
    
    nbf.v4.new_markdown_cell("## Set Parameters"),
    
    nbf.v4.new_code_cell("""# Training parameters
batch_size = 40
target_train_accuracy = 0.98  # for early stopping
test_size = 0.2  # for validation split
learning_rate = 0.0001
epochs = 10
class_weight = [1, 3] if NEG_CLASS == 1 else [3, 1]  # Good = 1, Anomaly = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

heatmap_thres = 0.7

print(f"Using device: {device}")"""),
    
    nbf.v4.new_markdown_cell("## Load Data"),
    
    nbf.v4.new_code_cell("""data_folder = "data/"
train_subset_name = ["capsule", "hazelnut", "leather"]
test_subset_name = ["bottle", "pill", "wood"]
train_roots = [os.path.join(data_folder, subset) for subset in train_subset_name]
test_roots = [os.path.join(data_folder, subset) for subset in test_subset_name]

train_loader = get_train_loaders(
    roots=train_roots,
    batch_size=batch_size,
    random_state=42
)"""),
    
    nbf.v4.new_markdown_cell("## Model Setup\n\nChoose whether to load an existing model or train a new one."),
    
    nbf.v4.new_code_cell("""# List available models
weight_files = [f for f in os.listdir("weights") if f.endswith(".pt")]
if weight_files:
    print("Available models:")
    for i, file in enumerate(weight_files):
        print(f"{i+1}. {file}")
else:
    print("No saved models found in weights directory")"""),
    
    nbf.v4.new_markdown_cell("### Option 1: Load Existing Model"),
    
    nbf.v4.new_code_cell("""# Uncomment and modify to load a specific model
# model_idx = 0  # Change this to load a different model
# model_path = os.path.join("weights", weight_files[model_idx])
# try:
#     # First try loading with weights_only=True
#     checkpoint = torch.load(model_path, map_location=device, weights_only=True)
# except Exception as e:
#     print(f"Loading with weights_only=True failed: {e}")
#     print("Trying to load with weights_only=False...")
#     checkpoint = torch.load(model_path, map_location=device, weights_only=False)
# 
# model = CustomVGG(activation=nn.ReLU, num_neurons=64).to(device)
# model.load_state_dict(checkpoint['model_state_dict'])
# print(f"Loaded model from {weight_files[model_idx]}")"""),
    
    nbf.v4.new_markdown_cell("### Option 2: Train New Model"),
    
    nbf.v4.new_code_cell("""# Experiment parameters
activations = [nn.ReLU, nn.Tanh]
num_neurons_list = [64, 128]
learning_rates = [0.001, 0.0001]
optimizers = [optim.Adam, optim.SGD]
batch_sizes = [32, 64]
epochs_list = [10, 20]

# Choose parameters for this run
current_params = {
    'activation': nn.ReLU,
    'num_neurons': 64,
    'lr': 0.0001,
    'optimizer_class': optim.Adam,
    'batch_size': 40,
    'epochs': 10
}

# Create and display model
model = CustomVGG(
    activation=current_params['activation'],
    num_neurons=current_params['num_neurons']
).to(device)

summary(model, input_size=(current_params['batch_size'], 3, 224, 224))"""),
    
    nbf.v4.new_code_cell("""# Setup training
class_weight_tensor = torch.tensor(class_weight).type(torch.FloatTensor).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
optimizer = current_params['optimizer_class'](model.parameters(), lr=current_params['lr'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

# Create validation loader
val_loader = get_test_loaders(
    roots=train_roots,
    batch_size=current_params['batch_size'],
    test_size=0.2,
    random_state=42
)

# Train model
trained_model, history = train(
    train_loader,
    val_loader,
    model,
    optimizer,
    criterion,
    current_params['epochs'],
    device,
    target_train_accuracy,
    scheduler
)"""),
    
    nbf.v4.new_markdown_cell("## Save Model"),
    
    nbf.v4.new_code_cell("""# Create experiment name and save model
experiment_name = f"{current_params['activation'].__name__}_{current_params['num_neurons']}_lr{current_params['lr']}_{current_params['optimizer_class'].__name__}_bs{current_params['batch_size']}_ep{current_params['epochs']}"

torch.save({
    'model_state_dict': trained_model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'activation': current_params['activation'],
    'num_neurons': current_params['num_neurons'],
    'learning_rate': current_params['lr'],
    'optimizer': current_params['optimizer_class'].__name__,
    'batch_size': current_params['batch_size'],
    'epochs': current_params['epochs'],
    'history': history
}, f"weights/model_{experiment_name}.pt")

print(f"Model saved as: model_{experiment_name}.pt")"""),
    
    nbf.v4.new_markdown_cell("## Evaluate Model"),
    
    nbf.v4.new_code_cell("""# Load test data
test_loader = get_test_loaders(
    roots=test_roots,
    batch_size=current_params['batch_size'],
    test_size=0.9,
    random_state=42
)

# Evaluate model
accuracy, loss = evaluate(trained_model, test_loader, device)
print(f"Final Test Accuracy: {accuracy:.4f}")
print(f"Final Test Loss: {loss:.4f}")"""),
    
    nbf.v4.new_markdown_cell("## Visualize Results"),
    
    nbf.v4.new_code_cell("""# Plot training curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Loss Curves')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['val_acc'], label='Val Acc')
plt.title('Accuracy Curves')
plt.legend()

plt.tight_layout()
plt.show()"""),
    
    nbf.v4.new_code_cell("""# Visualize predictions and localization
predict_localize(
    trained_model,
    test_loader,
    device,
    thres=heatmap_thres,
    n_samples=9,
    show_heatmap=True
)""")
]

# Add the cells to the notebook
nb['cells'] = cells

# Write the notebook to a file
with open('Training_new.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f) 