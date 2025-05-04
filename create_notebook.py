import nbformat as nbf

# Create a new notebook
nb = nbf.v4.new_notebook()

# Define the cells
cells = [
    nbf.v4.new_markdown_cell("# Anomaly Detection Training\n\nThis notebook implements the training and evaluation of the anomaly detection model using the CustomVGG architecture, matching the latest codebase."),
    
    nbf.v4.new_code_cell("""import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score
import gc

from utils.dataloader import get_train_loaders, get_test_loaders
from utils.model import CustomVGG
from utils.helper import train, evaluate, plot_dataset_comparison
from utils.constants import NEG_CLASS, INPUT_IMG_SIZE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
"""),
    
    nbf.v4.new_markdown_cell("## Set Parameters"),
    
    nbf.v4.new_code_cell("""BATCH_SIZE = 32
LEARNING_RATE = 0.0005
NUM_EPOCHS = 20
CLASS_WEIGHTS = torch.tensor([1.0, 3.0])
CLASSIFICATION_THRESHOLD = 16.00
"""),
    
    nbf.v4.new_markdown_cell("## Data Loading"),
    
    nbf.v4.new_code_cell("""train_loader, val_loader = get_train_loaders(BATCH_SIZE)
test_loader = get_test_loaders(BATCH_SIZE)
print(f'Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}, Test batches: {len(test_loader)}')
"""),
    
    nbf.v4.new_markdown_cell("## Model Initialization"),
    
    nbf.v4.new_code_cell("""activation = nn.ReLU
num_neurons = 64
model = CustomVGG(activation=activation, num_neurons=num_neurons).to(device)
criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(device), label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
"""),
    
    nbf.v4.new_markdown_cell("## Training"),
    
    nbf.v4.new_code_cell("""model, history = train(
    train_loader, val_loader, model, optimizer, criterion,
    NUM_EPOCHS, device, target_train_accuracy=0.90, scheduler=scheduler
)
"""),
    
    nbf.v4.new_markdown_cell("## Plot Training Curves"),
    
    nbf.v4.new_code_cell("""plt.figure(figsize=(12, 4))
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
plt.show()
"""),
    
    nbf.v4.new_markdown_cell("## Evaluation"),
    
    nbf.v4.new_code_cell("""accuracy, loss, conf_matrix = evaluate(model, test_loader, device)
print(f'Accuracy: {accuracy:.4f}, Loss: {loss:.4f}')
"""),
    
    nbf.v4.new_markdown_cell("## Additional Metrics"),
    
    nbf.v4.new_code_cell("""y_true = []
y_pred = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        preds = (outputs[:, 1] > CLASSIFICATION_THRESHOLD).long()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
balanced_acc = balanced_accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
print(f'Balanced Accuracy: {balanced_acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
"""),
    
    nbf.v4.new_markdown_cell("## Save Model"),
    
    nbf.v4.new_code_cell("""os.makedirs('weights', exist_ok=True)
torch.save(model.state_dict(), 'weights/trained_model.pt')
print('Model saved to weights/trained_model.pt')
"""),
    
    nbf.v4.new_markdown_cell("## Results & Metrics"),
    
    nbf.v4.new_code_cell("""results_df = pd.read_csv('results/metrics/experiment_results.csv')
print('Columns in results:', results_df.columns.tolist())
import os
import matplotlib.pyplot as plt
import seaborn as sns

def safe_boxplot(x, y, title, filename):
    if x in results_df.columns and y in results_df.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=x, y=y, data=results_df)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join('results/plots/parameter_analysis/', filename))
        plt.close()
        print(f"Saved: {filename}")

# Learning Rate Impact
if 'learning_rate' in results_df.columns and 'accuracy' in results_df.columns and 'optimizer' in results_df.columns:
    plt.figure(figsize=(10, 5))
    sns.lineplot(x='learning_rate', y='accuracy', hue='optimizer', data=results_df, marker='o')
    plt.title('Accuracy vs Learning Rate')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig('results/plots/parameter_analysis/learning_rate_impact.png')
    plt.close()
    print("Saved: learning_rate_impact.png")

safe_boxplot('optimizer', 'accuracy', 'Accuracy by Optimizer', 'optimizer_comparison.png')
safe_boxplot('activation', 'accuracy', 'Accuracy by Activation Function', 'activation_impact.png')
safe_boxplot('num_neurons', 'accuracy', 'Accuracy by Number of Neurons', 'neurons_impact.png')
safe_boxplot('batch_size', 'accuracy', 'Accuracy by Batch Size', 'batch_size_impact.png')

if 'balanced_accuracy' in results_df.columns:
    safe_boxplot('optimizer', 'balanced_accuracy', 'Balanced Accuracy by Optimizer', 'optimizer_balanced_accuracy.png')
    safe_boxplot('activation', 'balanced_accuracy', 'Balanced Accuracy by Activation Function', 'activation_balanced_accuracy.png')
    safe_boxplot('num_neurons', 'balanced_accuracy', 'Balanced Accuracy by Number of Neurons', 'neurons_balanced_accuracy.png')
    safe_boxplot('batch_size', 'balanced_accuracy', 'Balanced Accuracy by Batch Size', 'batch_size_balanced_accuracy.png')

print("Parameter analysis plots generated.")
""")
]

# Add the cells to the notebook
nb['cells'] = cells

# Write the notebook to a file
with open('Training_new.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f) 