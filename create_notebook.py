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
print(f'Using device: {device}')"""),
    
    nbf.v4.new_markdown_cell("## Set Parameters"),
    
    nbf.v4.new_code_cell("""# Model parameters
activation = nn.ReLU  # or nn.LeakyReLU
num_neurons = 64  # or 128

# Training parameters
BATCH_SIZE = 32  # or 16
LEARNING_RATE = 0.0005  # or 0.0001
NUM_EPOCHS = 20
CLASS_WEIGHTS = torch.tensor([1.0, 3.0])
CLASSIFICATION_THRESHOLD = 16.00

# Optimizer parameters
WEIGHT_DECAY = 1e-4
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5
MIN_LR = 1e-5"""),
    
    nbf.v4.new_markdown_cell("## Data Loading"),
    
    nbf.v4.new_code_cell("""train_loader, val_loader = get_train_loaders(BATCH_SIZE)
test_loader = get_test_loaders(BATCH_SIZE)
print(f'Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}, Test batches: {len(test_loader)}')

# Plot dataset comparison
plot_dataset_comparison(train_loader, test_loader, device)"""),
    
    nbf.v4.new_markdown_cell("## Model Initialization"),
    
    nbf.v4.new_code_cell("""model = CustomVGG(activation=activation, num_neurons=num_neurons).to(device)
criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(device), label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=SCHEDULER_FACTOR, 
    patience=SCHEDULER_PATIENCE, 
    min_lr=MIN_LR,
    verbose=True
)

# Show model summary
model.show_summary(batch_size=BATCH_SIZE)"""),
    
    nbf.v4.new_markdown_cell("## Training"),
    
    nbf.v4.new_code_cell("""model, history = train(
    train_loader, val_loader, model, optimizer, criterion,
    NUM_EPOCHS, device, target_train_accuracy=0.90, scheduler=scheduler
)"""),
    
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
plt.show()"""),
    
    nbf.v4.new_markdown_cell("## Evaluation"),
    
    nbf.v4.new_code_cell("""accuracy, loss, conf_matrix = evaluate(model, test_loader, device, threshold=CLASSIFICATION_THRESHOLD)
print(f'Accuracy: {accuracy:.4f}, Loss: {loss:.4f}')"""),
    
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

print(f'Balanced Accuracy: {balanced_acc:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Print class distribution
print("\\nClass distribution in predictions:")
print(np.bincount(y_pred))
print("\\nClass distribution in true labels:")
print(np.bincount(y_true))"""),
    
    nbf.v4.new_markdown_cell("## Save Model"),
    
    nbf.v4.new_code_cell("""# Create experiment name
experiment_name = f"{activation.__name__}_{num_neurons}_{LEARNING_RATE}_{optimizer.__class__.__name__}_{BATCH_SIZE}"

# Save model and training history
os.makedirs('weights', exist_ok=True)
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'activation': activation,
    'num_neurons': num_neurons,
    'learning_rate': LEARNING_RATE,
    'optimizer': optimizer.__class__.__name__,
    'batch_size': BATCH_SIZE,
    'history': history
}, f"weights/model_{experiment_name}.pt")

print(f'Model saved to weights/model_{experiment_name}.pt')

# Save metrics to CSV
os.makedirs('results/metrics', exist_ok=True)
metrics_df = pd.DataFrame([{
    'activation': activation.__name__,
    'num_neurons': num_neurons,
    'learning_rate': LEARNING_RATE,
    'optimizer': optimizer.__class__.__name__,
    'batch_size': BATCH_SIZE,
    'accuracy': accuracy,
    'loss': loss,
    'balanced_accuracy': balanced_acc,
    'precision': precision,
    'recall': recall,
    'f1_score': f1
}])

# Append to existing results or create new file
if os.path.exists('results/metrics/experiment_results.csv'):
    existing_df = pd.read_csv('results/metrics/experiment_results.csv')
    metrics_df = pd.concat([existing_df, metrics_df], ignore_index=True)
metrics_df.to_csv('results/metrics/experiment_results.csv', index=False)
print('Metrics saved to results/metrics/experiment_results.csv')""")
]

# Add the cells to the notebook
nb['cells'] = cells

# Write the notebook to a file
with open('Training.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f) 