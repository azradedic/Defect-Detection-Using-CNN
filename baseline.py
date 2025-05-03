import os
import sys

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
from utils.helper import train, evaluate, predict_localize, plot_dataset_comparison
from utils.constants import NEG_CLASS

# Add safe globals for model loading
torch.serialization.add_safe_globals([
    torch.nn.modules.activation.ReLU,
    torch.nn.modules.activation.LeakyReLU,
    torch.nn.modules.activation.ELU,
    torch.nn.modules.activation.SELU,
    torch.nn.modules.container.Sequential,
    torch.nn.modules.conv.Conv2d,
    torch.nn.modules.pooling.MaxPool2d,
    torch.nn.modules.linear.Linear,
    torch.nn.modules.dropout.Dropout,
    torch.nn.modules.batchnorm.BatchNorm2d,
    CustomVGG
])

def plot_parameter_analysis(results_df):
    """Create plots analyzing the impact of different parameters"""
    # 1. Learning Rate Impact
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    for opt in results_df['optimizer'].unique():
        subset = results_df[results_df['optimizer'] == opt]
        plt.plot(subset['learning_rate'], subset['accuracy'], 'o-', label=opt)
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.title('Impact of Learning Rate')
    plt.legend()
    plt.xscale('log')
    
    plt.subplot(1, 2, 2)
    for opt in results_df['optimizer'].unique():
        subset = results_df[results_df['optimizer'] == opt]
        plt.plot(subset['learning_rate'], subset['loss'], 'o-', label=opt)
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Impact of Learning Rate')
    plt.legend()
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig('results/plots/parameter_analysis/learning_rate_impact.png')
    plt.close()

    # 2. Optimizer Comparison
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.boxplot(x='optimizer', y='accuracy', data=results_df)
    plt.title('Accuracy by Optimizer')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x='optimizer', y='loss', data=results_df)
    plt.title('Loss by Optimizer')
    plt.tight_layout()
    plt.savefig('results/plots/parameter_analysis/optimizer_comparison.png')
    plt.close()

    # 3. Activation Function Impact
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.boxplot(x='activation', y='accuracy', data=results_df)
    plt.title('Accuracy by Activation Function')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x='activation', y='loss', data=results_df)
    plt.title('Loss by Activation Function')
    plt.tight_layout()
    plt.savefig('results/plots/parameter_analysis/activation_impact.png')
    plt.close()

    # 4. Number of Neurons Impact
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.boxplot(x='num_neurons', y='accuracy', data=results_df)
    plt.title('Accuracy by Number of Neurons')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x='num_neurons', y='loss', data=results_df)
    plt.title('Loss by Number of Neurons')
    plt.tight_layout()
    plt.savefig('results/plots/parameter_analysis/neurons_impact.png')
    plt.close()

    # 5. Batch Size Impact
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.boxplot(x='batch_size', y='accuracy', data=results_df)
    plt.title('Accuracy by Batch Size')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x='batch_size', y='loss', data=results_df)
    plt.title('Loss by Batch Size')
    plt.tight_layout()
    plt.savefig('results/plots/parameter_analysis/batch_size_impact.png')
    plt.close()

    # 6. Epochs Impact
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.boxplot(x='epochs', y='accuracy', data=results_df)
    plt.title('Accuracy by Number of Epochs')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x='epochs', y='loss', data=results_df)
    plt.title('Loss by Number of Epochs')
    plt.tight_layout()
    plt.savefig('results/plots/parameter_analysis/epochs_impact.png')
    plt.close()

    # Save detailed analysis results
    analysis_results = {
        'best_accuracy': results_df['accuracy'].max(),
        'best_loss': results_df['loss'].min(),
        'best_params': results_df.loc[results_df['accuracy'].idxmax()].to_dict(),
        'worst_params': results_df.loc[results_df['accuracy'].idxmin()].to_dict(),
        'mean_accuracy': results_df['accuracy'].mean(),
        'std_accuracy': results_df['accuracy'].std(),
        'mean_loss': results_df['loss'].mean(),
        'std_loss': results_df['loss'].std()
    }
    
    with open('results/metrics/analysis_results.txt', 'w') as f:
        for key, value in analysis_results.items():
            f.write(f"{key}: {value}\n")

## Parameters

batch_size = 16  # Reduced from 32 for better stability
target_train_accuracy = 0.99  # Target for "golden model" with error < 0.01
test_size = 0.2
learning_rate = 0.0001  # Reduced from 0.001 for better stability
epochs = 15  # Balanced number of epochs
class_weight = [1, 2] if NEG_CLASS == 1 else [2, 1]  # Reduced class weight imbalance
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classification_threshold = 16.00  # As specified in requirements
heatmap_thres = 0.7

# Data subsetting parameters - more balanced approach
max_samples_per_class = 200  # Increased from 100 to 200
train_subset_ratio = 0.7  # Increased from 0.5 to 0.7

## Load Training Data

data_folder = "data/"
train_subset_name = ["capsule", "hazelnut", "leather"]  # Keep all training datasets
test_subset_name = ["bottle", "pill"]  # Keep 2 test datasets for better validation
train_roots = [os.path.join(data_folder, subset) for subset in train_subset_name]
test_roots = [os.path.join(data_folder, subset) for subset in test_subset_name]

# Create results directory structure
os.makedirs("results/metrics", exist_ok=True)
os.makedirs("results/plots/training_curves", exist_ok=True)
os.makedirs("results/plots/confusion_matrices", exist_ok=True)
os.makedirs("results/plots/parameter_analysis", exist_ok=True)
os.makedirs("weights", exist_ok=True)

# Check for existing weights
existing_weights = [f for f in os.listdir("weights") if f.endswith(".pt")]
if existing_weights:
    print("\nFound existing model weights:")
    for i, weight in enumerate(existing_weights):
        print(f"{i+1}. {weight}")
    
    while True:
        choice = input("\nDo you want to:\n1. Use existing weights\n2. Train new models\nEnter choice (1 or 2): ")
        if choice in ['1', '2']:
            break
        print("Invalid choice. Please enter 1 or 2.")
    
    if choice == '1':
        # Load existing weights
        while True:
            try:
                weight_idx = int(input(f"\nEnter the number of the weights to load (1-{len(existing_weights)}): ")) - 1
                if 0 <= weight_idx < len(existing_weights):
                    break
                print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a number.")
        
        weight_path = os.path.join("weights", existing_weights[weight_idx])
        print(f"\nLoading weights from {weight_path}")
        checkpoint = torch.load(weight_path, map_location=device)
        
        # Extract parameters from checkpoint
        activation = checkpoint.get('activation', nn.ReLU)
        num_neurons = checkpoint.get('num_neurons', 64)
        learning_rate = checkpoint.get('learning_rate', 0.001)
        optimizer_name = checkpoint.get('optimizer', 'Adam')
        batch_size = checkpoint.get('batch_size', 32)
        
        print("\nModel parameters from checkpoint:")
        print(f"Activation: {activation.__name__}")
        print(f"Number of neurons: {num_neurons}")
        print(f"Learning rate: {learning_rate}")
        print(f"Optimizer: {optimizer_name}")
        print(f"Batch size: {batch_size}")
        
        # Create model with loaded parameters
        model = CustomVGG(activation=activation, num_neurons=num_neurons).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load data for evaluation
        print("\nLoading test data...")
        test_loader = get_test_loaders(
            roots=test_roots,
            batch_size=batch_size,
            test_size=0.9,
            random_state=42
        )
        
        # Evaluate loaded model
        print("\nEvaluating loaded model...")
        accuracy, loss, conf_matrix = evaluate(model, test_loader, device)
        print(f"\nModel Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Loss: {loss:.4f}")
        print("\nConfusion Matrix:")
        print(conf_matrix)
        
        # Visualization
        print("\nGenerating localization visualization...")
        predict_localize(
            model, test_loader, device, thres=heatmap_thres, n_samples=15, show_heatmap=False
        )
        
        sys.exit(0)

# Parameter grids for experiments - balanced combinations
activations = [nn.ReLU, nn.LeakyReLU]  # Reduced to two activation functions
num_neurons_list = [64, 128]  # Two different neuron counts
learning_rates = [0.0001, 0.0005]  # Reduced learning rates
optimizers = [optim.Adam]  # Only use Adam for better stability
batch_sizes = [16, 32]  # Reduced batch sizes

# Create a DataFrame to store all results
results_df = pd.DataFrame(columns=[
    'activation', 'num_neurons', 'learning_rate', 'optimizer', 'batch_size',
    'accuracy', 'loss', 'balanced_accuracy', 'precision', 'recall', 'f1_score'
])

try:
    # Load data once
    print("\nLoading training data...")
    train_loader = get_train_loaders(
        roots=train_roots,
        batch_size=batch_size,
        random_state=42,
        max_samples_per_class=max_samples_per_class,
        subset_ratio=train_subset_ratio
    )

    print("\nLoading validation data...")
    val_loader = get_test_loaders(
        roots=train_roots,
        batch_size=batch_size,
        test_size=0.2,
        random_state=42,
        max_samples_per_class=max_samples_per_class,
        subset_ratio=train_subset_ratio
    )

    print("\nLoading test data...")
    test_loader = get_test_loaders(
        roots=test_roots,
        batch_size=batch_size,
        test_size=0.9,
        random_state=42,
        max_samples_per_class=max_samples_per_class
    )

    # Print dataset information
    print("\nDataset Information:")
    print(f"Training datasets: {train_subset_name}")
    print(f"Test datasets: {test_subset_name}")
    print(f"Classification threshold: {classification_threshold}")
    print(f"Max samples per class: {max_samples_per_class}")
    print(f"Training subset ratio: {train_subset_ratio}")

    # Plot dataset comparison with different colors
    print("\nGenerating dataset comparison plot...")
    plot_dataset_comparison(train_loader, test_loader, device)
    plt.show()  # Show the plot

    total_combinations = len(list(itertools.product(
        activations, num_neurons_list, learning_rates, optimizers, batch_sizes
    )))
    current_combination = 0

    print(f"\nStarting training with {total_combinations} combinations")
    print("Estimated time: 45-60 minutes")  # Updated time estimate
    print("Progress will be shown for each combination")

    for activation, num_neurons, lr, optimizer_class, batch_size in itertools.product(
        activations, num_neurons_list, learning_rates, optimizers, batch_sizes
    ):
        current_combination += 1
        print(f"\nTraining combination {current_combination}/{total_combinations}")
        print(f"Activation: {activation.__name__}")
        print(f"Number of neurons: {num_neurons}")
        print(f"Learning rate: {lr}")
        print(f"Optimizer: {optimizer_class.__name__}")
        print(f"Batch size: {batch_size}")

        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Model with current parameters
        model = CustomVGG(activation=activation, num_neurons=num_neurons).to(device)
        
        # Print model summary
        summary(model, input_size=(batch_size, 3, 224, 224))

        class_weight_tensor = torch.tensor(class_weight).type(torch.FloatTensor).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
        optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=1e-4)  # Added weight decay
        
        # Initialize scheduler with more patience and higher minimum learning rate
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5,  # Increased patience
            min_lr=1e-5,  # Higher minimum learning rate
            verbose=True  # Added verbose output
        )

        # Train
        print("\nStarting training...")
        trained_model, history = train(train_loader, val_loader, model, optimizer, criterion, epochs, device, target_train_accuracy, scheduler)
        
        # Save model and history
        experiment_name = f"{activation.__name__}_{num_neurons}_{lr}_{optimizer_class.__name__}_{batch_size}"
        print(f"\nSaving model as {experiment_name}...")
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'activation': activation,
            'num_neurons': num_neurons,
            'learning_rate': lr,
            'optimizer': optimizer_class.__name__,
            'batch_size': batch_size,
            'history': history
        }, f"weights/model_{experiment_name}.pt")

        # Evaluate
        print("\nEvaluating model...")
        accuracy, loss, conf_matrix = evaluate(trained_model, test_loader, device)
        
        # Calculate additional metrics
        print("\nCalculating additional metrics...")
        y_true = []
        y_pred = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = trained_model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                # Apply classification threshold
                preds = (outputs[:, 1] > classification_threshold).long()
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        
        from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        balanced_acc = balanced_accuracy_score(y_true, y_pred)

        # Add results to DataFrame
        new_row = {
            'activation': activation.__name__,
            'num_neurons': num_neurons,
            'learning_rate': lr,
            'optimizer': optimizer_class.__name__,
            'batch_size': batch_size,
            'accuracy': accuracy,
            'loss': loss,
            'balanced_accuracy': balanced_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

        # Save results after each experiment
        results_df.to_csv('results/metrics/experiment_results.csv', index=False)
        
        # Plot training curves
        print("\nGenerating training curves...")
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title(f'Loss Curves - {experiment_name}')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Acc')
        plt.plot(history['val_acc'], label='Val Acc')
        plt.title(f'Accuracy Curves - {experiment_name}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"results/plots/training_curves/training_curves_{experiment_name}.png")
        plt.close()

        print(f"\nCompleted training and evaluation for {experiment_name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Loss: {loss:.4f}")
        print(f"Progress: {current_combination}/{total_combinations} combinations completed")

        print(np.bincount(y_true))
        print(np.bincount(y_pred))

        print("Sample outputs:", outputs[:, 1][:10].cpu().numpy())

    # After all experiments, create parameter analysis plots
    print("\nGenerating parameter analysis plots...")
    plot_parameter_analysis(results_df)

    # Find and load best model
    best_model_idx = results_df['accuracy'].idxmax()
    best_params = results_df.iloc[best_model_idx]
    print("\nBest Model Parameters:")
    print(best_params)

    # Load best model for final evaluation
    best_model_path = f"weights/model_{best_params['activation']}_{best_params['num_neurons']}_{best_params['learning_rate']}_{best_params['optimizer']}_{best_params['batch_size']}.pt"
    best_model = CustomVGG(
        activation=getattr(nn, best_params['activation']),
        num_neurons=best_params['num_neurons']
    ).to(device)
    best_model.load_state_dict(torch.load(best_model_path)['model_state_dict'])

    # Final evaluation with visualization
    print("\nPerforming final evaluation...")
    accuracy, loss, conf_matrix = evaluate(best_model, test_loader, device)
    print(f"\nFinal Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Loss: {loss:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # Visualization
    print("\nGenerating localization visualization...")
    predict_localize(
        best_model, test_loader, device, thres=heatmap_thres, n_samples=15, show_heatmap=False
    )

    print("\nTraining and evaluation completed successfully!")

except Exception as e:
    print(f"\nAn error occurred: {str(e)}")
    import traceback
    traceback.print_exc()
