import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
from tqdm import tqdm
from time import sleep
import os

from .constants import NEG_CLASS


def train(train_loader, val_loader, model, optimizer, criterion, epochs, device, target_train_accuracy, scheduler=None):
    """
    Train the model with early stopping and progress tracking
    """
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    patience = 10  # Increased patience for early stopping
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        # Training phase with gradient clipping
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        # Update learning rate if scheduler is provided
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # Early stopping check with minimum epochs
        if epoch >= 5:  # Don't stop before 5 epochs
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
        
        # Store metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Early stopping
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
        
        # Target accuracy reached
        if train_acc >= target_train_accuracy:
            print(f'Target accuracy {target_train_accuracy} reached after {epoch+1} epochs')
            break
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history


def evaluate(model, dataloader, device, threshold=16.00):
    """
    Script to evaluate a model after training.
    Uses threshold-based classification (threshold=16.00).
    Outputs accuracy and balanced accuracy, draws confusion matrix.
    Returns accuracy, loss, and confusion matrix values.
    """
    model.to(device)
    model.eval()
    class_names = ["Good", "Anomaly"] if NEG_CLASS == 1 else ["Anomaly", "Good"]

    running_corrects = 0
    running_loss = 0.0
    y_true = np.empty(shape=(0,))
    y_pred = np.empty(shape=(0,))
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Get predictions
            outputs = model(inputs)
            # During evaluation, model returns (probs, location), we only need probs
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            # Apply threshold-based classification using raw logits
            preds_class = (outputs[:, 1] > threshold).long()  # Class 1 if logit > threshold

            labels = labels.to("cpu").numpy()
            preds_class = preds_class.detach().to("cpu").numpy()

            y_true = np.concatenate((y_true, labels))
            y_pred = np.concatenate((y_pred, preds_class))

    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    avg_loss = running_loss / len(dataloader)
    conf_matrix = confusion_matrix(y_true, y_pred)

    print("Accuracy: {:.4f}".format(accuracy))
    print("Balanced Accuracy: {:.4f}".format(balanced_accuracy))
    print("Threshold: {:.2f}".format(threshold))
    print()
    plot_confusion_matrix(y_true, y_pred, class_names=class_names)
    
    return accuracy, avg_loss, conf_matrix


def plot_confusion_matrix(y_true, y_pred, class_names="auto"):
    confusion = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=[5, 5])
    sns.heatmap(
        confusion,
        annot=True,
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.ylabel("True labels")
    plt.xlabel("Predicted labels")
    plt.title("Confusion Matrix")
    
    # Save confusion matrix
    os.makedirs("results/plots/confusion_matrices", exist_ok=True)
    plt.savefig("results/plots/confusion_matrices/confusion_matrix.png")
    plt.show()
    
    
def get_bbox_from_heatmap(heatmap, thres=0.8):
    """
    Returns bounding box around the defected area:
    Upper left and lower right corner.
    
    Threshold affects size of the bounding box.
    The higher the threshold, the wider the bounding box.
    """
    binary_map = heatmap > thres

    x_dim = np.max(binary_map, axis=0) * np.arange(0, binary_map.shape[1])
    x_0 = int(x_dim[x_dim > 0].min())
    x_1 = int(x_dim.max())

    y_dim = np.max(binary_map, axis=1) * np.arange(0, binary_map.shape[0])
    y_0 = int(y_dim[y_dim > 0].min())
    y_1 = int(y_dim.max())

    return x_0, y_0, x_1, y_1


def predict_localize(
    model, dataloader, device, thres=0.8, n_samples=9, show_heatmap=False
):
    """
    Runs predictions for the samples in the dataloader.
    Shows image, its true label, predicted label and probability.
    If an anomaly is predicted, draws bbox around defected region and heatmap.
    """
    model.to(device)
    model.eval()

    class_names = ["Good", "Anomaly"] if NEG_CLASS == 1 else ["Anomaly", "Good"]
    transform_to_PIL = transforms.ToPILImage()

    n_cols = 3
    n_rows = int(np.ceil(n_samples / n_cols))
    plt.figure(figsize=[n_cols * 5, n_rows * 5])

    counter = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        out = model(inputs)
        probs, class_preds = torch.max(out[0], dim=-1)
        feature_maps = out[1].to("cpu")

        for img_i in range(inputs.size(0)):
            img = transform_to_PIL(inputs[img_i])
            class_pred = class_preds[img_i]
            prob = probs[img_i]
            label = labels[img_i]
            heatmap = feature_maps[img_i][NEG_CLASS].detach().numpy()

            counter += 1
            plt.subplot(n_rows, n_cols, counter)
            plt.imshow(img)
            plt.axis("off")
            plt.title(
                "Predicted: {}, Prob: {:.3f}, True Label: {}".format(
                    class_names[class_pred], prob, class_names[label]
                )
            )

            if class_pred == NEG_CLASS:
                x_0, y_0, x_1, y_1 = get_bbox_from_heatmap(heatmap, thres)
                rectangle = Rectangle(
                    (x_0, y_0),
                    x_1 - x_0,
                    y_1 - y_0,
                    edgecolor="red",
                    facecolor="none",
                    lw=3,
                )
                plt.gca().add_patch(rectangle)
                if show_heatmap:
                    plt.imshow(heatmap, cmap="Reds", alpha=0.3)

            if counter == n_samples:
                plt.tight_layout()
                # Save localization results
                os.makedirs("results/plots/localization", exist_ok=True)
                plt.savefig("results/plots/localization/localization_results.png")
                plt.show()
                return


def plot_dataset_comparison(train_loader, test_loader, device):
    """
    Plot original dataset (training) and verification data (test) for comparison.
    """
    # Get sample images from both loaders
    train_images, train_labels = next(iter(train_loader))
    test_images, test_labels = next(iter(test_loader))
    
    # Create figure
    plt.figure(figsize=(15, 5))
    
    # Plot training data
    plt.subplot(1, 2, 1)
    for i in range(min(5, len(train_images))):
        img = train_images[i].cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0,1]
        plt.imshow(img)
        plt.title(f'Training Sample {i+1}\nClass: {"Anomaly" if train_labels[i] == 1 else "Good"}')
        plt.axis('off')
        if i < 4:  # Don't show more than 5 images
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 2, 1)
    
    # Plot test data
    plt.subplot(1, 2, 2)
    for i in range(min(5, len(test_images))):
        img = test_images[i].cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0,1]
        plt.imshow(img)
        plt.title(f'Test Sample {i+1}\nClass: {"Anomaly" if test_labels[i] == 1 else "Good"}')
        plt.axis('off')
        if i < 4:  # Don't show more than 5 images
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 2, 2)
    
    plt.tight_layout()
    plt.savefig('results/plots/dataset_comparison.png')
    plt.close()