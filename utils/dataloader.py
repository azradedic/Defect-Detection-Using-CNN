import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, ConcatDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split, StratifiedKFold

from .constants import (
    GOOD_CLASS_FOLDER,
    DATASET_SETS,
    INPUT_IMG_SIZE,
    IMG_FORMAT,
    NEG_CLASS,
)


class MVTEC_AD_DATASET(Dataset): # inherit from Dataset class
    """
    Class to load subsets of MVTEC ANOMALY DETECTION DATASET
    Dataset Link: https://www.mvtec.com/company/research/datasets/mvtec-ad
    
    Root is path to the subset, for instance, `mvtec_anomaly_detection/leather`
    """

    def __init__(self, root):
        self.classes = ["Good", "Anomaly"] if NEG_CLASS == 1 else ["Anomaly", "Good"]
        self.img_transform = transforms.Compose(
            [transforms.Resize(INPUT_IMG_SIZE), transforms.ToTensor()]
        )

        (
            self.img_filenames,
            self.img_labels,
            self.img_labels_detailed,
        ) = self._get_images_and_labels(root)

    def _get_images_and_labels(self, root):
        image_names = []
        labels = []
        labels_detailed = []

        # DATASET_SETS includes train and test
        # All good-labeled data are in train folder
        # All no-good-labeled data are in test folder
        for folder in DATASET_SETS: 
            folder = os.path.join(root, folder) # folder = (root + 'train') or (root + 'test')

            for class_folder in os.listdir(folder):
                # label = 0: good, label = 1: anomaly
                label = (
                    1 - NEG_CLASS if class_folder == GOOD_CLASS_FOLDER else NEG_CLASS
                )

                # folder names are either 'good' or defect types
                # specify object type (root) for different types (good, defected) of images
                label_detailed = root + '_' + class_folder 

                class_folder = os.path.join(folder, class_folder)
                class_images = os.listdir(class_folder) # all image filenames in a list
                class_images = [
                    os.path.join(class_folder, image)
                    for image in class_images
                    if image.find(IMG_FORMAT) > -1 # file_path(type: str).find, check if file is of IMG_FORMAT
                ]

                image_names.extend(class_images)
                labels.extend([label] * len(class_images))
                labels_detailed.extend([label_detailed] * len(class_images))

        return image_names, labels, labels_detailed

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_fn = self.img_filenames[idx]
        label = self.img_labels[idx]
        img = Image.open(img_fn)
        img = self.img_transform(img) # reshape image into INPUT_IMG_SIZE
        label = torch.as_tensor(label, dtype=torch.long)
        return img, label


def get_train_loaders(roots, batch_size, random_state=42, max_samples_per_class=None, subset_ratio=1.0):
    """
    Returns train dataloaders from given roots (types of objects).
    
    Args:
        roots: List of dataset root paths
        batch_size: Batch size for the dataloader
        random_state: Random seed for reproducibility
        max_samples_per_class: Maximum number of samples per class (None for no limit)
        subset_ratio: Ratio of data to use (1.0 for all data)
    """
    lst_datasets = []
    total_length = 0

    for root in roots:
        dataset = MVTEC_AD_DATASET(root=root)
        
        # Apply max_samples_per_class if specified
        if max_samples_per_class is not None:
            # Get indices for each class
            good_indices = [i for i, label in enumerate(dataset.img_labels) if label == 0]
            anomaly_indices = [i for i, label in enumerate(dataset.img_labels) if label == 1]
            
            # Randomly select samples
            np.random.seed(random_state)
            if len(good_indices) > max_samples_per_class:
                good_indices = np.random.choice(good_indices, max_samples_per_class, replace=False)
            if len(anomaly_indices) > max_samples_per_class:
                anomaly_indices = np.random.choice(anomaly_indices, max_samples_per_class, replace=False)
            
            # Combine indices
            selected_indices = np.concatenate([good_indices, anomaly_indices])
            np.random.shuffle(selected_indices)
            
            # Create subset
            dataset = torch.utils.data.Subset(dataset, selected_indices)
        
        lst_datasets.append(dataset)
        total_length += len(dataset)
        print(f"Dataset loaded (training) {root}: N Images = {len(dataset)}, Percentage of defected images = {np.sum([dataset[i][1] for i in range(len(dataset))]) / len(dataset):.3f}")

    all_dataset = ConcatDataset(lst_datasets)
    
    # Apply subset_ratio if specified
    if subset_ratio < 1.0:
        indices = np.arange(total_length)
        np.random.seed(random_state)
        np.random.shuffle(indices)
        subset_size = int(total_length * subset_ratio)
        indices = indices[:subset_size]
        all_dataset = torch.utils.data.Subset(all_dataset, indices)
        total_length = len(all_dataset)

    train_idx = np.arange(total_length)
    train_sampler = SubsetRandomSampler(train_idx)
    train_loader = DataLoader(
        all_dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True
    )

    return train_loader


def get_test_loaders(roots, batch_size, test_size=0.2, random_state=42, max_samples_per_class=None, subset_ratio=1.0):
    """
    Returns test dataloaders from given roots (types of objects).
    Split dataset in stratified manner, considering various objects and defect types.
    
    Args:
        roots: List of dataset root paths
        batch_size: Batch size for the dataloader
        test_size: Ratio of data to use for testing
        random_state: Random seed for reproducibility
        max_samples_per_class: Maximum number of samples per class (None for no limit)
        subset_ratio: Ratio of data to use (1.0 for all data)
    """
    lst_datasets = []
    total_length = 0
    stratifier = []

    for root in roots:
        dataset = MVTEC_AD_DATASET(root=root)
        
        # Apply max_samples_per_class if specified
        if max_samples_per_class is not None:
            # Get indices for each class
            good_indices = [i for i, label in enumerate(dataset.img_labels) if label == 0]
            anomaly_indices = [i for i, label in enumerate(dataset.img_labels) if label == 1]
            
            # Randomly select samples
            np.random.seed(random_state)
            if len(good_indices) > max_samples_per_class:
                good_indices = np.random.choice(good_indices, max_samples_per_class, replace=False)
            if len(anomaly_indices) > max_samples_per_class:
                anomaly_indices = np.random.choice(anomaly_indices, max_samples_per_class, replace=False)
            
            # Combine indices
            selected_indices = np.concatenate([good_indices, anomaly_indices])
            np.random.shuffle(selected_indices)
            
            # Create subset
            dataset = torch.utils.data.Subset(dataset, selected_indices)
        
        lst_datasets.append(dataset)
        total_length += len(dataset)
        stratifier.extend([dataset[i][1] for i in range(len(dataset))])
        print(f"Dataset loaded (testing) {root}: N Images = {len(dataset)}, Percentage of defected images = {np.sum([dataset[i][1] for i in range(len(dataset))]) / len(dataset):.3f}")

    all_dataset = ConcatDataset(lst_datasets)
    
    # Apply subset_ratio if specified
    if subset_ratio < 1.0:
        indices = np.arange(total_length)
        np.random.seed(random_state)
        np.random.shuffle(indices)
        subset_size = int(total_length * subset_ratio)
        indices = indices[:subset_size]
        all_dataset = torch.utils.data.Subset(all_dataset, indices)
        total_length = len(all_dataset)
        stratifier = [stratifier[i] for i in indices]

    train_idx, test_idx = train_test_split(
        np.arange(total_length),
        test_size=test_size,
        shuffle=True,
        stratify=stratifier,
        random_state=random_state,
    )

    test_sampler = SubsetRandomSampler(test_idx)
    test_loader = DataLoader(all_dataset, batch_size=batch_size, sampler=test_sampler, drop_last=False)

    return test_loader
