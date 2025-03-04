import math
import torch
import torch.utils.data
from pathlib import Path
from torchvision import datasets, transforms


from src.helpers import compute_mean_and_std, get_data_location
import matplotlib.pyplot as plt


def get_data_loaders(
    batch_size: int = 32, valid_size: float = 0.2, num_workers: int = 1, limit: int = -1):
    """
    Create and returns the train_one_epoch, validation and test data loaders.

    :param batch_size: size of the mini-batches
    :param valid_size: fraction of the dataset to use for validation. For example 0.2
                       means that 20% of the dataset will be used for validation
    :param num_workers: number of workers to use in the data loaders. Use num_workers=1. 
    :param limit: maximum number of data points to consider
    :return a dictionary with 3 keys: 'train_one_epoch', 'valid' and 'test' containing respectively the
            train_one_epoch, validation and test data loaders
    """
    
    data_loaders = {"train": None, "valid": None, "test": None}

    base_path = Path(get_data_location())

    # Compute mean and std of the dataset for data normalization
    mean, std = compute_mean_and_std()
    print(f"Dataset mean: {mean}, std: {std}")

    # Data Augmentation: Landmark images can be from different angles and so data augmentation with artificial transforms is needed
    # Besides, the training dataset are only 5000 images which is small to train to recognize 50 landmarks hence data augmentation is needed
    # Transforms include Flip (no vertical flip which is an unlikely image), rotation, translation, color jitter (to match different lighting and camera/user customizations)
    # Transforms include resizing and crop at the end instead of start else images end up with huge and unuseful black borders
    # Transforms include normalization with the mean and std calculated from the dataset
    # Training, validation and testing datasets have different transformations as below
    data_transforms = {
        "train": transforms.Compose(
            [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.Resize(256),
            transforms.CenterCrop(224),  # Crop center to 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ]
        ),
        "valid": transforms.Compose(
           [ 
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ]
        ),
        "test": transforms.Compose(
            [ 
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ]
        ),
    }

    # Train and Validation datasets
    train_data = datasets.ImageFolder(
        base_path / "train",
        data_transforms['train']
    )
    # The validation dataset is a split from the train_one_epoch dataset, so we read
    # from the same folder, but we apply the transforms for validation
    valid_data = datasets.ImageFolder(
        base_path / "train",
        data_transforms['valid']
    )

    # obtain training indices that will be used for validation
    n_tot = len(train_data)
    indices = torch.randperm(n_tot)

    # limit the number of data points to consider as requested
    if limit > 0:
        indices = indices[:limit]
        n_tot = limit

    split = int(math.ceil(valid_size * n_tot))
    train_idx, valid_idx = indices[split:], indices[:split]

    # samplers for obtaining training and validation batches
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler  = torch.utils.data.SubsetRandomSampler(valid_idx)

    # data loaders
    data_loaders["train"] = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
    )
    data_loaders["valid"] = torch.utils.data.DataLoader(
        valid_data,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
    )

    # test data loader
    test_data = datasets.ImageFolder(
        base_path / "test",
        data_transforms['test']
    )


    data_loaders["test"] = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return data_loaders


def visualize_one_batch(data_loaders, max_n: int = 5):
    """
    Visualize one batch of data.

    :param data_loaders: dictionary containing data loaders
    :param max_n: maximum number of images to show
    :return: None
    """


    # batch of training images with an iterator from the train dataloader
    dataiter  = iter(data_loaders['train'])
    images, labels  = next(dataiter) 

    mean, std = compute_mean_and_std()
    invTrans = transforms.Compose(
        [
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=1 / std),
            transforms.Normalize(mean=-mean, std=[1.0, 1.0, 1.0]),
        ]
    )

    images = invTrans(images)

    # class names from the train data loader
    class_names  = data_loaders['train'].dataset.classes


    images = torch.permute(images, (0, 2, 3, 1)).clip(0, 1)

    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    for idx in range(max_n):
        ax = fig.add_subplot(1, max_n, idx + 1, xticks=[], yticks=[])
        ax.imshow(images[idx])

        ax.set_title(class_names[labels[idx].item()])
