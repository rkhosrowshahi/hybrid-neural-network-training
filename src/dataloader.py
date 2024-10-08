import os
import pickle
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset


def get_balanced_subset(dataset, num_samples, num_classes):
    samples_per_class = num_samples // num_classes
    if num_samples % num_classes > 0:
        samples_per_class += 1
    # Create an empty list to store the balanced dataset
    balanced_indices = []
    # Randomly select samples from each class for the training dataset
    for i in range(num_classes):
        class_indices = np.where(np.array(dataset.targets) == i)[0]
        selected_indices = np.random.choice(
            class_indices, samples_per_class, replace=False
        )
        balanced_indices.append(selected_indices)
    return np.asarray(balanced_indices).astype(int)


def get_val_test_dataloader(dataset, batch_size):
    val_set, test_set, num_classes = None, None, None
    if dataset == "cifar10":
        num_classes = 10
        val_num_samples = 1000
        test_num_samples = 9000
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_test
        )
        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )
        if os.path.exists("./splits/cifar-10_data_split.npz"):
            c = np.load("./splits/cifar-10_data_split.npz")
            val_indices = c["val"]
            test_indices = c["test"]
            train_indices = c["train"]

        else:
            balanced_indices = get_balanced_subset(
                trainset, val_num_samples, num_classes
            )
            train_indices = balanced_indices.flatten()
            np.random.shuffle(train_indices)

            balanced_indices = get_balanced_subset(
                testset, val_num_samples + test_num_samples, num_classes
            )
            val_indices = balanced_indices[
                :, : (val_num_samples // num_classes)
            ].flatten()
            np.random.shuffle(val_indices)
            test_indices = balanced_indices[
                :, (val_num_samples // num_classes) :
            ].flatten()
            # test_indices = balanced_indices.flatten()
            np.random.shuffle(test_indices)
            np.savez(
                "./splits/cifar-10_data_split.npz",
                val=val_indices,
                test=test_indices,
                train=train_indices,
            )

        val_set = Subset(testset, val_indices)
        test_set = Subset(testset, test_indices)
        train_sel = Subset(trainset, train_indices)
        val_set = ConcatDataset([val_set, train_sel])
        train_set = trainset

    elif dataset == "cifar100":
        num_classes = 100
        val_num_samples = 1000
        test_num_samples = 9000
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5071, 0.4865, 0.4409],
                    std=[0.2673, 0.2564, 0.2762],
                ),
            ]
        )
        trainset = torchvision.datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform_test
        )
        testset = torchvision.datasets.CIFAR100(
            root="./data", train=False, download=True, transform=transform_test
        )
        if os.path.exists("./splits/cifar-100_data_split.npz"):
            c = np.load("./splits/cifar-100_data_split.npz")
            val_indices = c["val"]
            test_indices = c["test"]
            train_indices = c["train"]

        else:
            balanced_indices = get_balanced_subset(
                trainset, val_num_samples, num_classes
            )
            train_indices = balanced_indices.flatten()
            np.random.shuffle(train_indices)

            balanced_indices = get_balanced_subset(
                testset, val_num_samples + test_num_samples, num_classes
            )
            val_indices = balanced_indices[
                :, : (val_num_samples // num_classes)
            ].flatten()
            np.random.shuffle(val_indices)
            test_indices = balanced_indices[
                :, (val_num_samples // num_classes) :
            ].flatten()
            # test_indices = balanced_indices.flatten()
            np.random.shuffle(test_indices)
            np.savez(
                "./splits/cifar-100_data_split.npz",
                val=val_indices,
                test=test_indices,
                train=train_indices,
            )

        val_set = Subset(testset, val_indices)
        test_set = Subset(testset, test_indices)
        train_sel = Subset(trainset, train_indices)
        val_set = ConcatDataset([val_set, train_sel])
        train_set = trainset

    elif dataset == "imagenet":
        num_classes = 1000
        val_num_samples = 10000
        test_num_samples = 40000
        transform_test = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        testset = torchvision.datasets.ImageNet(
            root="./data/imagenet-1000",
            split="val",
            transform=transform_test,
        )
        print(len(testset))
        balanced_indices = get_balanced_subset(
            testset, val_num_samples + test_num_samples, num_classes
        )

        val_set = Subset(
            testset, balanced_indices[:, : (val_num_samples // num_classes)].flatten()
        )
        test_set = Subset(
            testset, balanced_indices[:, (val_num_samples // num_classes) :].flatten()
        )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    return train_loader, val_loader, test_loader, num_classes
