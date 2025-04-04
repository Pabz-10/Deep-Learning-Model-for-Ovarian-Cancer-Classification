import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import datasets
import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
import seaborn as sns
import opendatasets as od
from sklearn.metrics import confusion_matrix
import kagglehub

import torch.multiprocessing


def main():
    np.random.seed(111)
    torch.manual_seed(111)
    torch.cuda.manual_seed_all(111)

    is_colab = False
    if is_colab:
        data_dir = './extracted-images/train/'
    else:
        data_dir = "./data/"
        os.makedirs(data_dir, exist_ok=True)

    download_images = False  # Set to True if you want to download the dataset
    if download_images:
        dataset_url = "https://www.kaggle.com/datasets/darshue/extracted-images"
        od.download(dataset_url, data_dir=data_dir)
        data_dir = os.path.join(data_dir, 'extracted-images/train/')
        print("Dataset downloaded to:", data_dir)
    else:
        data_dir = os.path.join(data_dir, 'extracted-images/train/')
        
    print("Path exists:", os.path.exists(data_dir))  # Should print True
    print("Contents:", os.listdir(data_dir))  # Should list the 5 subclasses

    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)
    
    if torch.cuda.is_available():
        print("CUDA available")
        device = torch.device("cuda")
    else:
        print("CUDA not available")
        device = torch.device("cpu")
    print("Device: ", device)

    train_test_split = 0.8
    batch_size = 32
    num_epochs = 1 # for testing purposes, set to 1, change to 50 for full training
    learning_rate = 0.001
    num_classes = 5
    if is_colab:
        num_workers = 4
    else:
        num_workers = 1

    mean=[0.8078, 0.6699, 0.8137]
    std=[0.0925, 0.1143, 0.0710]

    train_transform = transforms.Compose([
        # transforms.Grayscale(num_output_channels=1),
        transforms.Resize((496, 496)),
        transforms.RandomCrop(480),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
        # transforms.Grayscale(num_output_channels=1),
        transforms.Resize((496, 496)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir), transform=train_transform)
    test_dataset = datasets.ImageFolder(root=os.path.join(data_dir), transform=test_transform)

    train_size = int(train_test_split * len(train_dataset))
    test_size = len(train_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print('Number of training samples:', len(train_loader.dataset))
    print('Number of testing samples:', len(test_loader.dataset))

    classes = [ 'CC', 'EC', 'HGSC', 'LGSC', 'MC' ]

    from torchvision.models import resnet18
    model = resnet18(pretrained=True)
    model = model.to(device)

    models_dir = os.path.join(output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir, 'best_model_complete.pth')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.00025)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Store values for plotting
    epoch_losses = []
    val_accuracies = []
    test_accuracies = []

    best_val_accuracy = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Training loop with tqdm progress bar
        loop = tqdm.tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        epoch_losses.append(epoch_loss)

        # Validation Accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            val_loop = tqdm.tqdm(test_loader, desc="Validation")
            for images, labels in val_loop:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                predictions = outputs.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)

        # Test Accuracy (using the same test_loader here)
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                predictions = outputs.argmax(dim=1)
                correct_test += (predictions == labels).sum().item()
                total_test += labels.size(0)
        test_accuracy = correct_test / total_test
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

        # Save the best model based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved new best model at epoch {epoch+1}")

    # Plot Loss and Accuracy
    plt.figure()

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o', linestyle='-', color='b', label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), val_accuracies, marker='o', linestyle='-', color='g', label="Validation Accuracy")
    plt.plot(range(1, num_epochs + 1), test_accuracies, marker='s', linestyle='-', color='r', label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation & Test Accuracy Over Epochs")
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "training_validation_plot.png")
    plt.savefig(plot_path)
    print("Plot saved to:", plot_path)
    plt.show()

    # Confusion Matrix using the best saved model
    confuse_model = model_save_path
    
    use_dev_model = True # set to true to use the dev model that we trained, set to false to use the best model trained above
    
    if use_dev_model:
        model_save_path = os.path.join(models_dir, 'best_model_complete.pth')

    state_dict = torch.load(confuse_model, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    accuracy = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    print("Total Accuracy:", accuracy)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == '__main__':
    # comment out the next line if you can
    torch.multiprocessing.freeze_support()
    main()