import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

# from approach.ResEmoteNet import ResEmoteNet
from ResEmoteNet import ResEmoteNet
from get_dataset import Four4All

# Set device to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Transform the dataset
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# import os
# print("Current Working Directory:", os.getcwd())
# print("Expected File Path:", os.path.abspath('../data/valid_labels.csv'))

# Load the dataset
train_dataset = Four4All(csv_file='../data/train_labels.csv',
                         img_dir='../data/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

val_dataset = Four4All(csv_file='../data/val_labels.csv', 
                       img_dir='../data/valid/', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

test_dataset = Four4All(csv_file='../data/test_labels.csv', 
                        img_dir='../data/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load the model
model = ResEmoteNet().to(device)

# Print the number of parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
non_trainable_params = total_params - trainable_params

total_size = total_params * 4 / (1024 ** 2)  # assuming 4 bytes per float32
trainable_size = trainable_params * 4 / (1024 ** 2)
non_trainable_size = non_trainable_params * 4 / (1024 ** 2)

print(f'Total Parameters: {total_params:,} ({total_size:.2f} MB)')
print(f'Trainable Parameters: {trainable_params:,} ({trainable_size:.2f} MB)')
print(f'Non-trainable Parameters: {non_trainable_params:,} ({non_trainable_size:.2f} MB)')

# Hyperparameters
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

# Early stopping parameters
patience = 15
best_val_acc = 0
patience_counter = 0
epoch_counter = 0
num_epochs = 80

# Store losses and accuracies
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# Start training
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Evaluate on validation set
    model.eval()
    val_running_loss = 0.0
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_running_loss / len(val_loader)
    val_acc = val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, "
          f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    epoch_counter += 1
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0 
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        print(f"No improvement in validation accuracy for {patience_counter} epochs.")

    if patience_counter > patience:
        print("Stopping early due to lack of improvement in validation accuracy.")
        break

# Compute final test accuracy after training
print("Evaluating on test set...")
model.eval()
test_running_loss = 0.0
test_correct, test_total = 0, 0

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_loss = test_running_loss / len(test_loader)
test_acc = test_correct / test_total
print(f"Final Test Loss: {test_loss:.4f}, Final Test Accuracy: {test_acc:.4f}")

cm = confusion_matrix(all_labels, all_preds)
class_names = class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix on Test Set')

# 4. Save heatmap
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()
print("Saved confusion matrix heatmap as confusion_matrix.png")

# Save results to CSV
df = pd.DataFrame({
    'Epoch': range(1, epoch_counter+1),
    'Train Loss': train_losses,
    'Validation Loss': val_losses,
    'Train Accuracy': train_accuracies,
    'Validation Accuracy': val_accuracies
})
df.to_csv('result_four4all.csv', index=False)
