import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from transformers import ViTForImageClassification
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

# Define image transformations with additional augmentation layers
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.RandomRotation(10),  # Randomly rotate images by up to 10 degrees
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Random brightness and contrast
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Random cropping
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

# Load the dataset using ImageFolder
train_dataset = datasets.ImageFolder(root="./train", transform=transform)
test_dataset = datasets.ImageFolder(root="./test", transform=transform)

# Create DataLoaders with reduced batch size
batch_size = 32  # Reduced batch size to avoid OOM errors
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Print dataset information
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")
print(f"Classes: {train_dataset.classes}")

# Load pre-trained Vision Transformer (ViT) model
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",  # Pre-trained ViT model
    num_labels=7,  # FER2013 has 7 emotion classes
    ignore_mismatched_sizes=True  # Ignore size mismatch for classification head
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device is ",device)
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
non_trainable_params = total_params - trainable_params

total_size = total_params * 4 / (1024 ** 2)  # assuming 4 bytes per float32
trainable_size = trainable_params * 4 / (1024 ** 2)
non_trainable_size = non_trainable_params * 4 / (1024 ** 2)

print(f'Total Parameters: {total_params:,} ({total_size:.2f} MB)')
print(f'Trainable Parameters: {trainable_params:,} ({trainable_size:.2f} MB)')
print(f'Non-trainable Parameters: {non_trainable_params:,} ({non_trainable_size:.2f} MB)')
# Print model architecture
print(model)

# Define loss function (cross-entropy for classification)
criterion = nn.CrossEntropyLoss()

# Define optimizer with weight decay for regularization
optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)

# Define learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

# Define EarlyStopping callback
class EarlyStopping:
    def __init__(self, patience=3, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# Initialize EarlyStopping
early_stopping = EarlyStopping(patience=3, delta=0.01)

# Function to validate the model
def validate_model(model, test_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images).logits
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(test_loader)
    val_accuracy = 100 * correct / total
    return val_loss, val_accuracy

# Training loop
epochs = 15  # Number of epochs
# best_loss = float('inf')

def train():
    print("Starting training...")
    best_loss = float('inf') 
    for epoch in tqdm(range(epochs)):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        # print(f"Epoch [{epoch+1}/{epochs}]")
        for i, (images, labels) in tqdm(enumerate(train_loader)):

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images).logits
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # if (i + 1) % 50 == 0:
                # print(f"Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        train_accuracy = 100 * correct / total
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {epoch_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%", flush=True)
        # print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}", flush=True)

        # Validation step
        val_loss, val_accuracy = validate_model(model, test_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        # Step the scheduler
        scheduler.step(val_loss)

        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

        # Save the best model weights
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "best_model_weights.pth")
            print("Best model weights saved!")

# Evaluation loop
def evaluate():
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():  # Disable gradient calculation
        for images, labels in test_loader:  # Unpack the tuple (images, labels)
            # Move data to GPU
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images).logits  # Get the logits from the model
            preds = torch.argmax(outputs, dim=1)

            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=train_dataset.classes,
                yticklabels=train_dataset.classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix Heatmap")
    plt.tight_layout()
    plt.savefig("confusion_matrix_heatmap.png")  # Optional: Save to file
    plt.show()

# Save the fine-tuned model
def save_model():
    torch.save(model.state_dict(), "vit_fer2013.pth")
    print("Model saved to vit_fer2013.pth")

def main():
    train()
    model.load_state_dict(torch.load("best_model_weights.pth"))
    evaluate()
    save_model()

if __name__ == "__main__":
    main()
