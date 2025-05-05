import os
import torch
import pandas as pd
from torchvision import transforms, datasets
from transformers import ViTForImageClassification
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from PIL import Image  # Don't forget to import PIL

# Define the same image transformations as in the training code
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

# Define a custom dataset class for test images
class CustomTestDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.jpg') or fname.endswith('.png')]
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Open the image in RGB mode
        if self.transform:
            image = self.transform(image)
        return image, img_path  # Return the image and its path for saving predictions later

# Load the best model
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",  # Pre-trained ViT model
    num_labels=7,  # FER2013 has 7 emotion classes
    ignore_mismatched_sizes=True  # Ignore size mismatch for classification head
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.load_state_dict(torch.load("best_model_weights.pth"))  # Load the best saved model
model.eval()

# Load test images
test_dataset = datasets.ImageFolder(root="./newdata/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Test Dataset Size: {len(test_dataset)}")

# Prepare to store predictions and image paths

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():  # Disable gradient calculation
    for images, labels in tqdm(test_loader):  # Unpack the tuple (images, labels)
        # Move data to GPU
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images).logits  # Get the logits from the model
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)

        # Store predictions and labels
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        all_probs.extend(probs.cpu().numpy()) 

# Calculate accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Convert probabilities to DataFrame
probs_df = pd.DataFrame(all_probs, columns=[f"Class_{i}" for i in range(len(test_dataset.classes))])

# Optionally, add image paths (if you want to track which sample corresponds to which prediction)
image_paths = [img_path for _, img_path in test_loader.dataset.imgs]  # Extract image paths

# Add image paths to the DataFrame
probs_df['Image Path'] = image_paths[:len(probs_df)]  # Ensure the list is the same length as the predictions

# Save the probabilities to a CSV file
probs_df.to_csv('predicted_probabilities.csv', index=False)
print("Predicted probabilities saved to 'predicted_probabilities.csv'")

