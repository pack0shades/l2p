import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from timm import create_model
from tqdm import tqdm
from semalign import SemAlign

# Import dataset classes
from continual_datasets.continual_datasets import (
    MNIST_RGB, FashionMNIST, NotMNIST, SVHN,
    Flowers102, StanfordCars, CUB200, TinyImagenet,
    Scene67, Imagenet_R
)

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define the model and optimizer
def initialize_model(v_size, s_size, learning_rate):
    model = SemAlign(v_size, s_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer

# Save the model checkpoint
def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

# Generate embedding for an image and retrieve corresponding text embedding
def generate_embeddings(img, model, text_embeddings, label):
    # Generate the image embedding
    with torch.no_grad():
        img_embedding = model(img.unsqueeze(0)).squeeze(0).to(device)  # Add and remove batch dim

    # Retrieve text embedding
    text_embedding = torch.tensor(text_embeddings.get(str(label), np.zeros(768))).to(device)  # Handle missing embeddings

    return img_embedding, text_embedding

# Training loop
def train_semalign_model(model, pretrained_model, data_loaders, optimizer, text_embeddings, num_epochs, save_best=True):
    criterion = nn.MSELoss()  # Using MSE loss for embeddings similarity
    best_loss = float('inf')

    for epoch in range(num_epochs):
        total_loss = 0.0
        model.train()

        for data_loader in data_loaders:
            for img, label in tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                img = img.to(device)
                img_embedding, text_embedding = generate_embeddings(img, pretrained_model, text_embeddings, label.item())

                optimizer.zero_grad()
                outputs = model(img_embedding, text_embedding)
                loss = criterion(outputs, text_embedding)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        avg_loss = total_loss / sum(len(dl) for dl in data_loaders)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, avg_loss)

        # Save the best model
        if save_best and avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model updated and saved.")

if __name__ == "__main__":
    # Parameters
    v_size = 640  # Image embedding size from ViT
    s_size = 768  # Text embedding size
    num_epochs = 50
    learning_rate = 0.001

    # Load text embeddings from JSON file
    with open('/home/b23es1024/l2p/text_encoder/data/text_embeddings.json', 'r') as f:
        text_embeddings = json.load(f)

    # Initialize model and optimizer
    model, optimizer = initialize_model(v_size, s_size, learning_rate)

    # Load the pretrained ViT model for generating image embeddings
    pretrained_model = create_model('vit_base_patch16_224', pretrained=True, num_classes=0).to(device)
    pretrained_model.eval()

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Define dataset and data loaders
    root_dir = './local_datasets'
    dataset_classes = [
        lambda: MNIST_RGB(root_dir, train=True, transform=transform, download=True),
        lambda: FashionMNIST(root_dir, train=True, transform=transform, download=True),
        lambda: NotMNIST(root_dir, train=True, transform=transform, download=True),
        lambda: SVHN(root_dir, split='train', transform=transform, download=True),
        lambda: Flowers102(root_dir, transform=transform, download=True),
        lambda: StanfordCars(root_dir, transform=transform, download=True),
        lambda: CUB200(root_dir, train=True, transform=transform, download=True),
        lambda: TinyImagenet(root_dir, transform=transform, download=True),
        lambda: Scene67(root_dir, transform=transform, download=True),
        lambda: Imagenet_R(root_dir, train=True, transform=transform, download=True)
    ]

    # Create DataLoaders for each dataset
    data_loaders = [DataLoader(dataset_class(), batch_size=32, shuffle=True) for dataset_class in dataset_classes]

    # Train the model
    train_semalign_model(model, pretrained_model, data_loaders, optimizer, text_embeddings, num_epochs)


# import torch
# import torchvision.transforms as transforms
# from pathlib import Path
# from PIL import Image
# import os

# from timm import create_model  # Import from timm to create your registered model

# # Define image transformations for preprocessing
# def get_image_transform():
#     return transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

# # Load and preprocess a single image
# def load_and_preprocess_image(image_path, transform):
#     image = Image.open(image_path)
#     return transform(image).unsqueeze(0)  # Add batch dimension for model

# # Generate embedding for a single image
# def generate_image_embedding(model, image_tensor):
#     model.eval()  # Ensure model is in evaluation mode
#     with torch.no_grad():
#         embedding = model(image_tensor).squeeze(0)  # Remove batch dimension
#     return embedding

# # Save all embeddings to a file
# def save_embeddings(embeddings_dict, output_path):
#     torch.save(embeddings_dict, output_path)
#     print(f"Embeddings saved to {output_path}")

# # Main function to iterate through images, generate embeddings, and save them
# def create_image_embeddings(data_dir, output_file, model_name='vit_base_patch16_224', label_mapping=None):
#     # Initialize the specified ViT model from models.py
#     model = create_model(model_name, pretrained=True, num_classes=0)  # num_classes=0 for embeddings
#     transform = get_image_transform()
    
#     # Dictionary to store embeddings by unique IDs
#     image_embeddings = {}

#     # Iterate through all image files
#     for image_file in Path(data_dir).glob('*.ubyte'):
#         image_id = image_file.stem  # Unique ID for each image

#         # Preprocess the image
#         image_tensor = load_and_preprocess_image(image_file, transform)

#         # Generate embedding
#         embedding = generate_image_embedding(model, image_tensor)

#         # Store embedding and label (if available) in dictionary
#         image_embeddings[image_id] = {
#             'embedding': embedding,
#             'label': label_mapping.get(image_id, 'unknown') if label_mapping else 'unknown'
#         }

#     # Save embeddings to output file
#     save_embeddings(image_embeddings, output_file)

# # Example usage
# if __name__ == "__main__":
#     data_dir = '/l2p/local_datasets/FashionMNIST/raw'
#     output_file = 'image_embeddings.pt'
    
#     create_image_embeddings(data_dir, output_file, model_name='vit_base_patch16_224')
