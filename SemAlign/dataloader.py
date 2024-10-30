import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from timm import create_model
from PIL import Image
from pathlib import Path
import scipy.io as sio  # For loading .mat files
import gzip
import struct

class EmbeddingDataset(Dataset):
    def __init__(self, data_dir, embed_dir, model_name='vit_base_patch16_224', transform=None):
        self.data_dir = Path(data_dir)
        self.embed_dir = Path(embed_dir)
        self.transform = transform
        self.model = create_model(model_name, pretrained=True, num_classes=0)  # Load model for embeddings
        self.model.eval()  # Set model to evaluation mode
        self.embeddings = self.load_embeddings()  # Load or generate embeddings

    def load_embeddings(self):
        embeddings_dict = {}
        # Check if embeddings already exist
        if self.embed_dir.exists():
            for embed_file in self.embed_dir.glob('*.npy'):
                image_id = embed_file.stem
                embeddings_dict[image_id] = torch.from_numpy(np.load(embed_file))
            print(f"Loaded embeddings from {self.embed_dir}")
        else:
            # Create directory for embeddings if it doesn't exist
            self.embed_dir.mkdir(parents=True, exist_ok=True)

        # Iterate through datasets
        for dataset in self.data_dir.iterdir():
            if dataset.is_dir():
                for image_file in dataset.rglob('*'):
                    if image_file.is_file():
                        image_id = image_file.stem
                        label = self.get_label_from_path(image_file)  # Get label from directory structure
                        # Load images based on the dataset type
                        if 'MNIST' in dataset.name:
                            image_tensor = self.load_mnist_image(image_file)
                        elif 'CIFAR10' in dataset.name or 'cifar-100-python' in dataset.name:
                            if 'meta' in image_file.name:
                                continue  # Skip CIFAR-100 meta file
                        elif 'SVHN' in dataset.name:
                            image_tensor = self.load_svhn_image(image_file)
                        elif 'notMNIST' in dataset.name:
                            image_tensor = self.load_and_preprocess_notmnist_image(image_file)
                        elif 'FashionMNIST' in dataset.name:
                            image_tensor = self.load_binary_image(image_file)
                        elif image_file.suffix == '.mat':
                            image_tensor = self.load_mat_image(image_file)
                        else:
                            image_tensor = self.load_and_preprocess_image(image_file)
                            
                        with torch.no_grad():
                            embedding = self.model(image_tensor).squeeze(0)  # Generate embedding
                        # Save the embedding to disk with the label in the filename
                        np.save(self.embed_dir / f"{label}_{image_id}.npy", embedding.numpy())
                        embeddings_dict[f"{label}_{image_id}"] = embedding
        print(f"Generated and saved embeddings to {self.embed_dir}")
        return embeddings_dict

    def get_label_from_path(self, image_file):
        # Label from parent directory or higher-level folder (e.g., notMNIST/A/)
        return image_file.parent.name

    def load_mnist_image(self, file_path):
        with gzip.open(file_path, 'rb') as f:
            f.read(16)  # Skip header
            image_data = np.frombuffer(f.read(), np.uint8).reshape(-1, 28, 28)
        return self.process_image(image_data)

    def load_cifar_image(self, file_path):
        with open(file_path, 'rb') as f:
            batch = np.frombuffer(f.read(), np.uint8)
            images = batch[1:].reshape(-1, 3, 32, 32)
            return self.process_image(images)

    def load_svhn_image(self, file_path):
        data = sio.loadmat(file_path)
        images = data['X']  # shape (32, 32, 3, N)
        return self.process_image(images)

    def load_mat_image(self, file_path):
        data = sio.loadmat(file_path)
        images = data.get('X') or data.get('data')  # Adjust based on dataset structure
        if images is not None:
            return self.process_image(images)
        else:
            raise ValueError("No valid image data found in .mat file")

    def load_and_preprocess_notmnist_image(self, file_path):
        # Handle .png images in notMNIST directory structure
        image = Image.open(file_path).convert('RGB')
        return self.transform(image) if self.transform else torch.tensor(np.array(image))

    def load_binary_image(self, file_path):
        with open(file_path, 'rb') as f:
            f.read(16)  # Skip header for FashionMNIST or similar datasets
            image_data = np.frombuffer(f.read(), np.uint8).reshape(-1, 28, 28)
        return self.process_image(image_data)

    def load_and_preprocess_image(self, file_path):
        image = Image.open(file_path).convert('RGB')  # Handle general image files
        image_tensor = self.transform(image) if self.transform else torch.tensor(np.array(image))
        return image_tensor.unsqueeze(0)  # Add batch dimension

    def process_image(self, image_data):
        image_data = image_data.astype(np.float32) / 255.0  # Normalize to [0, 1]
        image_tensor = torch.tensor(image_data)  # Convert to tensor
        if self.transform:
            image_tensor = self.transform(image_tensor)  # Apply transformations
        return image_tensor.unsqueeze(0)  # Add batch dimension for model

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        image_id = list(self.embeddings.keys())[idx]
        embedding = self.embeddings[image_id]
        return {'image_id': image_id, 'embedding': embedding}

def build_embedding_dataloader(data_dir, embed_dir, batch_size=32, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = EmbeddingDataset(data_dir, embed_dir, transform=transform)
    
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

# Example usage
if __name__ == "__main__":
    data_dir = 'local_datasets'  # Update with your main data directory
    embed_dir = 'text_encoder/data/image_embeddings'  # Update with your embeddings directory
    batch_size = 32
    dataloader = build_embedding_dataloader(data_dir, embed_dir, batch_size=batch_size)
    
    for batch in dataloader:
        print(batch['image_id'])  # Print image IDs
        print(batch['embedding'].shape)  # Print shape of embeddings
