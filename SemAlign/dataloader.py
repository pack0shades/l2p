import os
import numpy as np
import torch
from torchvision import datasets, transforms
from timm import create_model

# Import dataset classes
from continual_datasets.continual_datasets import (
    MNIST_RGB,
    FashionMNIST,
    NotMNIST,
    SVHN,
    Flowers102,
    StanfordCars,
    CUB200,
    TinyImagenet,
    Scene67,
    Imagenet_R,
)

# Function to create embeddings and store them
def create_embeddings(root_dir, embed_dir, model_name='vit_base_patch16_224', download=False):
    os.makedirs(embed_dir, exist_ok=True)

    # Load the model for embeddings
    model = create_model(model_name, pretrained=True, num_classes=0)
    model.eval()

    # Transformations for the images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust size as needed
        transforms.ToTensor(),
    ])

    # List of dataset classes to process
    dataset_classes = [
        lambda: MNIST_RGB(root_dir, train=True, transform=transform, download=download),
        lambda: FashionMNIST(root_dir, train=True, transform=transform, download=download),
        lambda: NotMNIST(root_dir, train=True, transform=transform, download=download),
        lambda: SVHN(root_dir, split='train', transform=transform, download=download),
        lambda: Flowers102(root_dir, transform=transform, download=download),
        lambda: StanfordCars(root_dir, transform=transform, download=download),
        lambda: CUB200(root_dir, train=True, transform=transform, download=download),
        lambda: TinyImagenet(root_dir, transform=transform, download=download),
        lambda: Scene67(root_dir, transform=transform, download=download),
        lambda: Imagenet_R(root_dir, train=True, transform=transform, download=download)
    ]

    for dataset_class in dataset_classes:
        dataset = dataset_class()  # Instantiate the dataset class

        # Iterate over the images and labels in the dataset
        for img, label in dataset:
            with torch.no_grad():
                # Generate the embedding
                embedding = model(img.unsqueeze(0))  # Add batch dimension
                embedding = embedding.squeeze(0)  # Remove batch dimension

            # Prepare to save the embedding
            image_id = f"{len(os.listdir(embed_dir))}"  # Simple image ID based on count
            np.save(os.path.join(embed_dir, f"{image_id}_embedding.npy"), embedding.cpu().numpy())

            with open(os.path.join(embed_dir, f"{image_id}_label.txt"), 'w') as f:
                # Check if label is a tensor
                if isinstance(label, torch.Tensor):
                    f.write(str(label.item()))  # Convert tensor to int
                else:
                    f.write(str(label))  # Save the label

    print("Embeddings and labels saved successfully.")

# Define your root directory for the datasets and embedding output
root_dir = './local_datasets'  # Update this path
embed_dir = 'SemAlign/local_data/image_embeddings'  # Update this path

# Call the function to create embeddings
create_embeddings(root_dir, embed_dir, download=True)


# the below written function is for loading image label so i can use this while training semalign
# to retrieve text embeddings to concate with image
'''def load_embedding_and_label(embed_dir, image_id):
    # Construct the file paths
    embedding_path = os.path.join(embed_dir, f"{image_id}_embedding.npy")
    label_path = os.path.join(embed_dir, f"{image_id}_label.txt")

    # Load the embedding
    embedding = np.load(embedding_path)  # Load the numpy array
    # Load the label
    with open(label_path, 'r') as f:
        label = int(f.read().strip())  # Convert the read string to int

    return embedding, label

# Example usage
embed_dir = '/path/to/your/embed_dir'  # Replace with your directory
image_id = '0'  # Example image ID; change accordingly
embedding, label = load_embedding_and_label(embed_dir, image_id)

print(f'Loaded embedding: {embedding}')
print(f'Loaded label: {label}')'''

