import torch
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
import os

from timm import create_model  # Import from timm to create your registered model

# Define image transformations for preprocessing
def get_image_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Load and preprocess a single image
def load_and_preprocess_image(image_path, transform):
    image = Image.open(image_path)
    return transform(image).unsqueeze(0)  # Add batch dimension for model

# Generate embedding for a single image
def generate_image_embedding(model, image_tensor):
    model.eval()  # Ensure model is in evaluation mode
    with torch.no_grad():
        embedding = model(image_tensor).squeeze(0)  # Remove batch dimension
    return embedding

# Save all embeddings to a file
def save_embeddings(embeddings_dict, output_path):
    torch.save(embeddings_dict, output_path)
    print(f"Embeddings saved to {output_path}")

# Main function to iterate through images, generate embeddings, and save them
def create_image_embeddings(data_dir, output_file, model_name='vit_base_patch16_224', label_mapping=None):
    # Initialize the specified ViT model from models.py
    model = create_model(model_name, pretrained=True, num_classes=0)  # num_classes=0 for embeddings
    transform = get_image_transform()
    
    # Dictionary to store embeddings by unique IDs
    image_embeddings = {}

    # Iterate through all image files
    for image_file in Path(data_dir).glob('*.ubyte'):
        image_id = image_file.stem  # Unique ID for each image

        # Preprocess the image
        image_tensor = load_and_preprocess_image(image_file, transform)

        # Generate embedding
        embedding = generate_image_embedding(model, image_tensor)

        # Store embedding and label (if available) in dictionary
        image_embeddings[image_id] = {
            'embedding': embedding,
            'label': label_mapping.get(image_id, 'unknown') if label_mapping else 'unknown'
        }

    # Save embeddings to output file
    save_embeddings(image_embeddings, output_file)

# Example usage
if __name__ == "__main__":
    data_dir = '/l2p/local_datasets/FashionMNIST/raw'
    output_file = 'image_embeddings.pt'
    
    create_image_embeddings(data_dir, output_file, model_name='vit_base_patch16_224')
