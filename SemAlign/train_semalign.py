import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from SemAlign.semalignmodel import SemAlign

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir="checkpoints"):
    """Saves a checkpoint of the model and optimizer state."""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")


def train_semalign_model(model, train_loader, optimizer, num_epochs, device, save_best=True):
    model.train()
    criterion = nn.L2Loss()  # Use L2 loss
    best_loss = float('inf')  # Initialize with infinity to track the best model

    for epoch in range(num_epochs):
        total_loss = 0.0

        for semantic, contexts, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            semantic = semantic.to(device)
            contexts = contexts.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(semantic, contexts)

            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Save checkpoint after each epoch
        save_checkpoint(model, optimizer, epoch, avg_loss)

        # Save the best model if applicable
        if save_best and avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model updated and saved.")

if __name__ == "__main__":
    # Parameters
    v_size = 640  # Size of video embeddings
    s_size = 768  # Size of semantic embeddings
    num_epochs = 50
    learning_rate = 0.001

    model = SemAlign(v_size, s_size).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    train_semalign_model(model, train_loader, optimizer, num_epochs, device)