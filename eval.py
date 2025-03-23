import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
from data.load import TinyImageNetDataLoader
from models.lab2 import CustomNet  # Ensure this points to your model

# Initialize wandb
wandb.init(project="tinyimagenet-classification", name="evaluation")

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomNet().to(device)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Define loss function
criterion = nn.CrossEntropyLoss()

# Initialize DataLoader
data_loader = TinyImageNetDataLoader(batch_size=32)  # Adjust batch size if needed
_, val_loader = data_loader.get_loaders()

def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct, total = 0, 0

    progress_bar = tqdm(val_loader, desc="Validation", leave=False)
    
    with torch.no_grad():
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            progress_bar.set_postfix(loss=loss.item(), acc=100. * correct / total)

            # Log batch metrics
            wandb.log({
                "batch_val_loss": loss.item(),
                "batch_val_accuracy": 100. * correct / total
            })
    
    val_loss /= len(val_loader)
    val_accuracy = 100. * correct / total
    
    print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')

    # Log epoch-level metrics
    wandb.log({
        "val_loss": val_loss,
        "val_accuracy": val_accuracy
    })
    
    return val_loss, val_accuracy

# Run validation
validate(model, val_loader, criterion)
