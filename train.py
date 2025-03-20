import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data.load import TinyImageNetDataLoader
from models.lab2 import CustomNet  # Ensure this points to your model

# Hyperparameters
num_epochs = 10
learning_rate = 0.001

# Initialize DataLoader
data_loader = TinyImageNetDataLoader(batch_size=32)
train_loader, val_loader = data_loader.get_loaders()

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomNet().to(device)  # Ensure you have a model defined in `models.lab2`
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Function
def train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()  # Reset gradients
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()  # Compute gradients
        optimizer.step()  # Update weights

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Update progress bar with loss and accuracy
        progress_bar.set_postfix(loss=loss.item(), acc=100. * correct / total)

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')

# Training Loop
for epoch in range(1, num_epochs + 1):
    train(epoch, model, train_loader, criterion, optimizer)

# Save model after training
torch.save(model.state_dict(), "model.pth")
print("Model saved as model.pth")
