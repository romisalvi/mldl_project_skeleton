import torch

def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    size = max(1, len(val_loader))  # Avoid division by zero
    correct, total = 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / size
    val_accuracy = 100. * correct / total if total > 0 else 0  # Prevent division by zero

    print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')
    return val_accuracy  # Corrected indentation
