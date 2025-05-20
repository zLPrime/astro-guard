import torch.optim as optim
import torch.nn as nn
import torch
import gc
import csv
import os
from torch import cuda
from tqdm import tqdm

def astro_train(model, dataloader, epochs, device='cuda'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(tqdm(dataloader)):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            del images, labels, outputs, loss
            if (i + 1) % 5 == 0:
                cuda.empty_cache()
                gc.collect()
        
        cuda.empty_cache()
        gc.collect()
        
        print(f"\nEpoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader)}")
    del criterion, optimizer

def train_and_monitor(
    model,
    train_loader,
    epochs=10,
    device=torch.device('cpu'),
    optimizer=None,
    criterion=None,
    log_interval=10,
    log_file='training_log.csv',
    val_loader=None,
    best_model_path=None,
    patience=5
):
    model.to(device)
    optimizer = optimizer or optim.Adam(model.parameters(), lr=0.001)
    criterion = criterion or nn.CrossEntropyLoss()

    best_val_acc = 0.0

    # Prepare logging
    fieldnames = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc']
    with open(log_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    epochs_without_improvement = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}')

        for batch_idx, (inputs, targets) in loop:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if (batch_idx + 1) % log_interval == 0:
                loop.set_postfix({
                    'Train Loss': f'{running_loss / (batch_idx + 1):.4f}',
                    'Train Acc': f'{100. * correct / total:.2f}%'
                })

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total

        val_loss = val_acc = None
        if val_loader:
            model.eval()
            val_loss_total = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for val_inputs, val_targets in val_loader:
                    val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                    val_outputs = model(val_inputs)
                    loss = criterion(val_outputs, val_targets)
                    val_loss_total += loss.item()
                    _, val_pred = val_outputs.max(1)
                    val_total += val_targets.size(0)
                    val_correct += val_pred.eq(val_targets).sum().item()

            val_loss = val_loss_total / len(val_loader)
            val_acc = 100. * val_correct / val_total
            print(f'Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc:.2f}%')

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
                epochs_without_improvement = 0
                print(f'Best model saved with accuracy: {val_acc:.2f}%')
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        # Save metrics to log
        with open(log_file, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss if val_loss is not None else '',
                'val_acc': val_acc if val_acc is not None else '',
            })
