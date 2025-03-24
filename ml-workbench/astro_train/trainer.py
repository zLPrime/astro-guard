import torch.optim as optim
import torch.nn as nn
import gc
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