import torch
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

def evaluate(model, dataloader, device='cuda'):
    model.eval()
    model.to(device)
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            del images
    
    return all_labels, all_preds

def print_evaluate(model, data, label_encoder, device='cuda'):
    labels, preds = evaluate(model, data, device)
    decoded_labels = label_encoder.inverse_transform(labels)
    decoded_preds = label_encoder.inverse_transform(preds)
    
    print("\nAccuracy:", accuracy_score(decoded_labels, decoded_preds))
    print("Classification Report:\n", classification_report(decoded_labels, decoded_preds))