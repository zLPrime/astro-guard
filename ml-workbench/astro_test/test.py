import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score
from data.dataloader import get_dataloader  # Reuse the data loader
from models.model import SimpleCNN  # Import your model

def evaluate(model, dataloader, device='cuda'):
    model.eval()
    model.to(device)
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return all_labels, all_preds

def main():
    # Load config
    import yaml
    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    
    # Load test data
    test_loader = get_dataloader(config['data_dir'], config['batch_size'], shuffle=False)
    
    # Load the trained model
    model = SimpleCNN()
    model.load_state_dict(torch.load('model.pth'))  # Adjust filename as needed
    
    # Evaluate
    true_labels, predictions = evaluate(model, test_loader, config['device'])
    
    # Metrics
    print("Accuracy:", accuracy_score(true_labels, predictions))
    print("Classification Report:\n", classification_report(true_labels, predictions))

if __name__ == "__main__":
    main()
