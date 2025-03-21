from model.model import SimpleNN
from data.dataloader import get_dataloader
from train.trainer import train
import torch

if __name__ == "__main__":
    config = ...
    dataloader = get_dataloader(config['data_dir'], config['batch_size'])
    model = SimpleCNN()
    train(model, dataloader, config['epochs'], config['device'])