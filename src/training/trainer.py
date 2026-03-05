import numpy as np
import torch
from torch.utils.data import DataLoader

class UnifiedTrainer:
    def __init__(self, models, dataloaders, optimizer, criterion, device='cpu'):
        self.models = models  # list of models
        self.dataloaders = dataloaders  # dict of DataLoaders {model_name: DataLoader}
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self):
        for model in self.models:
            model.train()
            dataloader = self.dataloaders[type(model).__name__]
            epoch_loss = 0.0

            for batch in dataloader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()

                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            print(f'Training {type(model).__name__} - Loss: {epoch_loss / len(dataloader)}')

        return epoch_loss / len(dataloader)

    def validate_epoch(self):
        for model in self.models:
            model.eval()
            dataloader = self.dataloaders[type(model).__name__]
            epoch_loss = 0.0

            with torch.no_grad():
                for batch in dataloader:
                    inputs, targets = batch
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    outputs = model(inputs)
                    loss = self.criterion(outputs, targets)
                    epoch_loss += loss.item()

            print(f'Validation {type(model).__name__} - Loss: {epoch_loss / len(dataloader)}')

        return epoch_loss / len(dataloader)