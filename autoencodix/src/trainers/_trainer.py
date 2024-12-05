from torch.utils.data import DataLoader
from autoencodix.src.core._result import Result
import torch


class Trainer:
    """
    Handles model training process
    """

    def __init__(self):
        pass

    def train(
        self,
        model: torch.nn.Module,
        train: torch.utils.data.Dataset,
        valid: torch.utils.data.Dataset,
        result: Result) -> None:

            train_loader = DataLoader(train, batch_size=32, shuffle=True)
            valid_loader = DataLoader(valid, batch_size=32, shuffle=False)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = torch.nn.MSELoss()
            epochs = 10
            for epoch in range(epochs):
                model.train()
                for x, y in train_loader:
                    optimizer.zero_grad()
                    y_pred = model(x)
                    loss = loss_fn(y_pred, y)
                    loss.backward()
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    val_loss = sum(
                        loss_fn(model(x), y) for x, y in valid_loader
                    ) / len(valid_loader)
                print(f"Epoch {epoch}: val_loss={val_loss.item()}")
