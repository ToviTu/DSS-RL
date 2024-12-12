from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import math

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].squeeze(0), self.y[idx].squeeze(0)

def test_model(model, dataset):
    model.eval()
    test_loader = DataLoader(
        dataset,
        batch_size=1000, shuffle=False
    )

    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            feature = data
            output = model(feature.to('cuda'))
            pred = output.argmax(dim=1, keepdim=True).to('cpu')
            correct += pred.eq(target.view_as(pred)).sum().item()

    print(f'Test Accuracy: {correct / len(test_loader.dataset) * 100:.2f}%')
    return correct / len(test_loader.dataset)

def train_model(model, train_loader, criterion, optim, epochs=10, verbose=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optim.zero_grad()
            loss.backward()
            optim.step()

            if (i + 1) % 100 == 0 and verbose:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

def dpp(kernel_matrix, max_length, epsilon=1E-10):
    """
    Our proposed fast implementation of the greedy algorithm
    :param kernel_matrix: 2-d array
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list
    """
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        di2s[selected_item] = -np.inf
        selected_item = np.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)

        #print(f"Progress: {len(selected_items)}/{max_length}")
    return selected_items

def dolly_instruct_collator(batch):
    return {
        "text": batch["instruction"],
    }

def dolly_all_collator(batch):
    return {
        "text": " ".join([
            batch["context"],
            batch["instruction"],
            batch["response"]
        ])
    }

def alpaca_instruct_collator(batch):
    return {
        "text": batch["instruction"],
    }

def alpaca_all_collator(batch):
    return {
        "text": " ".join([
            batch["input"],
            batch["instruction"],
            batch["output"]
        ])
    }