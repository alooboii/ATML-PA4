"""
Federated learning utility functions
"""

import torch
import torch.nn as nn
import copy
import time
import numpy as np
from config import CONFIG


def train_local(model, loader, epochs, lr, device):
    """
    Train model locally on client data

    Args:
        model: PyTorch model
        loader: DataLoader for client data
        epochs: Number of local epochs
        lr: Learning rate
        device: Device to train on

    Returns:
        model: Trained model
        train_loss: Average training loss
    """
    model.train()
    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=CONFIG["momentum"],
        weight_decay=CONFIG["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0

    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return model, avg_loss


def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test set

    Args:
        model: PyTorch model
        test_loader: Test DataLoader
        device: Device to evaluate on

    Returns:
        accuracy: Test accuracy (0-1)
        loss: Test loss
    """
    model.eval()
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    accuracy = correct / total
    avg_loss = total_loss / total

    return accuracy, avg_loss


def aggregate_models(global_model, client_models, client_weights):
    """
    Aggregate client models using weighted averaging

    Args:
        global_model: Global model to update
        client_models: List of client models
        client_weights: List of weights (e.g., dataset sizes)

    Returns:
        global_model: Updated global model
    """
    # Normalize weights
    total_weight = sum(client_weights)
    weights = [w / total_weight for w in client_weights]

    # Get global model state dict
    global_state = global_model.state_dict()

    # Initialize aggregated state with zeros
    for key in global_state.keys():
        global_state[key] = torch.zeros_like(global_state[key], dtype=torch.float32)

    # Weighted sum of client model parameters
    for client_model, weight in zip(client_models, weights):
        client_state = client_model.state_dict()
        for key in global_state.keys():
            global_state[key] += client_state[key].float() * weight

    # Update global model
    global_model.load_state_dict(global_state)

    return global_model


def compute_weight_divergence(global_model, client_models):
    """
    Compute average weight divergence between clients and global model

    Args:
        global_model: Global model
        client_models: List of client models

    Returns:
        avg_divergence: Average L2 distance
    """
    global_state = global_model.state_dict()
    divergences = []

    for client_model in client_models:
        client_state = client_model.state_dict()
        divergence = 0.0

        for key in global_state.keys():
            if "weight" in key or "bias" in key:
                diff = global_state[key].float() - client_state[key].float()
                divergence += torch.norm(diff).item() ** 2

        divergences.append(np.sqrt(divergence))

    avg_divergence = np.mean(divergences)
    return avg_divergence


def federated_averaging(
    global_model,
    client_loaders,
    test_loader,
    num_rounds,
    local_epochs,
    lr,
    device,
    client_sampling_frac=1.0,
    log_fn=None,
):
    """
    FedAvg algorithm

    Args:
        global_model: Initial global model
        client_loaders: List of client data loaders
        test_loader: Test data loader
        num_rounds: Number of communication rounds
        local_epochs: Number of local epochs (K)
        lr: Learning rate
        device: Device to train on
        client_sampling_frac: Fraction of clients to sample each round
        log_fn: Optional logging function

    Returns:
        results: Dictionary with training history
    """
    num_clients = len(client_loaders)
    num_selected = max(1, int(num_clients * client_sampling_frac))

    results = {
        "rounds": [],
        "test_accuracy": [],
        "test_loss": [],
        "train_time": [],
        "weight_divergence": [],
    }

    print(f"\nStarting FedAvg:")
    print(f"  Clients: {num_clients} (selecting {num_selected} per round)")
    print(f"  Rounds: {num_rounds}")
    print(f"  Local epochs: {local_epochs}")
    print(f"  Learning rate: {lr}")
    print("=" * 70)

    for round_idx in range(num_rounds):
        round_start = time.time()

        # Sample clients
        selected_clients = np.random.choice(num_clients, num_selected, replace=False)

        client_models = []
        client_weights = []

        # Local training on selected clients
        for client_id in selected_clients:
            # Copy global model to client
            client_model = copy.deepcopy(global_model)

            # Train locally
            client_model, _ = train_local(
                client_model, client_loaders[client_id], local_epochs, lr, device
            )

            client_models.append(client_model)
            client_weights.append(len(client_loaders[client_id].dataset))

        # Aggregate models
        global_model = aggregate_models(global_model, client_models, client_weights)

        # Evaluate global model
        test_acc, test_loss = evaluate_model(global_model, test_loader, device)

        # Compute weight divergence
        weight_div = compute_weight_divergence(global_model, client_models)

        round_time = time.time() - round_start

        # Log results
        results["rounds"].append(round_idx)
        results["test_accuracy"].append(test_acc)
        results["test_loss"].append(test_loss)
        results["train_time"].append(round_time)
        results["weight_divergence"].append(weight_div)

        # Print progress
        print(
            f"Round {round_idx+1:3d}/{num_rounds} | "
            f"Acc: {test_acc:.4f} | "
            f"Loss: {test_loss:.4f} | "
            f"Div: {weight_div:.4f} | "
            f"Time: {round_time:.2f}s"
        )

        # Optional custom logging
        if log_fn:
            log_fn(round_idx, test_acc, test_loss, weight_div, round_time)

    print("=" * 70)
    print(f"Final Test Accuracy: {results['test_accuracy'][-1]:.4f}")
    print(f"Best Test Accuracy: {max(results['test_accuracy']):.4f}")

    return results


if __name__ == "__main__":
    print("Federated learning utilities loaded successfully")
