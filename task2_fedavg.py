"""
Task 2: FedAvg Implementation - Communication Efficiency Study

Experiments:
1. Varying local epochs K ∈ {1, 5, 10, 20}
2. Varying client sampling fraction ∈ {1.0, 0.5, 0.2}
"""

import torch
import argparse
import json
import os
from datetime import datetime

from config import CONFIG, print_config
from model import get_model
from data_utils import (
    set_seed,
    load_cifar10,
    dirichlet_partition,
    get_client_loaders,
    get_test_loader,
)
from fed_utils import federated_averaging


def save_results(results, args, filename):
    """Save results to JSON file"""
    os.makedirs("logs", exist_ok=True)

    output = {
        "task": "task2_fedavg",
        "experiment": args.experiment,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_clients": CONFIG["num_clients"],
            "total_rounds": args.rounds,
            "local_epochs": args.k,
            "learning_rate": CONFIG["lr"],
            "batch_size": CONFIG["local_batch_size"],
            "alpha": args.alpha,
            "client_sampling_frac": args.sampling_frac,
            "random_seed": CONFIG["random_seed"],
        },
        "results": {
            "rounds": results["rounds"],
            "test_accuracy": results["test_accuracy"],
            "test_loss": results["test_loss"],
            "train_time": results["train_time"],
            "weight_divergence": results["weight_divergence"],
        },
        "summary": {
            "final_test_accuracy": results["test_accuracy"][-1],
            "best_test_accuracy": max(results["test_accuracy"]),
            "total_time": sum(results["train_time"]),
            "avg_time_per_round": sum(results["train_time"])
            / len(results["train_time"]),
        },
    }

    filepath = os.path.join("logs", filename)
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {filepath}")


def run_task2(args):
    """Run Task 2 experiments"""

    # Set seed for reproducibility
    set_seed(CONFIG["random_seed"])

    # Print configuration
    print_config()
    print(f"\nTask 2 - Experiment: {args.experiment}")
    print(f"  Local epochs (K): {args.k}")
    print(f"  Client sampling: {args.sampling_frac * 100:.0f}%")
    print(f"  Data distribution: α={args.alpha}")
    print(f"  Run ID: {args.run_id}")

    # Load data
    print("\nLoading CIFAR-10...")
    trainset, testset = load_cifar10()

    # Partition data using Dirichlet
    print(f"\nPartitioning data (Dirichlet α={args.alpha})...")
    client_indices = dirichlet_partition(
        trainset,
        num_clients=CONFIG["num_clients"],
        alpha=args.alpha,
        seed=CONFIG["random_seed"],
    )

    # Create client loaders
    client_loaders = get_client_loaders(
        trainset, client_indices, batch_size=CONFIG["local_batch_size"]
    )

    # Create test loader
    test_loader = get_test_loader(testset, batch_size=CONFIG["test_batch_size"])

    # Initialize global model
    print("\nInitializing global model...")
    global_model = get_model()

    # Run FedAvg
    print(f"\n{'='*70}")
    print(f"Running FedAvg - {args.experiment}")
    print(f"{'='*70}")

    results = federated_averaging(
        global_model=global_model,
        client_loaders=client_loaders,
        test_loader=test_loader,
        num_rounds=args.rounds,
        local_epochs=args.k,
        lr=CONFIG["lr"],
        device=CONFIG["device"],
        client_sampling_frac=args.sampling_frac,
    )

    # Save results
    filename = f"task2_{args.experiment}_{args.run_id}.json"
    save_results(results, args, filename)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Task 2: FedAvg Experiments")

    # Experiment type
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=["vary_k", "vary_sampling"],
        help="Experiment type: vary_k or vary_sampling",
    )

    # Hyperparameters
    parser.add_argument(
        "--k", type=int, default=5, help="Number of local epochs (default: 5)"
    )
    parser.add_argument(
        "--sampling_frac",
        type=float,
        default=1.0,
        help="Client sampling fraction (default: 1.0)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=100.0,
        help="Dirichlet alpha for data distribution (default: 100.0 for IID)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=50,
        help="Number of communication rounds (default: 50)",
    )

    # Run identification
    parser.add_argument(
        "--run_id", type=str, required=True, help="Run identifier for logging"
    )

    args = parser.parse_args()

    # Run experiment
    run_task2(args)


if __name__ == "__main__":
    main()
