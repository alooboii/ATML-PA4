"""
Task 3: Exploring Data Heterogeneity Impact

Study how label heterogeneity affects FedAvg performance using Dirichlet distribution.
Test different α values: {100, 1.0, 0.1, 0.05}
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
        "task": "task3_heterogeneity",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_clients": CONFIG["num_clients"],
            "total_rounds": args.rounds,
            "local_epochs": args.k,
            "learning_rate": CONFIG["lr"],
            "batch_size": CONFIG["local_batch_size"],
            "alpha": args.alpha,
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
            "avg_weight_divergence": sum(results["weight_divergence"])
            / len(results["weight_divergence"]),
        },
    }

    filepath = os.path.join("logs", filename)
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {filepath}")


def run_task3(args):
    """Run Task 3 experiments"""

    # Set seed for reproducibility
    set_seed(CONFIG["random_seed"])

    # Print configuration
    print_config()
    print(f"\nTask 3 - Data Heterogeneity Study")
    print(f"  Dirichlet α: {args.alpha}")
    print(f"  Local epochs (K): {args.k}")
    print(f"  Run ID: {args.run_id}")

    # Interpret heterogeneity level
    if args.alpha >= 100:
        hetero_level = "IID (baseline)"
    elif args.alpha >= 1.0:
        hetero_level = "Moderate heterogeneity"
    elif args.alpha >= 0.1:
        hetero_level = "High heterogeneity"
    else:
        hetero_level = "Extreme heterogeneity"

    print(f"  Heterogeneity level: {hetero_level}")

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
    print(f"Running FedAvg with α={args.alpha}")
    print(f"{'='*70}")

    results = federated_averaging(
        global_model=global_model,
        client_loaders=client_loaders,
        test_loader=test_loader,
        num_rounds=args.rounds,
        local_epochs=args.k,
        lr=CONFIG["lr"],
        device=CONFIG["device"],
        client_sampling_frac=1.0,  # All clients participate in Task 3
    )

    # Save results
    filename = f"task3_alpha{args.alpha}_{args.run_id}.json"
    save_results(results, args, filename)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nKey Findings:")
    print(f"  Final accuracy: {results['test_accuracy'][-1]:.4f}")
    print(f"  Best accuracy: {max(results['test_accuracy']):.4f}")
    print(
        f"  Avg weight divergence: {sum(results['weight_divergence'])/len(results['weight_divergence']):.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Task 3: Data Heterogeneity Impact")

    # Hyperparameters
    parser.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="Dirichlet concentration parameter (100=IID, 1.0=moderate, 0.1=high, 0.05=extreme)",
    )
    parser.add_argument(
        "--k", type=int, default=5, help="Number of local epochs (default: 5)"
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

    # Validate alpha
    if args.alpha <= 0:
        raise ValueError("Alpha must be positive")

    # Run experiment
    run_task3(args)


if __name__ == "__main__":
    main()
