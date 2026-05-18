import flwr as fl
from typing import Dict, Optional, Tuple
from flwr.common import Metrics

# Function to average accuracy from all hospitals
def weighted_average(metrics):

    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {
        "accuracy": sum(accuracies) / sum(examples)
    }

# Custom Federated Averaging Strategy
strategy = fl.server.strategy.FedAvg(

    # All hospitals participate
    fraction_fit=1.0,

    # Minimum hospitals needed
    min_fit_clients=3,
    min_available_clients=3,
    min_evaluate_clients=3,

    # Train for 20 rounds
    evaluate_metrics_aggregation_fn=weighted_average,
)

print("Federated Strategy Initialized")