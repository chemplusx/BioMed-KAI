import math
from collections import Counter
from sklearn.metrics import precision_score, recall_score, fbeta_score

# Entropy Calculation
def calculate_entropy(probabilities):
    """
    Calculate entropy for entity detection uncertainty.
    Args:
        probabilities (list): List of probabilities for detected entities.
    Returns:
        float: Entropy value.
    """
    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
    return entropy

# F1β Score Calculation
def calculate_f1_beta(actual, expected, beta=1.0):
    """
    Calculate F1β score given actual and expected textual outputs.
    Args:
        actual (list): List of actual entities detected.
        expected (list): List of ground truth entities.
        beta (float): Weighting factor for precision vs recall.
    Returns:
        tuple: Precision, Recall, and F1β score.
    """
    # Convert entities to binary for precision/recall calculation
    all_entities = list(set(expected + actual))
    y_true = [1 if entity in expected else 0 for entity in all_entities]
    y_pred = [1 if entity in actual else 0 for entity in all_entities]

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1_beta = fbeta_score(y_true, y_pred, beta=beta)

    return precision, recall, f1_beta

# Example Usage
if __name__ == "__main__":
    # Example entity probabilities (from model detection)
    entity_probabilities = [0.8, 0.7, 0.6, 0.5]  # Example detection probabilities

    # Entropy Calculation
    entropy = calculate_entropy(entity_probabilities)
    print(f"Entropy-Based Entity Detection Uncertainty: {entropy:.4f}")

    # Example actual vs expected textual entities
    actual_entities = ["Entity1", "Entity2", "Entity3"]  # Model detected entities
    expected_entities = ["Entity1", "Entity3", "Entity4"]  # Ground truth entities

    # F1β Calculation (default β=1 for balanced precision/recall)
    precision, recall, f1_beta = calculate_f1_beta(actual_entities, expected_entities, beta=1.0)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1β Score (β=1): {f1_beta:.4f}")

    # Custom β for emphasis on recall
    beta = 2.0  # Emphasize recall
    _, _, f1_beta_recall = calculate_f1_beta(actual_entities, expected_entities, beta=beta)
    print(f"F1β Score (β={beta}): {f1_beta_recall:.4f}")
