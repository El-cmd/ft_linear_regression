def calculate_mae(predictions, real_prices):
    """Calcule l'erreur absolue moyenne entre prédictions et prix réels."""
    if not predictions or len(predictions) != len(real_prices):
        raise ValueError(
            "Predictions and real prices must have the same non-zero length"
        )

    absolute_errors = [
        abs(prediction - real_price)
        for prediction, real_price in zip(predictions, real_prices)
    ]

    return sum(absolute_errors) / len(absolute_errors)
