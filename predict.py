import math
from pathlib import Path

from model import estimate_price, normalize_mileage
from model_io import load_model_parameters


THETA_FILE = Path(__file__).parent / "theta.json"


def ask_mileage():
    """Demande un kilométrage valide à l'utilisateur."""
    while True:
        try:
            mileage = float(input("Enter a mileage: "))
        except ValueError:
            print("Error: mileage must be a number")
            continue

        if not math.isfinite(mileage) or mileage < 0:
            print("Error: mileage must be a positive finite number")
            continue

        return mileage


def main():
    try:
        parameters = load_model_parameters(THETA_FILE)
    except (OSError, ValueError) as error:
        print(f"Error: unable to load the trained model: {error}")
        print("Run train.py before making a prediction")
        return

    mileage = ask_mileage()
    min_mileage = parameters["min_mileage"]
    max_mileage = parameters["max_mileage"]

    # Une valeur extérieure au dataset reste calculable, mais sera extrapolée.
    if mileage < min_mileage or mileage > max_mileage:
        print(
            "Warning: mileage is outside the training range "
            f"[{min_mileage:.0f}, {max_mileage:.0f}]"
        )

    # Applique exactement la même normalisation que pendant l'entraînement.
    normalized_mileage = normalize_mileage(
        mileage, min_mileage, max_mileage
    )

    estimated_price = estimate_price(
        normalized_mileage,
        parameters["theta0"],
        parameters["theta1"],
    )

    print(f"Estimated price: {estimated_price:.2f}")


if __name__ == "__main__":
    main()
