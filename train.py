from pathlib import Path

from dataset import load_data
from model import estimate_price, normalize_mileage
from model_io import save_model_parameters
from visualization import TrainingVisualizer


# Construit le chemin vers le fichier CSV à partir de l'emplacement de ce script.
DATA_FILE = Path(__file__).parent / "data" / "data.csv"
GRAPH_FILE = Path(__file__).parent / "data_plot.png"
THETA_FILE = Path(__file__).parent / "theta.json"
LEARNING_RATE = 0.1
TOLERANCE = 1e-6
MAX_ITERATIONS = 100_000
GRAPH_REFRESH_INTERVAL = 10


def normalize_mileages(mileages):
    """Normalise tous les kilométrages et retourne aussi leurs bornes."""
    # Calcule les bornes nécessaires à la normalisation min-max.
    min_mileage = min(mileages)
    max_mileage = max(mileages)

    # Normalise chaque kilométrage entre 0 et 1.
    normalized_mileages = [
        normalize_mileage(mileage, min_mileage, max_mileage)
        for mileage in mileages
    ]

    return normalized_mileages, min_mileage, max_mileage


def initialize_parameters():
    """Initialise les paramètres du modèle avant son entraînement."""
    return 0.0, 0.0


def calculate_predictions_and_errors(mileages, prices, theta0, theta1):
    """Calcule les prédictions et les erreurs sans modifier les theta."""
    predictions = []
    errors = []

    for mileage, real_price in zip(mileages, prices):
        prediction = estimate_price(mileage, theta0, theta1)
        error = prediction - real_price
        predictions.append(prediction)
        errors.append(error)

    return predictions, errors


def calculate_parameter_updates(mileages, errors, learning_rate):
    """Calcule tmp_theta0 et tmp_theta1 selon les formules du sujet."""
    if not errors or len(mileages) != len(errors):
        raise ValueError("Mileages and errors must have the same non-zero length")

    # m correspond au nombre de voitures du dataset.
    m = len(errors)

    tmp_theta0 = learning_rate * sum(errors) / m
    tmp_theta1 = learning_rate * sum(
        error * mileage for mileage, error in zip(mileages, errors)
    ) / m

    return tmp_theta0, tmp_theta1


def train_model(
    mileages,
    prices,
    theta0,
    theta1,
    learning_rate,
    tolerance,
    max_iterations,
    on_iteration=None,
):
    """Entraîne le modèle jusqu'à ce que les corrections soient presque nulles."""
    iteration = 0

    while iteration < max_iterations:
        # Recalcule les erreurs avec les theta les plus récents.
        _, errors = calculate_predictions_and_errors(
            mileages, prices, theta0, theta1
        )

        # Calcule les deux corrections avant de modifier les theta.
        tmp_theta0, tmp_theta1 = calculate_parameter_updates(
            mileages, errors, learning_rate
        )

        # Arrête l'entraînement lorsque les deux corrections sont presque nulles.
        if abs(tmp_theta0) < tolerance and abs(tmp_theta1) < tolerance:
            return theta0, theta1, iteration

        # Calcule les deux nouvelles valeurs à partir des anciens theta.
        new_theta0 = theta0 - tmp_theta0
        new_theta1 = theta1 - tmp_theta1

        # Met seulement ensuite les deux theta à jour simultanément.
        theta0, theta1 = new_theta0, new_theta1
        iteration += 1

        # Informe un observateur éventuel, sans dépendre de Matplotlib.
        if on_iteration is not None:
            on_iteration(iteration, theta0, theta1)

    raise RuntimeError("The model did not converge before the iteration limit")


def display_training_data(mileages, normalized_mileages, prices, predictions, errors):
    """Affiche les données et les résultats dans le terminal."""
    print("Mileage (km) | Normalized | Real price | Prediction | Error")
    print("-------------|------------|------------|------------|------")

    for mileage, normalized_mileage, price, prediction, error in zip(
        mileages, normalized_mileages, prices, predictions, errors
    ):
        print(
            f"{mileage:12.0f} | {normalized_mileage:10.6f} | "
            f"{price:10.0f} | {prediction:10.2f} | {error:.2f}"
        )


def main():
    # Charge et prépare les données utilisées par l'entraînement.
    mileages, prices = load_data(DATA_FILE)
    normalized_mileages, min_mileage, max_mileage = normalize_mileages(mileages)

    # Initialise les paramètres du modèle avant la descente de gradient.
    theta0, theta1 = initialize_parameters()
    print(f"Initial theta0: {theta0}")
    print(f"Initial theta1: {theta1}")

    # Prépare la fenêtre qui animera la droite toutes les 10 itérations.
    visualizer = TrainingVisualizer(
        mileages, prices, GRAPH_REFRESH_INTERVAL
    )
    visualizer.update(0, theta0, theta1)

    # Répète la descente de gradient jusqu'à la convergence.
    theta0, theta1, iterations = train_model(
        normalized_mileages,
        prices,
        theta0,
        theta1,
        LEARNING_RATE,
        TOLERANCE,
        MAX_ITERATIONS,
        on_iteration=visualizer.update,
    )

    # Recalcule les prédictions et les erreurs avec les theta entraînés.
    predictions, errors = calculate_predictions_and_errors(
        normalized_mileages, prices, theta0, theta1
    )

    # Affiche les données dans le terminal sous la forme d'un tableau.
    display_training_data(
        mileages, normalized_mileages, prices, predictions, errors
    )

    print(f"Converged after {iterations} iterations")
    print(f"Trained theta0: {theta0}")
    print(f"Trained theta1: {theta1}")

    # Sauvegarde le modèle et les bornes utilisées pour la normalisation.
    save_model_parameters(
        THETA_FILE, theta0, theta1, min_mileage, max_mileage
    )
    print(f"Model saved to: {THETA_FILE}")

    # Affiche la droite finale et enregistre le graphique.
    visualizer.finish(iterations, theta0, theta1, GRAPH_FILE)


# Exécute main() uniquement lorsque ce fichier est lancé directement.
if __name__ == "__main__":
    main()
