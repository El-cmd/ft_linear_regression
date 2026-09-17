import json


def save_model_parameters(
    file_path, theta0, theta1, min_mileage, max_mileage
):
    """Sauvegarde les paramètres nécessaires à une future prédiction."""
    parameters = {
        "theta0": theta0,
        "theta1": theta1,
        "min_mileage": min_mileage,
        "max_mileage": max_mileage,
    }

    with file_path.open("w", encoding="utf-8") as json_file:
        json.dump(parameters, json_file, indent=4)


def load_model_parameters(file_path):
    """Charge et valide les paramètres sauvegardés dans le fichier JSON."""
    try:
        with file_path.open("r", encoding="utf-8") as json_file:
            parameters = json.load(json_file)
    except json.JSONDecodeError as error:
        raise ValueError("The model file does not contain valid JSON") from error

    required_parameters = (
        "theta0",
        "theta1",
        "min_mileage",
        "max_mileage",
    )

    try:
        return {
            name: float(parameters[name]) for name in required_parameters
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("The model file contains invalid parameters") from error
