def estimate_price(mileage, theta0, theta1):
    """Estime le prix d'une voiture à partir de son kilométrage."""
    return theta0 + theta1 * mileage


def normalize_mileage(mileage, min_mileage, max_mileage):
    """Normalise un kilométrage entre 0 et 1 avec la méthode min-max."""
    if min_mileage == max_mileage:
        raise ValueError("Cannot normalize identical mileage values")

    return (mileage - min_mileage) / (max_mileage - min_mileage)
