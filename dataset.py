import csv


def load_data(file_path):
    """Lit le fichier CSV et retourne les kilométrages et les prix."""
    # Chaque colonne du CSV est stockée dans sa propre liste.
    mileages = []
    prices = []

    # DictReader utilise la première ligne du CSV comme noms de colonnes.
    with file_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            # Les valeurs sont converties en nombres pour les futurs calculs.
            mileages.append(float(row["km"]))
            prices.append(float(row["price"]))

    return mileages, prices
