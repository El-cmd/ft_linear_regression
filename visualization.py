import matplotlib.pyplot as plt


class TrainingVisualizer:
    """Anime la droite de régression sans contenir la logique d'entraînement."""

    def __init__(self, mileages, prices, refresh_interval=10):
        if refresh_interval <= 0:
            raise ValueError("Refresh interval must be greater than zero")

        self.mileages = mileages
        self.prices = prices
        self.refresh_interval = refresh_interval
        self.min_mileage = min(mileages)
        self.max_mileage = max(mileages)
        self.has_graphical_backend = plt.get_backend().lower() != "agg"

        # Active le mode interactif pour pouvoir modifier la même fenêtre.
        plt.ion()
        self.figure, self.axes = plt.subplots()

        # Les points restent fixes pendant tout l'entraînement.
        self.axes.scatter(mileages, prices, color="blue", label="Dataset")

        # Cette ligne sera déplacée lorsque theta0 et theta1 changeront.
        (self.regression_line,) = self.axes.plot(
            [], [], color="red", label="Linear regression"
        )

        self.axes.set_xlabel("Mileage (km)")
        self.axes.set_ylabel("Price")
        self.axes.grid(True)
        self.axes.legend()
        self.figure.tight_layout()

        if self.has_graphical_backend:
            plt.show(block=False)

    def update(self, iteration, theta0, theta1, force=False):
        """Rafraîchit la droite selon l'intervalle choisi."""
        if not force and iteration % self.refresh_interval != 0:
            return

        # Les extrémités min et max correspondent à 0 et 1 après normalisation.
        line_x = [self.min_mileage, self.max_mileage]
        line_y = [theta0, theta0 + theta1]
        self.regression_line.set_data(line_x, line_y)
        self.axes.set_title(f"Training - iteration {iteration}")

        # Adapte l'axe vertical pour garder les points et la droite visibles.
        all_prices = self.prices + line_y
        min_price = min(all_prices)
        max_price = max(all_prices)
        price_margin = max((max_price - min_price) * 0.1, 1.0)
        self.axes.set_ylim(min_price - price_margin, max_price + price_margin)

        if self.has_graphical_backend:
            self.figure.canvas.draw_idle()
            self.figure.canvas.flush_events()
            plt.pause(0.001)

    def finish(self, iteration, theta0, theta1, output_file):
        """Affiche la droite finale et enregistre le graphique."""
        self.update(iteration, theta0, theta1, force=True)
        self.figure.savefig(output_file)

        plt.ioff()
        if self.has_graphical_backend:
            plt.show()
        else:
            print(f"Graph saved to: {output_file}")

        plt.close(self.figure)
