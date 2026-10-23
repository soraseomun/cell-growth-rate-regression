# =======================================================================
# CLASSES
# Define custom classes in this file.
# =======================================================================

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

import src.functions as sf
from src.constants import *


class GrowthRateStudy:
    def __init__(self, filepath, test_size, random_state, cv):
        """
        Utility class used to build, compare, and interpret regression models
        that predict the specific growth rate of cells.

        Parameters
        ----------
        filepath : str
            Input CSV dataset path.
        test_size : float
            Fraction of the data that is kept aside as the test set.
        random_state : int
            Seed used when splitting the data.
        cv : int
            Number of cross-validation folds.
        """
        self.df = pd.read_csv(filepath)

        # The target is the first column and the features are the remaining columns
        self.target = self.df.columns[0]
        self.features = self.df.columns[1:]

        self.x = self.df.loc[:, self.features].to_numpy()
        self.y = self.df.loc[:, self.target].to_numpy()

        # The test set is only used at the end to check the final models
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=test_size, random_state=random_state)

        self.cv = cv
        self.gscv_poly = None
        self.gscv_knn = None

        print(f"Loaded {self.x.shape[0]} samples with {self.x.shape[1]} features.")
        print(f"Training samples: {self.x_train.shape[0]} | Test samples: {self.x_test.shape[0]}")

    # ===================================================================
    #   Hyperparameter tuning
    # ===================================================================

    def tune_polynomial(self, degrees):
        """
        Finds the best polynomial degree using GridSearchCV on the training set.

        Parameters
        ----------
        degrees : list[int]
            Polynomial degrees to try.
        """
        param_grid = dict(polynomialfeatures__degree=degrees)
        self.gscv_poly = GridSearchCV(sf.make_poly_model(), param_grid=param_grid, cv=self.cv,
                                      scoring="neg_root_mean_squared_error", return_train_score=True)
        self.gscv_poly.fit(self.x_train, self.y_train)
        print(f"Best polynomial: {self.gscv_poly.best_params_}")

    def tune_knn(self, n_neighbors, weights):
        """
        Finds the best number of neighbors and weighting using GridSearchCV
        on the training set.

        Parameters
        ----------
        n_neighbors : list[int]
            Values of k to try.
        weights : list[str]
            Weighting options to try ("uniform" and/or "distance").
        """
        param_grid = dict(kneighborsregressor__n_neighbors=n_neighbors,
                          kneighborsregressor__weights=weights)
        self.gscv_knn = GridSearchCV(sf.make_knn_model(), param_grid=param_grid, cv=self.cv,
                                     scoring="neg_root_mean_squared_error", return_train_score=True)
        self.gscv_knn.fit(self.x_train, self.y_train)
        print(f"Best kNN: {self.gscv_knn.best_params_}")

    def get_best_degree(self):
        """
        Returns
        -------
        int
            Polynomial degree with the lowest validation RMSE.
        """
        return self.gscv_poly.best_params_["polynomialfeatures__degree"]

    def get_best_k(self, weights):
        """
        Finds the best k for one weighting option.

        Parameters
        ----------
        weights : str
            "uniform" or "distance".

        Returns
        -------
        int
            Value of k with the lowest validation RMSE for this weighting.
        """
        df = pd.DataFrame(self.gscv_knn.cv_results_)
        mask = (df.loc[:, "param_kneighborsregressor__weights"] == weights)
        n_neighbors = df.loc[mask, "param_kneighborsregressor__n_neighbors"].to_numpy()
        scores = df.loc[mask, "mean_test_score"].to_numpy()
        return int(n_neighbors[np.argmax(scores)])  # Highest score = lowest RMSE

    def get_poly_label(self):
        return f"Polynomial (degree {self.get_best_degree()})"

    def get_knn_label(self):
        k = self.gscv_knn.best_params_["kneighborsregressor__n_neighbors"]
        weights = self.gscv_knn.best_params_["kneighborsregressor__weights"]
        return f"kNN ({weights}, k={k})"

    # ===================================================================
    #   Exports
    # ===================================================================

    def export_data_overview(self, filepath):
        """
        Creates and saves scatter plots of the growth rate against each feature.

        Parameters
        ----------
        filepath : str
            Output figure path.
        """
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(2 * FIG_WIDTH, 1.6 * FIG_HEIGHT), dpi=200,
                                 layout="constrained", sharey=True)

        for idx, (ax, feature) in enumerate(zip(np.ravel(axes), self.features)):
            ax.scatter(self.x[:, idx], self.y, s=20, color=COLORS[0], alpha=0.6, edgecolors="black", linewidth=0.5)
            sf.format_axes(ax, feature, self.target)

        fig.savefig(filepath)
        plt.close(fig)

    def export_grid_search_poly(self, filepath):
        """
        Saves the GridSearchCV results of the polynomial model.

        Parameters
        ----------
        filepath : str
            Output CSV path.
        """
        df = pd.DataFrame(self.gscv_poly.cv_results_)

        # Recall that we used negative RMSE, so we flip the sign to report RMSE
        df_export = pd.DataFrame({
            "degree": df.loc[:, "param_polynomialfeatures__degree"].to_numpy(),
            "train_rmse_mean": -df.loc[:, "mean_train_score"].to_numpy(),
            "valid_rmse_mean": -df.loc[:, "mean_test_score"].to_numpy(),
            "valid_rmse_std": df.loc[:, "std_test_score"].to_numpy(),
            "rank": df.loc[:, "rank_test_score"].to_numpy(),
        })
        df_export.round(4).to_csv(filepath, index=False)

        print("\nPolynomial grid search:")
        print(df_export.round(4).to_string(index=False))

    def export_grid_search_knn(self, filepath):
        """
        Saves the GridSearchCV results of the kNN model, sorted from best to worst.

        Parameters
        ----------
        filepath : str
            Output CSV path.
        """
        df = pd.DataFrame(self.gscv_knn.cv_results_)
        df = df.sort_values(by=["rank_test_score"], ascending=True)

        df_export = pd.DataFrame({
            "n_neighbors": df.loc[:, "param_kneighborsregressor__n_neighbors"].to_numpy(),
            "weights": df.loc[:, "param_kneighborsregressor__weights"].to_numpy(),
            "train_rmse_mean": np.abs(df.loc[:, "mean_train_score"].to_numpy()),
            "valid_rmse_mean": -df.loc[:, "mean_test_score"].to_numpy(),
            "valid_rmse_std": df.loc[:, "std_test_score"].to_numpy(),
            "rank": df.loc[:, "rank_test_score"].to_numpy(),
        })
        df_export.round(4).to_csv(filepath, index=False)

        print("\nkNN grid search (top 5):")
        print(df_export.head(5).round(4).to_string(index=False))

    def export_validation_curve_poly(self, filepath, plot_n=4):
        """
        Creates and saves the validation curve of the polynomial model.

        Parameters
        ----------
        filepath : str
            Output figure path.
        plot_n : int
            Number of degrees to plot. Higher degrees are left out because
            their validation RMSE is so large that the other points become
            impossible to read (they are still in the table).
        """
        df = pd.DataFrame(self.gscv_poly.cv_results_)
        degrees = df.loc[:, "param_polynomialfeatures__degree"].to_numpy()
        rmse_train = -df.loc[:, "mean_train_score"].to_numpy()
        rmse_valid = -df.loc[:, "mean_test_score"].to_numpy()

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=200, layout="constrained")

        ax.plot(degrees[:plot_n], rmse_train[:plot_n], label="Train (Poly)", color=COLORS[0], linestyle="-",
                marker="o")
        ax.plot(degrees[:plot_n], rmse_valid[:plot_n], label="Valid (Poly)", color=COLORS[0], linestyle="--",
                marker="o")
        sf.format_axes(ax, "Polynomial Degree", f"RMSE [{self.target.split(' [')[-1][:-1]}]")
        ax.legend(fontsize=FONTSIZE_LEGEND)
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(MultipleLocator(1))

        fig.savefig(filepath)
        plt.close(fig)

    def export_validation_curve_knn(self, filepath):
        """
        Creates and saves the validation curves of the kNN model.

        Parameters
        ----------
        filepath : str
            Output figure path.
        """
        df = pd.DataFrame(self.gscv_knn.cv_results_)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=200, layout="constrained")

        for i, weights in enumerate(KNN_WEIGHTS, start=1):
            mask = (df.loc[:, "param_kneighborsregressor__weights"] == weights)
            df_sub = df.loc[mask, :].sort_values(by=["param_kneighborsregressor__n_neighbors"], ascending=True)
            n_neighbors = df_sub.loc[:, "param_kneighborsregressor__n_neighbors"].to_numpy()
            rmse_train = np.abs(df_sub.loc[:, "mean_train_score"].to_numpy())
            rmse_valid = -df_sub.loc[:, "mean_test_score"].to_numpy()

            ax.plot(n_neighbors, rmse_train, label=f"Train (kNN: {weights})", color=COLORS[i], linestyle="-")
            ax.plot(n_neighbors, rmse_valid, label=f"Valid (kNN: {weights})", color=COLORS[i], linestyle="--")

        sf.format_axes(ax, "k", f"RMSE [{self.target.split(' [')[-1][:-1]}]")
        ax.legend(fontsize=FONTSIZE_LEGEND)
        ax.set_ylim(bottom=-0.005)  # Slightly below zero so the distance-weighted training curve is visible

        fig.savefig(filepath)
        plt.close(fig)

    def export_model_comparison(self, filepath):
        """
        Compares the best polynomial and kNN models with reasonable
        alternatives using cross-validation and the test set.

        Parameters
        ----------
        filepath : str
            Output CSV path.
        """
        best_degree = self.get_best_degree()

        # Alternatives: one degree lower and one degree higher than the best
        # polynomial, and the best k for each kNN weighting option.
        models = {}
        for degree in sorted({max(1, best_degree - 1), best_degree, best_degree + 1}):
            models[f"Polynomial (degree {degree})"] = sf.make_poly_model(degree)
        for weights in KNN_WEIGHTS:
            k = self.get_best_k(weights)
            models[f"kNN ({weights}, k={k})"] = sf.make_knn_model(k, weights)

        records = []
        for name, model in models.items():
            results = sf.evaluate_model(model, self.x_train, self.y_train, self.x_test, self.y_test, self.cv)
            records.append([name, results["train_rmse"], results["valid_rmse_mean"], results["valid_rmse_std"],
                            results["valid_r2_mean"], results["test_rmse"], results["test_r2"]])

        columns = ["model", "train_rmse", "valid_rmse_mean", "valid_rmse_std", "valid_r2_mean",
                   "test_rmse", "test_r2"]
        df_export = pd.DataFrame(records, columns=columns)
        df_export.round(4).to_csv(filepath, index=False)

        print("\nModel comparison:")
        print(df_export.round(4).to_string(index=False))

    def export_learning_curve(self, n_samples_list, seed, filepath_table, filepath_figure):
        """
        Repeats the grid searches on random subsets of increasing size to see
        how each model improves as more data becomes available.

        Parameters
        ----------
        n_samples_list : list[int]
            Number of samples in each subset.
        seed : int
            Seed of the random number generator used to select the subsets.
        filepath_table : str
            Output CSV path.
        filepath_figure : str
            Output figure path.
        """
        rng = np.random.default_rng(seed=seed)
        n_total = self.x.shape[0]
        records = []

        for n_samples in n_samples_list:
            # Select a random subset of n_samples from the dataset
            idxs = rng.choice(n_total, size=min(n_samples, n_total), replace=False)
            x = self.x[idxs]
            y = self.y[idxs]

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE,
                                                                random_state=RANDOM_STATE)

            # k cannot be larger than the number of samples in a training fold
            k_max = int(x_train.shape[0] * (self.cv - 1) / self.cv)
            param_grid_knn = dict(kneighborsregressor__n_neighbors=[k for k in KNN_N_NEIGHBORS if k <= k_max],
                                  kneighborsregressor__weights=KNN_WEIGHTS)

            gscv_poly = GridSearchCV(sf.make_poly_model(), param_grid=dict(polynomialfeatures__degree=POLY_DEGREES),
                                     cv=self.cv, scoring="neg_root_mean_squared_error")
            gscv_knn = GridSearchCV(sf.make_knn_model(), param_grid=param_grid_knn,
                                    cv=self.cv, scoring="neg_root_mean_squared_error")
            gscv_poly.fit(x_train, y_train)
            gscv_knn.fit(x_train, y_train)

            records.append([
                x_train.shape[0],
                gscv_poly.best_params_["polynomialfeatures__degree"],
                root_mean_squared_error(y_train, gscv_poly.predict(x_train)),
                root_mean_squared_error(y_test, gscv_poly.predict(x_test)),
                gscv_knn.best_params_["kneighborsregressor__n_neighbors"],
                gscv_knn.best_params_["kneighborsregressor__weights"],
                root_mean_squared_error(y_train, gscv_knn.predict(x_train)),
                root_mean_squared_error(y_test, gscv_knn.predict(x_test)),
            ])

        columns = ["n_training_samples", "poly_degree", "poly_train_rmse", "poly_test_rmse",
                   "knn_k", "knn_weights", "knn_train_rmse", "knn_test_rmse"]
        df_export = pd.DataFrame(records, columns=columns)
        df_export.round(4).to_csv(filepath_table, index=False)

        print("\nLearning curve:")
        print(df_export.round(4).to_string(index=False))

        n = df_export.loc[:, "n_training_samples"].to_numpy()
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=200, layout="constrained")
        ax.plot(n, df_export.loc[:, "poly_train_rmse"], label="RMSE Train (Poly)", color=COLORS[0], linestyle="-",
                marker="o")
        ax.plot(n, df_export.loc[:, "poly_test_rmse"], label="RMSE Test (Poly)", color=COLORS[0], linestyle="--",
                marker="o")
        ax.plot(n, df_export.loc[:, "knn_train_rmse"], label="RMSE Train (kNN)", color=COLORS[1], linestyle="-",
                marker="D")
        ax.plot(n, df_export.loc[:, "knn_test_rmse"], label="RMSE Test (kNN)", color=COLORS[1], linestyle="--",
                marker="D")
        sf.format_axes(ax, "Number of training samples", f"RMSE [{self.target.split(' [')[-1][:-1]}]")
        ax.legend(fontsize=FONTSIZE_LEGEND)
        ax.set_ylim(bottom=-0.005)

        fig.savefig(filepath_figure)
        plt.close(fig)

    def export_parity_plot(self, filepath):
        """
        Creates and saves predicted vs. measured plots of the test set for the
        best polynomial and kNN models.

        Parameters
        ----------
        filepath : str
            Output figure path.
        """
        models = {self.get_poly_label(): self.gscv_poly, self.get_knn_label(): self.gscv_knn}
        limits = (-0.05, 0.8)

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(2 * FIG_WIDTH, FIG_HEIGHT), dpi=200, layout="constrained")

        for i, (ax, (name, model)) in enumerate(zip(axes, models.items())):
            y_test_pred = model.predict(self.x_test)
            rmse = root_mean_squared_error(self.y_test, y_test_pred)
            r2 = r2_score(self.y_test, y_test_pred)

            ax.plot(limits, limits, color="black", linewidth=1, label="y = x")
            ax.scatter(self.y_test, y_test_pred, s=30, color=COLORS[i], edgecolors="black", linewidth=0.5,
                       label="Test samples")
            ax.text(x=0.03, y=0.97, s=f"RMSE = {rmse:.3f}\nR^2  = {r2:.3f}", transform=ax.transAxes,
                    va="top", ha="left", fontname="monospace", fontsize=FONTSIZE_TEXT + 1)
            ax.set_xlim(limits)
            ax.set_ylim(limits)
            ax.set_title(name, fontsize=FONTSIZE_XY_LABEL)
            sf.format_axes(ax, f"Measured {self.target}", f"Predicted {self.target}")
            ax.legend(loc="lower right", fontsize=FONTSIZE_LEGEND)

        fig.savefig(filepath)
        plt.close(fig)

    def export_poly_coefficients(self, filepath):
        """
        Saves the coefficients of the best polynomial model, sorted from the
        largest to the smallest magnitude.

        Since the features are standardized, the coefficients can be compared
        on the same scale. Each coefficient is the change in growth rate for a
        one standard deviation change in that term.

        Parameters
        ----------
        filepath : str
            Output CSV path.
        """
        model = self.gscv_poly.best_estimator_
        powers = model["polynomialfeatures"].powers_
        coefs = model["linearregression"].coef_
        intercept = model["linearregression"].intercept_

        records = [["Intercept", intercept]]
        for i in range(powers.shape[0]):
            partial_terms = []
            for j in range(powers.shape[1]):
                feature_name = sf.get_feature_name(self.features[j])
                if powers[i, j] == 1:
                    partial_terms.append(f"[{feature_name}]")
                elif powers[i, j] > 1:
                    partial_terms.append(f"[{feature_name}]^{powers[i, j]}")
            records.append([" * ".join(partial_terms), coefs[i]])

        df_export = pd.DataFrame(records, columns=["term", "coefficient"])

        # Keep the intercept first, then sort the other terms by magnitude
        df_terms = df_export.iloc[1:, :]
        order = np.argsort(-np.abs(df_terms.loc[:, "coefficient"].to_numpy()))
        df_export = pd.concat([df_export.iloc[:1, :], df_terms.iloc[order, :]])
        df_export.round(4).to_csv(filepath, index=False)

        print(f"\n{self.get_poly_label()} coefficients (standardized features):")
        print(df_export.round(4).to_string(index=False))

    def export_response_curves(self, filepath):
        """
        Creates and saves the predicted growth rate when one feature is changed
        and the other three are held at their median values.

        Parameters
        ----------
        filepath : str
            Output figure path.
        """
        medians = np.median(self.x_train, axis=0)
        models = {self.get_poly_label(): self.gscv_poly, self.get_knn_label(): self.gscv_knn}

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(2 * FIG_WIDTH, 1.6 * FIG_HEIGHT), dpi=200,
                                 layout="constrained", sharey=True)

        for idx, (ax, feature) in enumerate(zip(np.ravel(axes), self.features)):
            # Every row starts at the median and only the column of interest is changed
            x_plot = np.tile(medians, (N_PLOT, 1))
            x_plot[:, idx] = np.linspace(np.min(self.x[:, idx]), np.max(self.x[:, idx]), N_PLOT)

            ax.scatter(self.x_train[:, idx], self.y_train, s=12, color="grey", alpha=0.3, edgecolors="none",
                       label="Training samples")
            for i, (name, model) in enumerate(models.items()):
                ax.plot(x_plot[:, idx], model.predict(x_plot), color=COLORS[i], linewidth=2, label=name)

            ax.hlines(y=0, xmin=np.min(self.x[:, idx]), xmax=np.max(self.x[:, idx]), color="black", linewidth=0.5)
            sf.format_axes(ax, feature, self.target)

        np.ravel(axes)[0].legend(fontsize=FONTSIZE_LEGEND, loc="lower center")
        fig.savefig(filepath)
        plt.close(fig)

    def export_response_maps(self, filepath_figure, filepath_table):
        """
        Creates and saves maps of the polynomial model's predicted growth rate for
        two pairs of features while the other two features are held at their
        median values. A summary table reports where each map is highest and how
        much of each map is predicted below zero (physically impossible).

        Parameters
        ----------
        filepath_figure : str
            Output figure path.
        filepath_table : str
            Output CSV path.
        """
        medians = np.median(self.x_train, axis=0)
        pairs = [(0, 1), (2, 3)]  # (temperature, pH) and (glucose, lactate)

        cmap = plt.get_cmap("viridis").copy()
        cmap.set_under("lightgrey")  # Predictions below zero are shown in grey

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(2 * FIG_WIDTH, FIG_HEIGHT), dpi=200, layout="constrained")
        records = []

        for ax, (i, j) in zip(axes, pairs):
            xi = np.linspace(np.min(self.x[:, i]), np.max(self.x[:, i]), N_PLOT)
            xj = np.linspace(np.min(self.x[:, j]), np.max(self.x[:, j]), N_PLOT)
            mesh_i, mesh_j = np.meshgrid(xi, xj)

            x_grid = np.tile(medians, (mesh_i.size, 1))
            x_grid[:, i] = np.ravel(mesh_i)
            x_grid[:, j] = np.ravel(mesh_j)
            y_grid = self.gscv_poly.predict(x_grid)

            pcm = ax.pcolormesh(mesh_i, mesh_j, y_grid.reshape(mesh_i.shape), cmap=cmap, vmin=0, vmax=0.7)
            ax.scatter(self.x_train[:, i], self.x_train[:, j], s=8, color="white", edgecolors="black",
                       linewidth=0.3)
            sf.format_axes(ax, self.features[i], self.features[j])

            idx_max = np.argmax(y_grid)
            records.append([
                f"{sf.get_feature_name(self.features[i])} x {sf.get_feature_name(self.features[j])}",
                x_grid[idx_max, i], x_grid[idx_max, j], y_grid[idx_max],
                100 * np.mean(y_grid < 0),
            ])

        cbar = fig.colorbar(pcm, ax=axes, extend="min")
        cbar.set_label(f"Predicted {self.target}", fontsize=FONTSIZE_XY_LABEL)

        fig.savefig(filepath_figure)
        plt.close(fig)

        columns = ["map", "feature_1_at_max", "feature_2_at_max", "max_growth_rate", "percent_below_zero"]
        df_export = pd.DataFrame(records, columns=columns)
        df_export.round(3).to_csv(filepath_table, index=False)

        print(f"\nResponse maps of {self.get_poly_label()} (other features at their median):")
        print(df_export.round(3).to_string(index=False))

    def export_scenarios(self, scenarios, filepath):
        """
        Predicts the growth rate of example operating conditions with the best
        polynomial and kNN models and flags conditions outside the training data.

        Parameters
        ----------
        scenarios : list[tuple]
            Each tuple contains (name, temperature, pH, glucose, lactate).
        filepath : str
            Output CSV path.
        """
        names = [scenario[0] for scenario in scenarios]
        x_scenarios = np.array([scenario[1:] for scenario in scenarios])

        # A condition is inside the data range if every feature is between the
        # minimum and maximum values seen in the training set
        inside_range = np.all((x_scenarios >= np.min(self.x_train, axis=0)) &
                              (x_scenarios <= np.max(self.x_train, axis=0)), axis=1)

        df_export = pd.DataFrame(x_scenarios, columns=self.features)
        df_export.insert(0, "scenario", names)
        df_export["poly_prediction"] = self.gscv_poly.predict(x_scenarios)
        df_export["knn_prediction"] = self.gscv_knn.predict(x_scenarios)
        df_export["inside_data_range"] = inside_range
        df_export.round(3).to_csv(filepath, index=False)

        print("\nScenario predictions:")
        print(df_export.round(3).to_string(index=False))
