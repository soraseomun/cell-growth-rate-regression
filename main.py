# =====================================================================
# MAIN PROGRAM
# Predicts the specific growth rate of cells from temperature, pH,
# glucose, and lactate using polynomial and kNN regression.
# =====================================================================

import os

from src.classes import GrowthRateStudy
from src.constants import *

study = GrowthRateStudy(filepath=DATASET_PATH, test_size=TEST_SIZE, random_state=RANDOM_STATE, cv=CV)

# 1. Look at the data
study.export_data_overview(filepath=os.path.join("figures", "figure_01_data_overview.png"))

# 2. Find the best hyperparameters of each model
study.tune_polynomial(degrees=POLY_DEGREES)
study.tune_knn(n_neighbors=KNN_N_NEIGHBORS, weights=KNN_WEIGHTS)

study.export_grid_search_poly(filepath=os.path.join("tables", "table_01_grid_search_poly.csv"))
study.export_grid_search_knn(filepath=os.path.join("tables", "table_02_grid_search_knn.csv"))
study.export_validation_curve_poly(filepath=os.path.join("figures", "figure_02_validation_curve_poly.png"))
study.export_validation_curve_knn(filepath=os.path.join("figures", "figure_03_validation_curve_knn.png"))

# 3. Compare the best models with reasonable alternatives
study.export_model_comparison(filepath=os.path.join("tables", "table_03_model_comparison.csv"))

# 4. Check how the models change with the amount of data
study.export_learning_curve(n_samples_list=LEARNING_CURVE_N_SAMPLES, seed=LEARNING_CURVE_SEED,
                            filepath_table=os.path.join("tables", "table_04_learning_curve.csv"),
                            filepath_figure=os.path.join("figures", "figure_04_learning_curve.png"))

# 5. Check the predictions on the test set
study.export_parity_plot(filepath=os.path.join("figures", "figure_05_parity_test_set.png"))

# 6. Interpret how each feature affects the growth rate
study.export_poly_coefficients(filepath=os.path.join("tables", "table_05_poly_coefficients.csv"))
study.export_response_curves(filepath=os.path.join("figures", "figure_06_response_curves.png"))
study.export_response_maps(filepath_figure=os.path.join("figures", "figure_07_response_maps.png"),
                           filepath_table=os.path.join("tables", "table_06_response_maps.csv"))

# 7. Predict the growth rate of example operating conditions
study.export_scenarios(scenarios=SCENARIOS, filepath=os.path.join("tables", "table_07_scenarios.csv"))
