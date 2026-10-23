# =======================================================================
# CONSTANTS
# Define constant values that do not change throughout your program.
# =======================================================================

import os

# Location of the dataset
DATASET_PATH = os.path.join("datasets", "dataset_cell_growth_4factor.csv")

# Settings used to split the data and perform cross-validation
TEST_SIZE = 0.2
RANDOM_STATE = 0
CV = 5

# Hyperparameter values that will be tried in GridSearchCV
POLY_DEGREES = [i for i in range(1, 7)]
KNN_N_NEIGHBORS = [i for i in range(1, 51)]
KNN_WEIGHTS = ["uniform", "distance"]

# Number of samples used at each point of the learning curve
LEARNING_CURVE_N_SAMPLES = [40, 80, 120, 160, 200, 240, 280, 318]
LEARNING_CURVE_SEED = 1

# Example operating conditions used to test the final models.
# Each tuple: (name, temperature [C], pH, glucose [mM], lactate [mM])
SCENARIOS = [
    ("Fresh medium", 36.5, 7.1, 30.0, 2.0),
    ("Typical culture", 37.0, 7.0, 15.0, 15.0),
    ("Glucose depletion", 37.0, 7.0, 1.0, 15.0),
    ("Acidic pH and high lactate", 37.0, 6.4, 10.0, 30.0),
    ("Alkaline pH", 37.0, 7.7, 15.0, 15.0),
    ("Low temperature", 32.0, 7.0, 15.0, 15.0),
    ("High temperature", 41.5, 7.0, 15.0, 15.0),
    ("Outside data range (45 C)", 45.0, 7.0, 15.0, 15.0),
    ("Outside data range (60 mM glucose)", 37.0, 7.0, 60.0, 0.0),
]

# Number of points used to draw model curves and maps
N_PLOT = 100

# Plotting constants
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
MARKERS = ["o", "D", "s", "^"]
FIG_WIDTH = 14.39 / 2.54
FIG_HEIGHT = 12.09 / 2.54
FONTSIZE_XY_LABEL = 14
FONTSIZE_TICKS = 12
FONTSIZE_LEGEND = 10
FONTSIZE_TEXT = 9
