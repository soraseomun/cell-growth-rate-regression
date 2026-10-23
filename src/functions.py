# =======================================================================
# FUNCTIONS
# Define reusable helper functions in this file.
# =======================================================================

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from src.constants import *


def make_poly_model(degree=2):
    """
    Creates a polynomial regression pipeline.

    The features are standardized before the polynomial terms are created
    so that every term has a similar scale.

    Parameters
    ----------
    degree : int
        Degree of the polynomial.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted polynomial regression model.
    """
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression()
    )


def make_knn_model(n_neighbors=5, weights="uniform"):
    """
    Creates a k-nearest neighbors (kNN) regression pipeline.

    The features are standardized because kNN uses distances, and the
    distances would otherwise be dominated by the feature with the
    largest values (e.g., glucose in mM compared to pH).

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors (k).
    weights : str
        "uniform" or "distance".

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted kNN regression model.
    """
    return make_pipeline(
        StandardScaler(),
        KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
    )


def get_feature_name(feature):
    """
    Removes the units from a column name (e.g., "Glucose [mM]" -> "Glucose").
    """
    feature_unit_splitter = " ["
    if feature_unit_splitter in feature:
        return feature.split(feature_unit_splitter)[0]
    return feature


def evaluate_model(model, x_train, y_train, x_test, y_test, cv):
    """
    Evaluates a model with cross-validation on the training set and then
    on the test set.

    Parameters
    ----------
    model : sklearn estimator
        Unfitted model.
    x_train, y_train : np.ndarray
        Training data.
    x_test, y_test : np.ndarray
        Test data.
    cv : int
        Number of cross-validation folds.

    Returns
    -------
    dict
        Mean and standard deviation of the cross-validation metrics and the
        test set metrics.
    """
    cv_results = cross_validate(model, x_train, y_train, cv=cv,
                                scoring=("neg_root_mean_squared_error", "r2"),
                                return_train_score=True)

    ms_train_rmse = -cv_results["train_neg_root_mean_squared_error"]
    ms_valid_rmse = -cv_results["test_neg_root_mean_squared_error"]
    ms_valid_r2 = cv_results["test_r2"]

    # We need to fit the model since it was only fit inside cross_validate()
    model.fit(x_train, y_train)
    y_test_pred = model.predict(x_test)

    return dict(
        train_rmse=np.mean(ms_train_rmse),
        valid_rmse_mean=np.mean(ms_valid_rmse),
        valid_rmse_std=np.std(ms_valid_rmse),
        valid_r2_mean=np.mean(ms_valid_r2),
        test_rmse=root_mean_squared_error(y_test, y_test_pred),
        test_r2=r2_score(y_test, y_test_pred),
    )


def format_axes(ax, xlabel, ylabel):
    """
    Applies the same labels and font sizes to an axes.
    """
    ax.set_xlabel(xlabel, fontsize=FONTSIZE_XY_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE_XY_LABEL)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE_TICKS)
