import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
import logging
from sklearn.metrics import r2_score, mean_absolute_error, max_error, mean_squared_error
from typing import Dict, Tuple, Any


def split_data(daily_data: pd.DataFrame, encoded_data: pd.DataFrame) -> tuple:
    # Preparar features (X) y target (y)
    X = pd.concat([
        daily_data[['mes','dia_semana','dia']],
        encoded_data
    ], axis=1)
    y = daily_data['ocupacion_total']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series ) -> Tuple[xgb.XGBRegressor, float]:
    """Trains the XGBoost regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.
        X_test: Optional test data for evaluation during training.
        y_test: Optional test target data for evaluation during training.

    Returns:
        Tuple containing:
        - Trained XGBoost model
        - Training accuracy (R² score)
        - Evaluation results dictionary
    """
    # eval_set = [(X_train, y_train), (X_test, y_test)]

    # Initialize XGBoost regressor
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        # eval_metric='rmse'
    )
    
    # Train the model
    xgb_model.fit(
        X_train, y_train,
        # eval_set=eval_set,
        verbose=False
    )
    
    # Calculate training accuracy (R² score)
    accuracy = xgb_model.score(X_train, y_train)
    
    # # Get evaluation results
    # results = xgb_model.evals_result()
    
    return xgb_model, accuracy



def evaluate_model(
    xgb_model: xgb.XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, float]:
    """Calculates and logs evaluation metrics for XGBoost model.

    Args:
        regressor: Trained XGBoost model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.

    Returns:
        Dictionary containing evaluation metrics.
    """
    # Make predictions
    y_pred = xgb_model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    me = max_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # Log the main metric
    logger = logging.getLogger(__name__)
    logger.info("XGBoost model has a coefficient R^2 of %.3f on test data.", r2)
    logger.info("Additional metrics - MAE: %.3f, RMSE: %.3f, Max Error: %.3f", mae, rmse, me)
    
    return {
        "r2_score": r2,
        "mae": mae,
        "max_error": me,
        "mse": mse,
        "rmse": rmse
    }

