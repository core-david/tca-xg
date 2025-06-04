import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
import logging
from sklearn.metrics import r2_score, mean_absolute_error, max_error, mean_squared_error
from typing import Dict, Tuple, Any
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import io
from sklearn.metrics import r2_score


def split_data(daily_data: pd.DataFrame, encoded_data: pd.DataFrame) -> tuple:
    # Preparar features (X) y target (y)
    X = pd.concat([
        daily_data[['mes','dia_semana','dia']],
        encoded_data
    ], axis=1)
    y = daily_data['ocupacion_total']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series, xgb_params: Dict[str, Any] ) -> xgb.XGBRegressor:
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
    # xgb_model = xgb.XGBRegressor(
    #     objective='reg:squarederror',
    #     random_state=42,
    #     n_estimators=200,
    #     learning_rate=0.1,
    #     max_depth=4,
    #     subsample=0.8,
    #     colsample_bytree=0.8,
    #     # eval_metric='rmse'
    # )
    xgb_model = xgb.XGBRegressor(**xgb_params)
    
    # Train the model
    xgb_model.fit(
        X_train, y_train,
        # eval_set=eval_set,
        verbose=False
    )
    
    return xgb_model



def evaluate_model(
    xgb_model: xgb.XGBRegressor, 
    X_test: pd.DataFrame, 
    y_test: pd.Series
) -> Tuple[float, float, float, float, float]:
    """Calculates and logs evaluation metrics for XGBoost model.
    
    Args:
        xgb_model: Trained XGBoost model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    
    Returns:
        Tuple containing evaluation metrics (r2, mae, max_error, mse, rmse).
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
    
    return r2, mae, me, mse, rmse


def create_model_evaluation_plot(
    xgb_model: Any, 
    X_test: pd.DataFrame, 
    y_test: pd.Series
) -> Figure:
    """Creates a model evaluation plot showing actual vs predicted values.
    
    Args:
        xgb_model: Trained XGBoost model.
        X_test: Test features.
        y_test: Test target values.
        
    Returns:
        Matplotlib Figure object for MatplotlibWriter.
    """
    # Make predictions
    y_pred = xgb_model.predict(X_test)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('XGBoost Model Evaluation', fontsize=16, fontweight='bold')
    
    # 1. Actual vs Predicted scatter plot
    axes[0, 0].scatter(y_test, y_pred, alpha=0.6, color='blue')
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title('Actual vs Predicted Values')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add R² score to the plot
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, y_pred)
    axes[0, 0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[0, 0].transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Residuals plot
    residuals = y_test - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='green')
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Residuals histogram
    axes[1, 0].hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Residuals')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Feature importance (top 10)
    if hasattr(xgb_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        axes[1, 1].barh(feature_importance['feature'], feature_importance['importance'], color='purple')
        axes[1, 1].set_xlabel('Importance')
        axes[1, 1].set_title('Top 10 Feature Importances')
        axes[1, 1].invert_yaxis()
    else:
        axes[1, 1].text(0.5, 0.5, 'Feature importance\nnot available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Feature Importance')
    
    # Adjust layout
    plt.tight_layout()
    
    # Return the figure object (DO NOT close it here)
    return fig