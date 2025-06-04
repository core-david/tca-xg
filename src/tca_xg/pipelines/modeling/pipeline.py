from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_data, train_model, evaluate_model, create_model_evaluation_plot


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["daily_data", "encoded_data"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node"
            ),
            node(
                func= train_model, 
                inputs=["X_train", "y_train", "params:xgb_model"],
                outputs="xgb_model",
                name="train_model_node"            ),
            node(
                func=evaluate_model,
                inputs=["xgb_model", "X_test", "y_test"],
                outputs=["eval_r2_score", "eval_mae", "eval_max_error", "eval_mse", "eval_rmse"],
                name="evaluate_model_node",
            ),
            node(
                func=create_model_evaluation_plot,
                inputs=["xgb_model", "X_test", "y_test"],
                outputs="evaluation_plot",
                name="create_evaluation_plot_node",
            ),
        ]
    )
