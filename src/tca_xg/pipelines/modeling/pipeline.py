from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_data, train_model, evaluate_model


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
                inputs=["X_train", "y_train"],
                outputs=["xgb_model", "accuracy"],
                name="train_model_node"            ),
            node(
                func= evaluate_model,
                inputs=["xgb_model", "X_test", "y_test"],
                outputs=None,
                name="evaluate_model_node"            ),
        ]
    )
