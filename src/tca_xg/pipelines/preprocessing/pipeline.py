from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    daily_hotel_data,
    encoded_hotel_data,
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
          node(
                func=daily_hotel_data,
                inputs="eda_final",
                outputs="daily_data",
                name="daily_hotel_data_node"
            ),
            node(
                func=encoded_hotel_data,
                inputs="daily_data",
                outputs="encoded_data",
                name="encoded_hotel_data_node"
        ),
        ]
    )
