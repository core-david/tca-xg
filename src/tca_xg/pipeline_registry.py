# """Project pipelines."""
# from __future__ import annotations

# from kedro.framework.project import find_pipelines
# from kedro.pipeline import Pipeline


# def register_pipelines() -> dict[str, Pipeline]:
#     """Register the project's pipelines.

#     Returns:
#         A mapping from pipeline names to ``Pipeline`` objects.
#     """
#     pipelines = find_pipelines()
#     pipelines["__default__"] = sum(pipelines.values())
#     return pipelines



from kedro.pipeline import Pipeline
from tca_xg.pipelines.cleaning import pipeline as dp_pipeline
from tca_xg.pipelines.preprocessing import pipeline as dp_pipeline
from tca_xg.pipelines.modeling import pipeline as dp_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    return {
        "cleaning": dp_pipeline.create_pipeline(),
        "preprocessing": dp_pipeline.create_pipeline(),
        "modeling": dp_pipeline.create_pipeline(),
        "__default__": dp_pipeline.create_pipeline(),
    }