from kedro.pipeline import Pipeline, node, pipeline
from .nodes import limpiar_reservas_invalidas, imputar_fechas, convertir_y_filtrar_fechas

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=limpiar_reservas_invalidas,
            inputs="raw_hotel_data",
            outputs="eda_limpio",
            name="limpiar_reservas_invalidas_node",
        ),
        node(
            func=imputar_fechas,
            inputs="eda_limpio",
            outputs="eda_imputado",
            name="imputar_fechas_node",
        ),
        node(
            func=convertir_y_filtrar_fechas,
            inputs="eda_imputado",
            outputs="eda_final",
            name="convertir_y_filtrar_fechas_node",
        ),
    ])
