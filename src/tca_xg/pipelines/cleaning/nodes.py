import pandas as pd
from typing import Tuple

def limpiar_reservas_invalidas(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df.numero_personas > 0]
    df = df[~(
        (df.numero_adultos + df.numero_menores == 0) & 
        (df.cliente_disponible > 0)
    )]
    df = df.drop(columns=["reservacion"], errors="ignore")
    return df

def imputar_fechas(df: pd.DataFrame) -> pd.DataFrame:
    def llenar_nans_fechas(row):
        if pd.isna(row["fecha_llegada"]):
            if pd.isna(row["fecha_salida"]):
                row["fecha_salida"] = row["fecha_confirmacion_pickup"]
            row["fecha_llegada"] = row["fecha_salida"] - row["cantidad_noches"]
        if pd.isna(row["fecha_salida"]):
            row["fecha_salida"] = row["fecha_llegada"] + row["cantidad_noches"]
        if pd.isna(row["fecha_registro"]):
            if row["estatus_de_la_reservacion"] in ["RESERVACION CANCELADA", "NO SHOW"]:
                row["fecha_registro"] = row["fecha_reservacion"]
            elif row["estatus_de_la_reservacion"] in ["ROOMING LIST", "RESERVACION O (R)REGISTRO"]:
                row["fecha_registro"] = row["fecha_confirmacion_pickup"]
        return row

    df = df.apply(llenar_nans_fechas, axis=1)
    df[["fecha_llegada", "fecha_registro", "fecha_salida"]] = df[["fecha_llegada", "fecha_registro", "fecha_salida"]].astype("int64")
    return df

def convertir_y_filtrar_fechas(df: pd.DataFrame) -> pd.DataFrame:
    for column in ["fecha_reservacion", "fecha_llegada", "fecha_registro", "fecha_salida", "fecha_confirmacion_pickup"]:
        df[column] = pd.to_datetime(df[column].astype(str), format="%Y%m%d")
    ultima_fecha = df["fecha_reservacion"].max()
    df = df[df["fecha_salida"] <= ultima_fecha]
    return df
