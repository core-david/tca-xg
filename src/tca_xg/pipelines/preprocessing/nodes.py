import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def daily_hotel_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        df: Raw hotel reservation data
        
    Returns:
        Preprocessed daily data
    """
    
    # Convert date column
    df['fecha_llegada'] = pd.to_datetime(df['fecha_llegada'])
    # df['agencia'] = df['agencia'].str.strip()
    # df['canal'] = df['agencia'].str.strip()
    # df['segmento_alterno'] = df['agencia'].str.strip()
    # df['tipo_habitacion_nombre'] = df['agencia'].str.strip()
    
    # 2 Expand each reservation into daily rows
    expanded_rows = []
    
    for _, row in df.iterrows():
        for i in range(row['cantidad_noches']):
            dia = row['fecha_llegada'] + pd.Timedelta(days=i)
            expanded_rows.append({
                'fecha': dia,
                'habitaciones': row['habitaciones'],
                'numero_personas': row['numero_personas'],
                'numero_adultos': row['numero_adultos'],
                'numero_menores': row['numero_menores'],
                'agencia': row['agencia'],
                'canal': row['canal'],
                'segmento_alterno': row['segmento_alterno'],
                'tipo_habitacion_nombre': row['tipo_habitacion_nombre']
            })
    
    df_expanded = pd.DataFrame(expanded_rows)
    
    # 3 Aggregate by date
    df_daily = df_expanded.groupby('fecha').agg({
        'habitaciones': 'sum',
        'numero_personas': 'mean',
        'numero_adultos': 'mean',
        'numero_menores': 'mean',
        'agencia': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
        'canal': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
        'segmento_alterno': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
        'tipo_habitacion_nombre': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
    }).reset_index().rename(columns={'fecha': 'ds'})

    # 3 Crear target: ocupación total (habitaciones sumadas)
    df_daily['ocupacion_total'] = df_daily['habitaciones']
    df_daily = df_daily[(df_daily["ds"]< "2020-04-30 00:00:00") & (df_daily["ds"] > "2019-02-12 00:00:00")]
    # Variables temporales enriquecidas
    df_daily['dia_semana'] = df_daily['ds'].dt.weekday
    df_daily['mes'] = df_daily['ds'].dt.month
    df_daily['anio'] = df_daily['ds'].dt.year
    df_daily['dia'] = df_daily['ds'].dt.day

    # Ratios internos
    df_daily['personas_por_habitacion'] = df_daily['numero_personas'] / (df_daily['habitaciones'] + 1)
    df_daily['adultos_por_menor'] = df_daily['numero_adultos'] / (df_daily['numero_menores'] + 1)

    return df_daily

def encoded_hotel_data(df_daily: pd.DataFrame) -> pd.DataFrame:

    # Codificar variables categóricas
    cat_cols = ['agencia', 'canal', 'segmento_alterno', 'tipo_habitacion_nombre']
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded = encoder.fit_transform(df_daily[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols), index=df_daily.index)

    return encoded_df
