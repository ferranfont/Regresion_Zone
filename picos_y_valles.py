from scipy import signal
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression

def encontrar_picos_valles(df, prominence=1, distance=5):
    # Usa los índices del df original tal cual, respeta la zona horaria.
    highs = df['High'].values
    picos, _ = signal.find_peaks(highs, prominence=prominence, distance=distance)
    fechas_picos = df.index[picos]
    df_picos = pd.DataFrame({'fecha': fechas_picos, 'precio': highs[picos]})

    lows = df['Low'].values
    valles, _ = signal.find_peaks(-lows, prominence=prominence, distance=distance)
    fechas_valles = df.index[valles]
    df_valles = pd.DataFrame({'fecha': fechas_valles, 'precio': lows[valles]})

    # Calcular media y desviación típica
    mean_picos = np.mean(highs)
    std_picos = np.std(highs)
    mean_valles = np.mean(lows)
    std_valles = np.std(lows)

    df_picos['sigma'] = ((df_picos['precio'] - mean_picos) / std_picos).round(1) if std_picos > 0 else 0
    df_valles['sigma'] = ((df_valles['precio'] - mean_valles) / std_valles).round(1) if std_valles > 0 else 0

    # Calcular regresión para picos
    X_picos = (df_picos['fecha'].astype(np.int64) // 10**9).values.reshape(-1, 1)
    reg_picos = LinearRegression().fit(X_picos, df_picos['precio'].values)
    df_picos['regresion'] = reg_picos.predict(X_picos).round(2)

    # Calcular regresión para valles
    X_valles = (df_valles['fecha'].astype(np.int64) // 10**9).values.reshape(-1, 1)
    reg_valles = LinearRegression().fit(X_valles, df_valles['precio'].values)
    df_valles['regresion'] = reg_valles.predict(X_valles).round(2)

    # Guardar CSVs
    os.makedirs("outputs", exist_ok=True)
    df_picos.to_csv("outputs/picos.csv", index=False, date_format='%Y-%m-%d %H:%M:%S%z')
    df_valles.to_csv("outputs/valles.csv", index=False, date_format='%Y-%m-%d %H:%M:%S%z')

    return df_picos, df_valles
