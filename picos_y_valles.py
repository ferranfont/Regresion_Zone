import numpy as np
import pandas as pd
from scipy import signal

def encontrar_picos_valles(df, prominence=1, top_n=15):
    """
    Encuentra los picos (usando 'High') y valles (usando 'Low') más prominentes de un DataFrame OHLC.
    
    Args:
        df (pd.DataFrame): DataFrame con columnas 'High' y 'Low', y el índice como fechas.
        prominence (float): Prominencia mínima para considerar un pico/valle.
        top_n (int): Número de picos/valles más prominentes a devolver.
        
    Returns:
        df_picos (pd.DataFrame): DataFrame con fecha y precio de los picos ('High').
        df_valles (pd.DataFrame): DataFrame con fecha y precio de los valles ('Low').
    """
    # --- Picos usando High ---
    highs = df['High'].values
    picos, props_picos = signal.find_peaks(highs, prominence=prominence)
    if len(picos) > 0:
        top_picos = picos[np.argsort(props_picos['prominences'])[-top_n:]]
        fechas_picos = df.index[top_picos]
        valores_picos = highs[top_picos]
        df_picos = pd.DataFrame({'fecha': fechas_picos, 'precio': valores_picos}).sort_values('fecha').reset_index(drop=True)
    else:
        df_picos = pd.DataFrame(columns=['fecha', 'precio'])

    # --- Valles usando Low ---
    lows = df['Low'].values
    valles, props_valles = signal.find_peaks(-lows, prominence=prominence)
    if len(valles) > 0:
        top_valles = valles[np.argsort(props_valles['prominences'])[-top_n:]]
        fechas_valles = df.index[top_valles]
        valores_valles = lows[top_valles]
        df_valles = pd.DataFrame({'fecha': fechas_valles, 'precio': valores_valles}).sort_values('fecha').reset_index(drop=True)
    else:
        df_valles = pd.DataFrame(columns=['fecha', 'precio'])

    return df_picos, df_valles
