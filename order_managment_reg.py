import pandas as pd
import os
from sklearn.linear_model import LinearRegression
import numpy as np

def order_management_reg(
    df, df_picos, df_valles,
    reg_picos, reg_valles,
    hora_fin,
    outlier_sigma=0.1,
    stop_points=10,
    target_points=10
):
    trades = []

    # Solo consideramos picos y valles después de la hora de inicio de trading (hora_fin)
    df_picos_post = df_picos[df_picos['fecha'].dt.time >= hora_fin]
    df_valles_post = df_valles[df_valles['fecha'].dt.time >= hora_fin]

    # Procesar picos (ventas)
    for _, fila in df_picos_post.iterrows():
        fecha = fila['fecha']
        timestamp = int(fecha.timestamp())
        precio_real = fila['precio']
        sigma = fila.get('sigma', None)

        if sigma is None:
            continue

        precio_estimado = reg_picos.predict(np.array([[timestamp]]))[0]

        if sigma > outlier_sigma and precio_real > precio_estimado:
            trades.append({
                'tipo': 'venta',
                'fecha_entrada': fecha,
                'precio_entrada': precio_real,
                'regresion': precio_estimado,
                'diferencia': precio_real - precio_estimado,
                'sigma': sigma,
                'stop_points': stop_points,
                'target_points': target_points
            })

    # Procesar valles (compras)
    for _, fila in df_valles_post.iterrows():
        fecha = fila['fecha']
        timestamp = int(fecha.timestamp())
        precio_real = fila['precio']
        sigma = fila.get('sigma', None)

        if sigma is None:
            continue

        precio_estimado = reg_valles.predict(np.array([[timestamp]]))[0]

        if sigma < -outlier_sigma and precio_real < precio_estimado:
            trades.append({
                'tipo': 'compra',
                'fecha_entrada': fecha,
                'precio_entrada': precio_real,
                'regresion': precio_estimado,
                'diferencia': precio_real - precio_estimado,
                'sigma': sigma,
                'stop_points': stop_points,
                'target_points': target_points
            })

    df_trades = pd.DataFrame(trades)

    # Guardar CSV
    os.makedirs("outputs", exist_ok=True)
    if not df_trades.empty:
        fecha_str = df_trades['fecha_entrada'].dt.strftime('%Y-%m-%d').iloc[0]
        output_path = f"outputs/trades_{fecha_str}.csv"
        df_trades.to_csv(output_path, index=False)
        print(f"✅ Archivo guardado como '{output_path}'")
    else:
        print("⚠️ No se generaron operaciones. CSV no creado.")

    return df_trades
