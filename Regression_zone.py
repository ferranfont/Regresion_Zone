from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, time
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import os

load_dotenv()

now_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
fecha = '2023-10-02'
START_DATE = pd.Timestamp(fecha, tz='Europe/Madrid')
END_DATE = pd.Timestamp(fecha, tz='Europe/Madrid')

# === LECTURA DATOS ===
directorio = '../DATA'
nombre_fichero = 'export_es_2015_formatted.csv'
ruta_completa = os.path.join(directorio, nombre_fichero)
df = pd.read_csv(ruta_completa)
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df.set_index('Date', inplace=True)
df.index = df.index.tz_convert('Europe/Madrid')
df = df[(df.index.date >= START_DATE.date()) & (df.index.date <= END_DATE.date())]

# === FILTRA A PARTIR DE UNA HORA DETERMINADA ===
hora_inicio = '07:50:00'
hora_fin = '16:30:00'
hora_inicio = datetime.strptime(hora_inicio, '%H:%M:%S').time()
hora_fin = datetime.strptime(hora_fin, '%H:%M:%S').time()
df_reg = df[(df.index.time >= hora_inicio) & (df.index.time <= hora_fin)].copy()

# --- Prepara los datos para la regresión ---
X = np.arange(len(df_reg)).reshape(-1, 1)
y = df_reg['Close'].values

# --- Regresión lineal ---
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# --- Bandas ±2σ regresión lineal ---
residuals_linear = y - y_pred
sigma_linear = np.std(residuals_linear)
upper_band_linear = y_pred + 2 * sigma_linear
lower_band_linear = y_pred - 2 * sigma_linear

# --- Plot ---
fig = go.Figure()

# Línea de precios original (todo el día)
fig.add_trace(go.Scatter(
    x=df.index,
    y=df['Close'],
    mode='lines',
    name='Close',
    line=dict(color='blue', width=1)
))

# Regresión lineal solo desde las 07:50
fig.add_trace(go.Scatter(
    x=df_reg.index,
    y=y_pred,
    mode='lines',
    name='Regresión lineal (desde 07:50)',
    line=dict(color='red', width=1)
))

# Banda +2σ de la regresión lineal
fig.add_trace(go.Scatter(
    x=df_reg.index,
    y=upper_band_linear,
    mode='lines',
    name='Lineal +2σ',
    line=dict(color='red', width=1, dash='dot')
))
# Banda -2σ de la regresión lineal
fig.add_trace(go.Scatter(
    x=df_reg.index,
    y=lower_band_linear,
    mode='lines',
    name='Lineal -2σ',
    line=dict(color='red', width=1, dash='dot')
))

# Líneas verticales a las 07:50, 15:30, 16:30
for h in ['07:50:00', '15:30:00', '16:30:00']:
    h_time = datetime.strptime(h, '%H:%M:%S').time()
    vline_stamp = pd.Timestamp.combine(START_DATE.date(), h_time).tz_localize('Europe/Madrid')
    fig.add_vline(
        x=vline_stamp.to_pydatetime(),
        line=dict(color='gray', width=1),
        opacity=0.5
    )

fig.update_layout(
    dragmode='pan',
    title=f"Cierre del {fecha} + Regresión lineal y bandas ±2σ",
    xaxis_title='Fecha',
    yaxis_title='Close',
    showlegend=True,
    template='plotly_white',
    xaxis=dict(showgrid=False, linecolor='gray', linewidth=1),
    yaxis=dict(showgrid=False, linecolor='gray', linewidth=1)
)

config = dict(scrollZoom=True)
fig.show(config=config)

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from datetime import time

# Filtra solo los datos desde las 15:00h
hora_inicio = time(15, 0)
df_post1500 = df[df.index.time >= hora_inicio].copy()

# Datos principales
precios = df_post1500['Close'].values
tiempos = df_post1500.index

# Gráfico transpuesto: Precio en X, Tiempo (DatetimeIndex) en Y
plt.figure(figsize=(10, 6))
plt.plot(precios, tiempos, color='blue', linewidth=1, label='Precio vs Tiempo')

# Líneas verticales rojas
for precio in [4692, 4678, 4665]:
    plt.axvline(x=precio, color='red', linestyle='-', linewidth=1.5)

# Campana de Gauss
media = np.mean(precios)
std = np.std(precios)
x_gauss = np.linspace(precios.min(), precios.max(), 300)
y_gauss = norm.pdf(x_gauss, media, std)

# Para escalar la gaussiana al eje Y (tiempo):
# La idea: el punto más alto de la campana = max(tiempos), el más bajo = min(tiempos)
y_gauss_scaled = (y_gauss - y_gauss.min()) / (y_gauss.max() - y_gauss.min())
y_gauss_scaled = y_gauss_scaled * (tiempos.max() - tiempos.min()) + tiempos.min()
plt.plot(x_gauss, y_gauss_scaled, color='grey', linewidth=1, label='Campana de Gauss')

plt.title('Precio (X) vs Tiempo (Y) desde 15:00h + Campana de Gauss')
plt.xlabel('Precio')
plt.ylabel('Fecha/Hora')
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

