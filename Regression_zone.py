import os
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from sklearn.linear_model import LinearRegression
import picos_y_valles as pv
import chart_reg as chartreg
import plot_matplotlib_transpuesto as pvt
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.stats import norm
import webbrowser

# ===========   LECTURA DE DATOS ==========
fecha = '2025-01-08'  # Fecha a estudiar
directorio = '../DATA'
nombre_fichero = 'export_es_2015_formatted.csv'
ruta_completa = os.path.join(directorio, nombre_fichero)
df = pd.read_csv(ruta_completa)
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df.set_index('Date', inplace=True)
df.index = df.index.tz_convert('Europe/Madrid')
START_DATE = pd.Timestamp(fecha, tz='Europe/Madrid')
END_DATE = pd.Timestamp(fecha, tz='Europe/Madrid')
df = df[(df.index.date >= START_DATE.date()) & (df.index.date <= END_DATE.date())]

# =========== BUSQUEDA DE PICOS Y VALLES ===========
df_picos, df_valles = pv.encontrar_picos_valles(df, prominence=1, top_n=15)

# --- Rango principal para regresión
hora_inicio = '07:50:00'
apertura_mercado = '15:30:00'  # OPEN RANGE MARKET START
hora_fin = '16:30:00'          # OPEN RANGE MARKET
apertura_mercado = datetime.strptime(apertura_mercado, '%H:%M:%S').time()
hora_inicio = datetime.strptime(hora_inicio, '%H:%M:%S').time()
hora_fin = datetime.strptime(hora_fin, '%H:%M:%S').time()
mask_main = (df.index.time >= hora_inicio) & (df.index.time <= hora_fin)

# --- Rango para proyección futura
hora_future = '20:00:00'
hora_future = datetime.strptime(hora_future, '%H:%M:%S').time()
dt_future = pd.Timestamp.combine(START_DATE.date(), hora_future).tz_localize('Europe/Madrid')

# --- Regresión sobre picos
df_picos_rango = df_picos[(df_picos['fecha'].dt.time >= hora_inicio) & (df_picos['fecha'].dt.time <= hora_fin)]
X_picos = (df_picos_rango['fecha'].astype(np.int64) // 10**9).values.reshape(-1, 1)
y_picos = df_picos_rango['precio'].values
reg_picos = LinearRegression().fit(X_picos, y_picos)

# --- Regresión sobre valles
df_valles_rango = df_valles[(df_valles['fecha'].dt.time >= hora_inicio) & (df_valles['fecha'].dt.time <= hora_fin)]
X_valles = (df_valles_rango['fecha'].astype(np.int64) // 10**9).values.reshape(-1, 1)
y_valles = df_valles_rango['precio'].values
reg_valles = LinearRegression().fit(X_valles, y_valles)

# --- Fechas para dibujar la recta sólida y dash
dt_start = pd.Timestamp.combine(START_DATE.date(), hora_inicio).tz_localize('Europe/Madrid')
dt_fin = pd.Timestamp.combine(START_DATE.date(), hora_fin).tz_localize('Europe/Madrid')
dt_proj = dt_future
fechas_solid = pd.date_range(dt_start, dt_fin, freq='min')
fechas_dash = pd.date_range(dt_fin + timedelta(minutes=1), dt_proj, freq='min')

# --- Predicciones
X_solid_picos = (fechas_solid.astype(np.int64) // 10**9).to_numpy().reshape(-1, 1)
y_solid_picos = reg_picos.predict(X_solid_picos)
X_dash_picos = (fechas_dash.astype(np.int64) // 10**9).to_numpy().reshape(-1, 1)
y_dash_picos = reg_picos.predict(X_dash_picos)

X_solid_valles = (fechas_solid.astype(np.int64) // 10**9).to_numpy().reshape(-1, 1)
y_solid_valles = reg_valles.predict(X_solid_valles)
X_dash_valles = (fechas_dash.astype(np.int64) // 10**9).to_numpy().reshape(-1, 1)
y_dash_valles = reg_valles.predict(X_dash_valles)

# --- Filtra datos para dibujar precio desde 07:50 a 20:00
hora_future = time(20, 30)
mask_full = (df.index.time >= hora_inicio) & (df.index.time <= hora_future)
df_plot = df[mask_full]

# --- Picos y valles para scatter sólo en rango 07:50-20:00
dt_start = pd.Timestamp.combine(START_DATE.date(), hora_inicio).tz_localize('Europe/Madrid')
mask_picos_plot = (df_picos['fecha'] >= dt_start) & (df_picos['fecha'] <= dt_proj)
mask_valles_plot = (df_valles['fecha'] >= dt_start) & (df_valles['fecha'] <= dt_proj)

# =================== PLOTLY GRÁFICO INTERACTIVO (desde 15:30) ===================
html_path = chartreg.plotly_regresion_chart(
    df_plot, df_picos, df_valles,
    fechas_solid, y_solid_picos, y_solid_valles,
    fechas_dash, y_dash_picos, y_dash_valles,
    mask_picos_plot, mask_valles_plot,
    apertura_mercado, hora_fin,
    START_DATE
)
webbrowser.open('file://' + os.path.realpath(html_path))

# =================== MATPLOTLIB TRANSPUESTO (desde 15:00) ===================
hora_inicio_plt = '15:00:00'
hora_inicio_plt = datetime.strptime(hora_inicio_plt, '%H:%M:%S').time()
dt_start_plt = pd.Timestamp.combine(START_DATE.date(), hora_inicio_plt).tz_localize('Europe/Madrid')

# Filtra solo los datos y regresiones desde la hora de inicio
mask_post_plt = df.index.time >= hora_inicio_plt
mask_solid_plt = fechas_solid >= dt_start_plt
mask_dash_plt  = fechas_dash  >= dt_start_plt
precios = df.loc[mask_post_plt, 'Close'].values
tiempos = df.index[mask_post_plt]

pvt.plot_matplotlib_transpuesto(
    precios,
    tiempos,
    fechas_solid[mask_solid_plt], y_solid_picos[mask_solid_plt], y_solid_valles[mask_solid_plt],
    fechas_dash[mask_dash_plt], y_dash_picos[mask_dash_plt], y_dash_valles[mask_dash_plt],
    df_picos, df_valles
)

