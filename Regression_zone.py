import os
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from sklearn.linear_model import LinearRegression
import picos_y_valles as pv
import plot_matplotlib_transpuesto as pvt
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.stats import norm
import webbrowser

# === LECTURA DATOS ===
directorio = '../DATA'
nombre_fichero = 'export_es_2015_formatted.csv'
ruta_completa = os.path.join(directorio, nombre_fichero)
df = pd.read_csv(ruta_completa)
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df.set_index('Date', inplace=True)
df.index = df.index.tz_convert('Europe/Madrid')
fecha = '2023-10-04'
START_DATE = pd.Timestamp(fecha, tz='Europe/Madrid')
END_DATE = pd.Timestamp(fecha, tz='Europe/Madrid')
df = df[(df.index.date >= START_DATE.date()) & (df.index.date <= END_DATE.date())]

# === Picos y Valles ===
df_picos, df_valles = pv.encontrar_picos_valles(df, prominence=1, top_n=15)

# --- Rango principal para regresión
hora_inicio = '07:50:00'
apertura_mercado = '15:30:00'
hora_fin = '16:30:00'
apertura_mercado = datetime.strptime(apertura_mercado, '%H:%M:%S').time()
hora_inicio = datetime.strptime(hora_inicio, '%H:%M:%S').time()
hora_fin = datetime.strptime(hora_fin, '%H:%M:%S').time()
mask_main = (df.index.time >= hora_inicio) & (df.index.time <= hora_fin)

# --- Rango para proyección futura
hora_future = time(20, 0)
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
hora_future = time(20, 0)
mask_full = (df.index.time >= hora_inicio) & (df.index.time <= hora_future)
df_plot = df[mask_full]

# --- Picos y valles para scatter sólo en rango 07:50-20:00
dt_start = pd.Timestamp.combine(START_DATE.date(), hora_inicio).tz_localize('Europe/Madrid')
mask_picos_plot = (df_picos['fecha'] >= dt_start) & (df_picos['fecha'] <= dt_proj)
mask_valles_plot = (df_valles['fecha'] >= dt_start) & (df_valles['fecha'] <= dt_proj)

# =================== PLOTLY: EXPORTAR HTML Y ABRIR EN CHROME ===================

# =================== PLOTLY: EXPORTAR HTML Y ABRIR EN CHROME ===================

chart_dir = 'charts'
os.makedirs(chart_dir, exist_ok=True)
html_path = os.path.join(chart_dir, 'chart_picos_valles_regresion.html')

fig = go.Figure()

# --- PRECIO
fig.add_trace(go.Scatter(
    x=df_plot.index, y=df_plot['Close'],
    mode='lines', name='Close', line=dict(color='blue', width=1)
))

# --- PICOS Y VALLES
fig.add_trace(go.Scatter(
    x=df_picos['fecha'][mask_picos_plot], y=df_picos['precio'][mask_picos_plot],
    mode='markers', name='Picos (High)', marker=dict(color='green', size=8, symbol='circle')
))
fig.add_trace(go.Scatter(
    x=df_valles['fecha'][mask_valles_plot], y=df_valles['precio'][mask_valles_plot],
    mode='markers', name='Valles (Low)', marker=dict(color='red', size=8, symbol='circle')
))

# --- LÍNEAS DE REGRESIÓN
fig.add_trace(go.Scatter(
    x=fechas_solid, y=y_solid_picos, mode='lines',
    name='Recta Picos', line=dict(color='green', width=1)
))
fig.add_trace(go.Scatter(
    x=fechas_dash, y=y_dash_picos, mode='lines',
    name='Recta Picos Extensión', line=dict(color='green', width=1, dash='dash')
))
fig.add_trace(go.Scatter(
    x=fechas_solid, y=y_solid_valles, mode='lines',
    name='Recta Valles', line=dict(color='red', width=1)
))
fig.add_trace(go.Scatter(
    x=fechas_dash, y=y_dash_valles, mode='lines',
    name='Recta Valles Extensión', line=dict(color='red', width=1, dash='dash')
))



# --- RELLENO ENTRE REGRESIONES (solo entre 15:30 y 16:30)
from datetime import time

hora_rango_1 = apertura_mercado
hora_rango_2 = hora_fin
dt_rango_1 = pd.Timestamp.combine(START_DATE.date(), hora_rango_1).tz_localize('Europe/Madrid')
dt_rango_2 = pd.Timestamp.combine(START_DATE.date(), hora_rango_2).tz_localize('Europe/Madrid')

idx_1 = np.argmin(np.abs(fechas_solid - dt_rango_1))
idx_2 = np.argmin(np.abs(fechas_solid - dt_rango_2))

x_polygon = [
    fechas_solid[idx_1],
    fechas_solid[idx_2],
    fechas_solid[idx_2],
    fechas_solid[idx_1],
    fechas_solid[idx_1]  # Cierra el polígono
]
y_polygon = [
    y_solid_valles[idx_1],
    y_solid_valles[idx_2],
    y_solid_picos[idx_2],
    y_solid_picos[idx_1],
    y_solid_valles[idx_1]
]

fig.add_trace(go.Scatter(
    x=x_polygon,
    y=y_polygon,
    fill='toself',
    fillcolor='rgba(135, 206, 250, 0.2)',  # Color translúcido azul celeste
    line=dict(color='rgba(0,0,0,0)'),
    hoverinfo='skip',
    showlegend=False,
    name='Zona entre regresiones'
))

# === RELLENO VERDE PASTEL CUANDO PRECIO > REGRESIÓN VERDE DASH ===

# Crea un DataFrame con los datos de la regresión y del precio para las fechas dash
df_dash = pd.DataFrame({
    'fecha': fechas_dash,
    'regresion': y_dash_picos,
})
df_dash.set_index('fecha', inplace=True)

# Filtra los precios para las fechas dash
df_price_dash = df_plot[df_plot.index.isin(fechas_dash)]
df_merged = df_dash.join(df_price_dash[['Close']], how='inner')

# Filtra solo donde el precio está por encima de la regresión
df_above = df_merged[df_merged['Close'] > df_merged['regresion']]

if not df_above.empty:
    x_fill = list(df_above.index) + list(df_above.index[::-1])
    y_fill = list(df_above['Close']) + list(df_above['regresion'][::-1])

    fig.add_trace(go.Scatter(
        x=x_fill,
        y=y_fill,
        fill='toself',
        fillcolor='rgba(152, 251, 152, 0.3)',  # PaleGreen translúcido
        line=dict(color='rgba(0,0,0,0)'),
        hoverinfo='skip',
        name='Precio > Regresión Picos',
        showlegend=False
    ))

# === RELLENO ROJO PASTEL CUANDO PRECIO < REGRESIÓN VALLES DASH ===

# Crea un DataFrame con los datos de la regresión y del precio para las fechas dash
df_dash_valles = pd.DataFrame({
    'fecha': fechas_dash,
    'regresion': y_dash_valles,
})
df_dash_valles.set_index('fecha', inplace=True)

# Filtra los precios para las fechas dash
df_price_dash = df_plot[df_plot.index.isin(fechas_dash)]
df_merged_valles = df_dash_valles.join(df_price_dash[['Close']], how='inner')

# Filtra solo donde el precio está por debajo de la regresión
df_below = df_merged_valles[df_merged_valles['Close'] < df_merged_valles['regresion']]

if not df_below.empty:
    x_fill_below = list(df_below.index) + list(df_below.index[::-1])
    y_fill_below = list(df_below['Close']) + list(df_below['regresion'][::-1])

    fig.add_trace(go.Scatter(
        x=x_fill_below,
        y=y_fill_below,
        fill='toself',
        fillcolor='rgba(255, 160, 160, 0.3)',  # Pastel red (light coral-ish)
        line=dict(color='rgba(0,0,0,0)'),
        hoverinfo='skip',
        name='Precio < Regresión Valles',
        showlegend=False
    ))

# --- SIGMA CON EXCESOS DE PICOS Y VALLES

outputs_dir = 'outputs'
os.makedirs(outputs_dir, exist_ok=True)

# --- ZONA PICO: Precio por encima de la verde ---
df_above = df_merged[df_merged['Close'] > df_merged['regresion']].copy()
df_above['Exceso'] = df_above['Close'] - df_above['regresion']
sigma_pico = df_above['Exceso'].std()
df_above['Sigma (desviación estándar)'] = df_above['Exceso'] / sigma_pico
df_above['Tag'] = 'pico'

# Guarda CSV y HTML
csv_pico_path = os.path.join(outputs_dir, 'dispersión_pico_sobre_verde.csv')
html_pico_path = os.path.join(outputs_dir, 'dispersión_pico_sobre_verde.html')
df_above.to_csv(csv_pico_path)
df_above.to_html(html_pico_path)

print(f'✅ Guardado: {csv_pico_path}')
print(f'✅ Guardado: {html_pico_path}')

# --- ZONA VALLE: Precio por debajo de la roja ---
df_below = df_merged_valles[df_merged_valles['Close'] < df_merged_valles['regresion']].copy()
df_below['Exceso'] = df_below['Close'] - df_below['regresion']
sigma_valle = df_below['Exceso'].std()
df_below['Sigma (desviación estándar)'] = df_below['Exceso'] / sigma_valle
df_below['Tag'] = 'valle'

# Guarda CSV y HTML
csv_valle_path = os.path.join(outputs_dir, 'dispersión_valle_bajo_roja.csv')
html_valle_path = os.path.join(outputs_dir, 'dispersión_valle_bajo_roja.html')
df_below.to_csv(csv_valle_path)
df_below.to_html(html_valle_path)

print(f'✅ Guardado: {csv_valle_path}')
print(f'✅ Guardado: {html_valle_path}')

# --- OPCIONAL: Une ambos y guarda una sola tabla si lo prefieres ---
df_union = pd.concat([df_above, df_below]).sort_index()
csv_union_path = os.path.join(outputs_dir, 'dispersión_picos_valles.csv')
html_union_path = os.path.join(outputs_dir, 'dispersión_picos_valles.html')
df_union.to_csv(csv_union_path)
df_union.to_html(html_union_path)
print(f'✅ Guardado conjunto: {csv_union_path}')
print(f'✅ Guardado conjunto: {html_union_path}')


# --- LÍNEAS VERTICALES HORARIAS
for h in ['07:50:00', '15:30:00', '16:30:00', '20:00:00']:
    vline_time = datetime.strptime(h, '%H:%M:%S').time()
    vline_stamp = pd.Timestamp.combine(START_DATE.date(), vline_time).tz_localize('Europe/Madrid')
    fig.add_vline(
        x=vline_stamp, line=dict(color='gray', width=1), opacity=0.5
    )

# --- LAYOUT Y EXPORTACIÓN
fig.update_layout(
    dragmode='pan', 
    title=f'Regresiones – {START_DATE.date().strftime("%Y-%m-%d")}',
    xaxis_title='Fecha',
    yaxis_title='Precio',
    showlegend=False,
    template='plotly_white',
    xaxis=dict(showgrid=False, linecolor='gray', linewidth=1),
    yaxis=dict(showgrid=False, linecolor='gray', linewidth=1)
)

fig.write_html(html_path, config={"scrollZoom": True})
print(f"✅ Gráfico Plotly guardado como '{html_path}'")
webbrowser.open('file://' + os.path.realpath(html_path))


# =================== MATPLOTLIB TRANSPUESTO (desde 15:00) ===================

hora_inicio_plt = time(15, 0)
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

