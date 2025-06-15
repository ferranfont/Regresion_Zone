import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import picos_y_valles as pv
import chart_reg as chartreg
import plot_matplotlib_transpuesto as pvt
import order_managment_reg as omr
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.stats import norm
import webbrowser

# ========== LECTURA DE FECHAS A OPERAR ==========
unique_dates_file = os.path.join('outputs', 'unique_dates.txt')
FECHA_MINIMA = '2025-04-10'

with open(unique_dates_file, 'r') as f:
    fechas = [line.strip() for line in f.readlines() if line.strip() >= FECHA_MINIMA]

print(f"✅ {len(fechas)} fechas cargadas")

#fecha = '2024-02-05' 
for fecha in fechas:
    print(f"\n📅 Analizando el día: {fecha}")


    hora_inicio_picos = '09:30:00'            #hora pera empezar el cálculo de los picos
    hora_inicio_valles = '07:30:00'           #hora pera empezar el cálculo de los valles   
    apertura_mercado = '14:30:00'             #hora para empezar a mirar el rango de la pre apertura

    hora_fin = '15:30:00'                     #hora inicio trading
    hora_future = '17:30:00'                  #hora final trading
    num_posiciones = 2               

    # ===========   LECTURA DE DATOS ==========

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
    df_picos, df_valles = pv.encontrar_picos_valles(df, prominence=2, distance=5)


    # Asegura datetime
    df_picos['fecha']  = pd.to_datetime(df_picos['fecha'])
    df_valles['fecha'] = pd.to_datetime(df_valles['fecha'])
    df_picos['fecha']  = df_picos['fecha'].dt.tz_convert(df.index.tz)
    df_valles['fecha'].dt.tz_convert(df.index.tz)

    # --- Ajustes de hora en Variables para formatear
    apertura_mercado = datetime.strptime(apertura_mercado, '%H:%M:%S').time()
    hora_inicio_picos = datetime.strptime(hora_inicio_picos, '%H:%M:%S').time()
    hora_inicio_valles = datetime.strptime(hora_inicio_valles, '%H:%M:%S').time()
    hora_fin = datetime.strptime(hora_fin, '%H:%M:%S').time()
    hora_future = datetime.strptime(hora_future, '%H:%M:%S').time()
    dt_future = pd.Timestamp.combine(START_DATE.date(), hora_future).tz_localize('Europe/Madrid')

    fecha_str = datetime.strptime(fecha, '%Y-%m-%d')
    dia_semana = fecha_str.strftime('%A')  # Día completo en inglés
    print(f"Analizando el dia: {fecha_str} que fue un {dia_semana}")

    # --- Regresión sobre picos
    df_picos_rango = df_picos[(df_picos['fecha'].dt.time >= hora_inicio_picos) & (df_picos['fecha'].dt.time <= hora_fin)]
    X_picos = (df_picos_rango['fecha'].astype(np.int64) // 10**9).to_numpy().reshape(-1, 1)
    y_picos = df_picos_rango['precio'].values
    reg_picos = LinearRegression().fit(X_picos, y_picos)

    # --- Regresión sobre valles
    df_valles_rango = df_valles[(df_valles['fecha'].dt.time >= hora_inicio_valles) & (df_valles['fecha'].dt.time <= hora_fin)]
    X_valles = (df_valles_rango['fecha'].astype(np.int64) // 10**9).to_numpy().reshape(-1, 1)
    y_valles = df_valles_rango['precio'].values
    reg_valles = LinearRegression().fit(X_valles, y_valles)

    # --- Fechas para dibujar la recta sólida y dash, INDEPENDIENTES
    dt_start_picos = pd.Timestamp.combine(START_DATE.date(), hora_inicio_picos).tz_localize('Europe/Madrid')
    dt_start_valles = pd.Timestamp.combine(START_DATE.date(), hora_inicio_valles).tz_localize('Europe/Madrid')
    dt_fin = pd.Timestamp.combine(START_DATE.date(), hora_fin).tz_localize('Europe/Madrid')
    dt_proj = dt_future

    fechas_solid_picos = pd.date_range(dt_start_picos, dt_fin, freq='min')
    fechas_solid_valles = pd.date_range(dt_start_valles, dt_fin, freq='min')
    fechas_dash = pd.date_range(dt_fin + timedelta(minutes=1), dt_proj, freq='min')

    # --- Predicciones, INDEPENDIENTES
    X_solid_picos = (fechas_solid_picos.astype(np.int64) // 10**9).to_numpy().reshape(-1, 1)
    y_solid_picos = reg_picos.predict(X_solid_picos)
    X_dash_picos = (fechas_dash.astype(np.int64) // 10**9).to_numpy().reshape(-1, 1)
    y_dash_picos = reg_picos.predict(X_dash_picos)
    X_solid_valles = (fechas_solid_valles.astype(np.int64) // 10**9).to_numpy().reshape(-1, 1)
    y_solid_valles = reg_valles.predict(X_solid_valles)
    X_dash_valles = (fechas_dash.astype(np.int64) // 10**9).to_numpy().reshape(-1, 1)
    y_dash_valles = reg_valles.predict(X_dash_valles)

    # --- Filtra datos para dibujar precio desde el rango más amplio (para todo el chart/volumen)
    hora_min_graf = min(hora_inicio_picos, hora_inicio_valles)
    hora_max_graf = hora_future
    mask_full = (df.index.time >= hora_min_graf) & (df.index.time <= hora_max_graf)
    df_plot = df[mask_full]

    # --- Picos y valles para scatter sólo en rango de cada curva
    mask_picos_plot = (df_picos['fecha'] >= dt_start_picos) & (df_picos['fecha'] <= dt_proj)
    mask_valles_plot = (df_valles['fecha'] >= dt_start_valles) & (df_valles['fecha'] <= dt_proj)

    # ... Tu código anterior para cargar datos y calcular regresiones ...

    # =================== ORDER MANAGEMENT REGRESION ===================

    df_trades = omr.order_management_reg(
        df, df_picos, df_valles,
        reg_picos, reg_valles,
        hora_fin,
        outlier_sigma=0.1,
        stop_points=10,
        target_points=10,
        num_pos=3,
        hora_limite_operaciones=hora_future  
    )

    if not df_trades.empty:
        # Renombra columnas para que sean las que espera el chart
        df_trades = df_trades.rename(columns={
            'tipo': 'entry_type',
            'fecha_entrada': 'entry_time',
            'precio_entrada': 'entry_price',
            'regresion': 'regression',
            'diferencia': 'diff'
        })

        # Limpia cualquier cosa rara: normaliza compra/venta, y cualquier Long/Short con espacios o saltos
        def normalize_entry_type(val):
            v = str(val).strip().lower()
            if v == 'compra':
                return 'Long'
            elif v == 'venta':
                return 'Short'
            elif 'long' in v:
                return 'Long'
            elif 'short' in v:
                return 'Short'
            else:
                return ''
        df_trades['entry_type'] = df_trades['entry_type'].apply(normalize_entry_type)

        print('Valores únicos en entry_type (post-normalización):', df_trades['entry_type'].unique())



    # =================== PLOTLY GRÁFICO INTERACTIVO ===================

    html_path = chartreg.plotly_regresion_chart(
        df, df_picos, df_valles,
        fechas_solid_picos, y_solid_picos, fechas_solid_valles, y_solid_valles,
        fechas_dash, y_dash_picos, y_dash_valles,
        mask_picos_plot, mask_valles_plot,
        apertura_mercado, hora_fin, hora_inicio_picos, hora_inicio_valles, hora_future,
        START_DATE,
        df_trades  # <--- pásalo directamente, no hace falta leer CSV
    )

    webbrowser.open('file://' + os.path.realpath(html_path))

    # ========== APPEND AL FICHERO FINAL ==========
    if not df_trades.empty:
        df_trades['fecha'] = fecha  # Añade la columna con la fecha actual (como string)

        output_file = os.path.join('outputs', 'traders_final.csv')
        file_exists = os.path.isfile(output_file)
        
        # Guarda con append, solo pone cabecera si el archivo no existe aún
        df_trades.to_csv(output_file, mode='a', header=not file_exists, index=False)




    '''''
    # =================== MATPLOTLIB TRANSPUESTO (desde 15:00) ===================
    hora_inicio_plt = '15:00:00'
    hora_inicio_plt = datetime.strptime(hora_inicio_plt, '%H:%M:%S').time()
    dt_start_plt = pd.Timestamp.combine(START_DATE.date(), hora_inicio_plt).tz_localize('Europe/Madrid')

    # Filtra solo los datos y regresiones desde la hora de inicio
    mask_post_plt = df.index.time >= hora_inicio_plt
    mask_solid_picos_plt = fechas_solid_picos >= dt_start_plt
    mask_solid_valles_plt = fechas_solid_valles >= dt_start_plt
    mask_dash_plt  = fechas_dash  >= dt_start_plt

    precios = df.loc[mask_post_plt, 'Close'].values
    tiempos = df.index[mask_post_plt]

    pvt.plot_matplotlib_transpuesto(
        precios,
        tiempos,
        fechas_solid_picos[mask_solid_picos_plt], y_solid_picos[mask_solid_picos_plt],
        fechas_solid_valles[mask_solid_valles_plt], y_solid_valles[mask_solid_valles_plt],
        fechas_dash[mask_dash_plt], y_dash_picos[mask_dash_plt], y_dash_valles[mask_dash_plt],
        df_picos, df_valles
    )

    '''''

print("✅ Proceso terminado. Todas las operaciones guardadas en outputs/traders_final.csv")