import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import pandas as pd
import numpy as np

def plotly_regresion_chart(
    df_plot, df_picos, df_valles,
    fechas_solid_picos, y_solid_picos,
    fechas_solid_valles, y_solid_valles,
    fechas_dash, y_dash_picos, y_dash_valles,
    mask_picos_plot, mask_valles_plot,
    apertura_mercado, hora_fin,
    hora_inicio_picos, hora_inicio_valles,
    START_DATE
):
    chart_dir = 'charts'
    os.makedirs(chart_dir, exist_ok=True)
    fecha_str = START_DATE.date().strftime('%Y-%m-%d')
    html_path = os.path.join(chart_dir, f'chart_picos_valles_regresion_{fecha_str}.html')

    # =============== SUBPLOTS ===============
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.80, 0.20],
        vertical_spacing=0.03,
    )

    # === PRECIO ===
    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot['Close'],
        mode='lines', name='Close', line=dict(color='blue', width=1)
    ), row=1, col=1)

    # === PICOS Y VALLES ===
    fig.add_trace(go.Scatter(
        x=df_picos['fecha'][mask_picos_plot], y=df_picos['precio'][mask_picos_plot],
        mode='markers', name='Picos (High)', marker=dict(color='green', size=8, symbol='circle')
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df_valles['fecha'][mask_valles_plot], y=df_valles['precio'][mask_valles_plot],
        mode='markers', name='Valles (Low)', marker=dict(color='red', size=8, symbol='circle')
    ), row=1, col=1)

    # === LÍNEAS DE REGRESIÓN (PICOS) ===
    fig.add_trace(go.Scatter(
        x=fechas_solid_picos, y=y_solid_picos, mode='lines',
        name='Recta Picos', line=dict(color='green', width=1)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=fechas_dash, y=y_dash_picos, mode='lines',
        name='Recta Picos Extensión', line=dict(color='green', width=1, dash='dash')
    ), row=1, col=1)

    # === LÍNEAS DE REGRESIÓN (VALLES) ===
    fig.add_trace(go.Scatter(
        x=fechas_solid_valles, y=y_solid_valles, mode='lines',
        name='Recta Valles', line=dict(color='red', width=1)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=fechas_dash, y=y_dash_valles, mode='lines',
        name='Recta Valles Extensión', line=dict(color='red', width=1, dash='dash')
    ), row=1, col=1)

    # === RELLENO ENTRE REGRESIONES (solo entre 15:30 y 16:30) ===
    # === RELLENO ENTRE REGRESIONES (solo entre apertura_mercado y hora_fin) ===
    dt_rango_1 = pd.Timestamp.combine(START_DATE.date(), apertura_mercado).tz_localize('Europe/Madrid')
    dt_rango_2 = pd.Timestamp.combine(START_DATE.date(), hora_fin).tz_localize('Europe/Madrid')

    # Busca los índices más cercanos en cada curva
    idx_1_picos = np.argmin(np.abs(fechas_solid_picos - dt_rango_1))
    idx_2_picos = np.argmin(np.abs(fechas_solid_picos - dt_rango_2))
    idx_1_valles = np.argmin(np.abs(fechas_solid_valles - dt_rango_1))
    idx_2_valles = np.argmin(np.abs(fechas_solid_valles - dt_rango_2))

    # Polígono entre regresiones
    x_polygon = [
        fechas_solid_valles[idx_1_valles], fechas_solid_valles[idx_2_valles],
        fechas_solid_picos[idx_2_picos], fechas_solid_picos[idx_1_picos], fechas_solid_valles[idx_1_valles]
    ]
    y_polygon = [
        y_solid_valles[idx_1_valles], y_solid_valles[idx_2_valles],
        y_solid_picos[idx_2_picos], y_solid_picos[idx_1_picos], y_solid_valles[idx_1_valles]
    ]
    fig.add_trace(go.Scatter(
        x=x_polygon,
        y=y_polygon,
        fill='toself',
        fillcolor='rgba(135, 206, 250, 0.2)',  # Azul celeste translúcido
        line=dict(color='rgba(0,0,0,0)'),
        hoverinfo='skip',
        showlegend=False,
        name='Zona entre regresiones'
    ), row=1, col=1)

    # === RELLENO VERDE/ROJO PASTEL ===
    df_dash = pd.DataFrame({'fecha': fechas_dash, 'regresion': y_dash_picos})
    df_dash.set_index('fecha', inplace=True)
    df_price_dash = df_plot[df_plot.index.isin(fechas_dash)]
    df_merged = df_dash.join(df_price_dash[['Close']], how='inner')
    df_above = df_merged[df_merged['Close'] > df_merged['regresion']]
    if not df_above.empty:
        x_fill = list(df_above.index) + list(df_above.index[::-1])
        y_fill = list(df_above['Close']) + list(df_above['regresion'][::-1])
        fig.add_trace(go.Scatter(
            x=x_fill, y=y_fill, fill='toself',
            fillcolor='rgba(152, 251, 152, 0.3)',
            line=dict(color='rgba(0,0,0,0)'),
            hoverinfo='skip',
            name='Precio > Regresión Picos',
            showlegend=False
        ), row=1, col=1)

    df_dash_valles = pd.DataFrame({'fecha': fechas_dash, 'regresion': y_dash_valles})
    df_dash_valles.set_index('fecha', inplace=True)
    df_price_dash = df_plot[df_plot.index.isin(fechas_dash)]
    df_merged_valles = df_dash_valles.join(df_price_dash[['Close']], how='inner')
    df_below = df_merged_valles[df_merged_valles['Close'] < df_merged_valles['regresion']]
    if not df_below.empty:
        x_fill_below = list(df_below.index) + list(df_below.index[::-1])
        y_fill_below = list(df_below['Close']) + list(df_below['regresion'][::-1])
        fig.add_trace(go.Scatter(
            x=x_fill_below, y=y_fill_below, fill='toself',
            fillcolor='rgba(255, 160, 160, 0.3)',
            line=dict(color='rgba(0,0,0,0)'),
            hoverinfo='skip',
            name='Precio < Regresión Valles',
            showlegend=False
        ), row=1, col=1)

    # === VOLUMEN SUBPLOT (fila 2) ===
    if 'Volumen' in df_plot.columns:
        fig.add_trace(go.Bar(
            x=df_plot.index,
            y=df_plot['Volumen'],
            marker_color='blue',
            opacity=0.99,
            name='Volumen'
        ), row=2, col=1)

    # === LÍNEAS VERTICALES HORARIAS ===
    hora_inicio_picos_str = hora_inicio_picos.strftime('%H:%M:%S') if hasattr(hora_inicio_picos, 'strftime') else str(hora_inicio_picos)
    hora_inicio_valles_str = hora_inicio_valles.strftime('%H:%M:%S') if hasattr(hora_inicio_valles, 'strftime') else str(hora_inicio_valles)
    apertura_mercado_str = apertura_mercado.strftime('%H:%M:%S') if hasattr(apertura_mercado, 'strftime') else str(apertura_mercado)
    hora_fin_str = hora_fin.strftime('%H:%M:%S') if hasattr(hora_fin, 'strftime') else str(hora_fin)

    # Añade ambos horarios de inicio para líneas verticales
    for h in [hora_inicio_picos_str, hora_inicio_valles_str, apertura_mercado_str, '15:30:00', hora_fin_str, '20:00:00']:
        vline_time = datetime.strptime(h, '%H:%M:%S').time()
        vline_stamp = pd.Timestamp.combine(START_DATE.date(), vline_time).tz_localize('Europe/Madrid')
        fig.add_vline(
            x=vline_stamp, line=dict(color='gray', width=1), opacity=0.5, row=1, col=1
        )

    # === LAYOUT FINAL ===
    fig.update_layout(
        dragmode='pan',
        title=f'Regresiones – {fecha_str}',
        width=1500,
        height=int(1400 * 0.6),
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(size=12, color="black"),
        plot_bgcolor='rgba(255,255,255,0.05)',
        paper_bgcolor='rgba(240,240,240,0.6)',
        showlegend=False,
        template='plotly_white',
        xaxis=dict(showgrid=False, linecolor='gray', linewidth=1),
        yaxis=dict(showgrid=False, linecolor='gray', linewidth=1),
        xaxis2=dict(showgrid=False, linecolor='gray', linewidth=1),
        yaxis2=dict(showgrid=False, linecolor='gray', linewidth=1),
    )

    fig.write_html(html_path, config={"scrollZoom": True})
    print(f"✅ Gráfico Plotly guardado como '{html_path}'")
    return html_path
