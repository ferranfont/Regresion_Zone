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
    hora_inicio_picos, hora_inicio_valles, hora_future,
    START_DATE,
    df_trades=None
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

    # === RELLENO ENTRE REGRESIONES (solo entre apertura_mercado y hora_fin) ===
    dt_rango_1 = pd.Timestamp.combine(START_DATE.date(), apertura_mercado).tz_localize('Europe/Madrid')
    dt_rango_2 = pd.Timestamp.combine(START_DATE.date(), hora_fin).tz_localize('Europe/Madrid')
    idx_1_picos = np.argmin(np.abs(fechas_solid_picos - dt_rango_1))
    idx_2_picos = np.argmin(np.abs(fechas_solid_picos - dt_rango_2))
    idx_1_valles = np.argmin(np.abs(fechas_solid_valles - dt_rango_1))
    idx_2_valles = np.argmin(np.abs(fechas_solid_valles - dt_rango_2))
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
        fillcolor='rgba(135, 206, 250, 0.2)',
        line=dict(color='rgba(0,0,0,0)'),
        hoverinfo='skip',
        showlegend=False,
        name='Zona entre regresiones'
    ), row=1, col=1)

    # === ENTRADAS Y SALIDAS DE TRADES (si hay) ===
    if df_trades is not None and not df_trades.empty:
        # Normaliza y limpia espacios/tipos
        df_trades['entry_type'] = df_trades['entry_type'].astype(str).str.strip().str.capitalize()

        print('\n--- Test de símbolos que debería dibujar Plotly ---')
        for idx, row in df_trades.iterrows():
            entry_type_str = str(row['entry_type']).strip().capitalize()
            # Debug para ver el tipo
            if entry_type_str == 'Long':
                print(f"Fila {idx}: triangle-up VERDE (Long)")
            elif entry_type_str == 'Short':
                print(f"Fila {idx}: triangle-down ROJO (Short)")
            else:
                print(f"Fila {idx}: SÍMBOLO DEFAULT/GRIS (entry_type='{entry_type_str}')")

            # Entrada
            if pd.notnull(row['entry_time']) and pd.notnull(row['entry_price']):
                color = 'limegreen' if entry_type_str == 'Long' else 'red'
                symbol = 'triangle-up' if entry_type_str == 'Long' else 'triangle-down'
                fig.add_trace(go.Scatter(
                    x=[row['entry_time']],
                    y=[row['entry_price']],
                    mode='markers',
                    marker=dict(color=color, size=18, symbol=symbol),
                    name='Entry'
                ), row=1, col=1)

            # Salida
            if pd.notnull(row['exit_time']) and pd.notnull(row['exit_price']):
                fig.add_trace(go.Scatter(
                    x=[row['exit_time']],
                    y=[row['exit_price']],
                    mode='markers',
                    marker=dict(color='black', size=14, symbol='x'),
                    name='Exit'
                ), row=1, col=1)

            # Línea discontinua de entrada a salida
            if (
                pd.notnull(row['entry_time']) and pd.notnull(row['entry_price']) and
                pd.notnull(row['exit_time']) and pd.notnull(row['exit_price'])
            ):
                fig.add_trace(go.Scatter(
                    x=[row['entry_time'], row['exit_time']],
                    y=[row['entry_price'], row['exit_price']],
                    mode='lines',
                    line=dict(color='gray', width=1, dash='dot'),
                    name='Entry to Exit'
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
    hora_future_str = hora_future.strftime('%H:%M:%S') if hasattr(hora_future, 'strftime') else str(hora_future)
    hora_fin_str = hora_fin.strftime('%H:%M:%S') if hasattr(hora_fin, 'strftime') else str(hora_fin)

    for h in [hora_inicio_picos_str, hora_inicio_valles_str, apertura_mercado_str, '15:30:00', hora_fin_str, hora_future_str,'21:45:00']:
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
