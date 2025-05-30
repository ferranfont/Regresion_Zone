# Traspone el grafico de precios y tiempos, mostrando los precios en el eje X y las fechas en el eje Y
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

def plot_matplotlib_transpuesto(
    precios, tiempos,
    fechas_solid_plt, y_solid_picos_plt, y_solid_valles_plt,
    fechas_dash_plt, y_dash_picos_plt, y_dash_valles_plt,
    df_picos=None, df_valles=None
):
    plt.figure(figsize=(10, 6))
    plt.plot(precios, tiempos, color='blue', linewidth=1, label='Precio vs Tiempo')

    # Si tienes picos y valles, plota los puntos (opcional)
    if df_picos is not None and df_valles is not None:
        mask_picos_plt = (df_picos['fecha'] >= tiempos[0]) & (df_picos['fecha'] <= tiempos[-1])
        mask_valles_plt = (df_valles['fecha'] >= tiempos[0]) & (df_valles['fecha'] <= tiempos[-1])
        plt.scatter(df_picos['precio'][mask_picos_plt], df_picos['fecha'][mask_picos_plt],
                    color='green', s=60, label='Picos (High)', zorder=5)
        plt.scatter(df_valles['precio'][mask_valles_plt], df_valles['fecha'][mask_valles_plt],
                    color='red', s=60, label='Valles (Low)', zorder=5)

    # Regresiones SOLO desde la hora de inicio
    plt.plot(y_solid_picos_plt, fechas_solid_plt, color='green', linewidth=1, label='Recta Picos')
    plt.plot(y_dash_picos_plt, fechas_dash_plt, color='green', linewidth=1, linestyle='--', label='Proy. Picos')
    plt.plot(y_solid_valles_plt, fechas_solid_plt, color='red', linewidth=1, label='Recta Valles')
    plt.plot(y_dash_valles_plt, fechas_dash_plt, color='red', linewidth=1, linestyle='--', label='Proy. Valles')

    # Campana de Gauss (distribución) como fondo
    media = np.mean(precios)
    std = np.std(precios)
    x_gauss = np.linspace(precios.min(), precios.max(), 300)
    y_gauss = norm.pdf(x_gauss, media, std)
    y_gauss_scaled = (y_gauss - y_gauss.min()) / (y_gauss.max() - y_gauss.min())
    y_gauss_scaled = y_gauss_scaled * (tiempos.max() - tiempos.min()) + tiempos.min()
    plt.plot(x_gauss, y_gauss_scaled, color='gray', linewidth=1, alpha=0.5, label='Campana de Gauss')

    # --- Obtén la fecha principal del gráfico ---
    if len(fechas_solid_plt) > 0:
        fecha_str = fechas_solid_plt[0].strftime('%Y-%m-%d')
    else:
        fecha_str = "sin_fecha"

    titulo = f'Precio (X) vs Tiempo (Y) transpuesto desde 15:00h – {fecha_str}'
    plt.title(titulo)
    plt.xlabel('Precio')
    plt.ylabel('Fecha/Hora')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()

    # --- Guardar PNG ---
    charts_dir = 'charts'
    os.makedirs(charts_dir, exist_ok=True)
    output_path = os.path.join(charts_dir, f'plot_transpuesto_{fecha_str}.png')
    plt.savefig(output_path, dpi=180)
    print(f'✅ Gráfico guardado como {output_path}')

    plt.show()
