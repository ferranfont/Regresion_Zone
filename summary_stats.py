import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import webbrowser

# === ... (tu código de métricas, generación de gráficos y guardado .write_html) ... ===

# RUTAS DE LOS GRÁFICOS
chart_paths = [
    "outputs/charts/profit_acumulado_por_dia.html",
    "outputs/charts/drawdown.html",
    "outputs/charts/winrate_por_dia.html",
    "outputs/charts/distribucion_profit_operacion.html"
]

# Intentar abrir SIEMPRE con Google Chrome
chrome_paths = [
    "C:/Program Files/Google/Chrome/Application/chrome.exe %s",
    "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s"
]

opened = False
for chrome_path in chrome_paths:
    try:
        browser = webbrowser.get(chrome_path)
        for path in chart_paths:
            browser.open(os.path.abspath(path))
        opened = True
        print("✅ Gráficos abiertos en Chrome.")
        break
    except Exception as e:
        print(f"Chrome no encontrado en: {chrome_path}")

if not opened:
    print("❌ No se pudo abrir en Chrome automáticamente. Ábrelos manualmente o revisa la ruta de chrome.exe.")
    for path in chart_paths:
        print("Abre manualmente:", os.path.abspath(path))
