import pandas as pd

# Lee el CSV (ajusta el nombre de archivo si cambia)
df = pd.read_csv('outputs/trades_final_2024-02-15.csv')

print("\n--- Valores de entry_type e índice ---")
for idx, val in enumerate(df['entry_type']):
    print(f"{idx}: {val}")

# Imprime el valor y el índice de la última fila
last_idx = df.index[-1]
last_val = df['entry_type'].iloc[-1]
print(f"\nÚltimo valor entry_type: {last_val} (índice {last_idx})")

# Muestra los valores únicos encontrados
print('\nValores únicos en entry_type:', df['entry_type'].unique())

# Muestra tabla resumen con columnas clave
print('\nResumen de operaciones:')
print(df[['entry_type', 'entry_time', 'entry_price']])

# Test visual de lo que se debería graficar (triángulos o error)
print("\n--- Test de símbolos que debería dibujar Plotly ---")
for idx, val in enumerate(df['entry_type']):
    vstr = str(val).strip().lower()
    if vstr == 'long':
        print(f"Fila {idx}: triangle-up VERDE (Long)")
    elif vstr == 'short':
        print(f"Fila {idx}: triangle-down ROJO (Short)")
    else:
        print(f"Fila {idx}: ERROR/FALLO, sale círculo gris: valor='{val}'")
