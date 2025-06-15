import pandas as pd
import numpy as np

def order_management_reg(
    df, df_picos, df_valles,
    reg_picos, reg_valles,
    hora_fin,
    outlier_sigma=0.1,
    stop_points=10,
    target_points=10,
    num_pos=3,
    hora_limite_operaciones=None
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

    if not df_trades.empty:
        df_trades = df_trades.rename(columns={
            'tipo': 'entry_type',
            'fecha_entrada': 'entry_time',
            'precio_entrada': 'entry_price'
        })

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
        df_trades = df_trades.sort_values('entry_time')

        results = []
        posiciones_abiertas = []

        for idx, row in df_trades.iterrows():
            entry_time = row['entry_time']
            if pd.isnull(entry_time):
                continue
            if entry_time.tzinfo is None:
                entry_time = pd.to_datetime(entry_time).tz_localize('Europe/Madrid')
            else:
                entry_time = pd.to_datetime(entry_time).tz_convert('Europe/Madrid')

            if hora_limite_operaciones and entry_time.time() > hora_limite_operaciones:
                print(f"⛔ Entrada ignorada por hora: {entry_time.time()} > {hora_limite_operaciones}")
                continue

            posiciones_abiertas = [pos for pos in posiciones_abiertas if entry_time < pos['exit_time']]

            if len(posiciones_abiertas) < num_pos:
                entry_price = row['entry_price']
                entry_type = row['entry_type']
                target = row['target_points']
                stop = row['stop_points']

                df_future = df[df.index >= entry_time]
                if df_future.empty:
                    continue

                exit_time = None
                exit_price = None
                output_tag = None

                if entry_type == 'Long':
                    target_level = entry_price + target
                    stop_level = entry_price - stop
                    reached_target = df_future[df_future['Close'] >= target_level]
                    reached_stop = df_future[df_future['Close'] <= stop_level]
                    if not reached_target.empty and not reached_stop.empty:
                        if reached_target.index[0] < reached_stop.index[0]:
                            exit_time = reached_target.index[0]
                            exit_price = reached_target['Close'].iloc[0]
                            output_tag = 'target_out'
                        else:
                            exit_time = reached_stop.index[0]
                            exit_price = reached_stop['Close'].iloc[0]
                            output_tag = 'stop_out'
                    elif not reached_target.empty:
                        exit_time = reached_target.index[0]
                        exit_price = reached_target['Close'].iloc[0]
                        output_tag = 'target_out'
                    elif not reached_stop.empty:
                        exit_time = reached_stop.index[0]
                        exit_price = reached_stop['Close'].iloc[0]
                        output_tag = 'stop_out'
                    else:
                        exit_time = df_future.index[-1]
                        exit_price = df_future['Close'].iloc[-1]
                        output_tag = 'no_exit'
                    profit_points = exit_price - entry_price

                elif entry_type == 'Short':
                    target_level = entry_price - target
                    stop_level = entry_price + stop
                    reached_target = df_future[df_future['Close'] <= target_level]
                    reached_stop = df_future[df_future['Close'] >= stop_level]
                    if not reached_target.empty and not reached_stop.empty:
                        if reached_target.index[0] < reached_stop.index[0]:
                            exit_time = reached_target.index[0]
                            exit_price = reached_target['Close'].iloc[0]
                            output_tag = 'target_out'
                        else:
                            exit_time = reached_stop.index[0]
                            exit_price = reached_stop['Close'].iloc[0]
                            output_tag = 'stop_out'
                    elif not reached_target.empty:
                        exit_time = reached_target.index[0]
                        exit_price = reached_target['Close'].iloc[0]
                        output_tag = 'target_out'
                    elif not reached_stop.empty:
                        exit_time = reached_stop.index[0]
                        exit_price = reached_stop['Close'].iloc[0]
                        output_tag = 'stop_out'
                    else:
                        exit_time = df_future.index[-1]
                        exit_price = df_future['Close'].iloc[-1]
                        output_tag = 'no_exit'
                    profit_points = entry_price - exit_price
                else:
                    continue

                if pd.isnull(exit_time):
                    continue
                if isinstance(exit_time, pd.Timestamp):
                    if exit_time.tzinfo is None:
                        exit_time = exit_time.tz_localize('Europe/Madrid')
                    else:
                        exit_time = exit_time.tz_convert('Europe/Madrid')

                time_in_market = (exit_time - entry_time).total_seconds() / 60
                profit_usd = profit_points * 50

                row_out = row.to_dict()
                row_out.update({
                    'exit_time': exit_time,
                    'exit_price': exit_price,
                    'output_tag': output_tag,
                    'time_in_market': time_in_market,
                    'profit_in_points': profit_points,
                    'profit_in_$': profit_usd
                })
                results.append(row_out)
                posiciones_abiertas.append({'exit_time': exit_time})

        df_trades_final = pd.DataFrame(results)
        if not df_trades_final.empty:
            if isinstance(df_trades_final['entry_time'].iloc[0], pd.Timestamp):
                fecha_str = df_trades_final['entry_time'].dt.strftime('%Y-%m-%d').iloc[0]
            else:
                fecha_str = str(df_trades_final['entry_time'].iloc[0])[:10]
            output_path = f"outputs/trades_final_{fecha_str}.csv"
            df_trades_final.to_csv(output_path, index=False)

            total_profit = df_trades_final['profit_in_$'].sum()
            num_winners = (df_trades_final['profit_in_$'] > 0).sum()
            win_rate = num_winners / len(df_trades_final) * 100
            print(f"\U0001F3C1 Beneficio acumulado total: ${total_profit:.2f}")
            print(f"✅ Porcentaje de aciertos (beneficio positivo): {win_rate:.1f}% ({num_winners} de {len(df_trades_final)})")

            return df_trades_final

    print("⚠️ No se generaron operaciones. CSV no creado.")
    return df_trades
