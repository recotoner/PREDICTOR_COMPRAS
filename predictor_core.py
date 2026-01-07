# predictor_core.py — ETS/Croston + SS + inbound + diagnósticos (zr, nz, ADI, CV2)
import math
import numpy as np
import pandas as pd

# Statsmodels es opcional; si no está, caemos a MA6
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    _HAS_SM = True
except Exception:
    _HAS_SM = False


# ---------------------------
# Helpers básicos
# ---------------------------
def _ceil_to_multiple(x: float, m: int) -> int:
    if m <= 1:
        return int(math.ceil(max(0, x)))
    return int(math.ceil(max(0, x) / m) * m)

def _days_per_period(freq: str) -> int:
    return 30 if freq.upper().startswith("M") else 7  # M≈30, W=7

def _season_length(freq: str) -> int:
    return 12 if freq.upper().startswith("M") else 52

def _freq_code(freq: str) -> str:
    return "MS" if freq.upper().startswith("M") else "W-MON"


# ---------------------------
# Preparación de series
# ---------------------------
def _prep_series(ventas: pd.DataFrame, freq: str) -> pd.DataFrame:
    v = ventas.copy()
    v["fecha"] = pd.to_datetime(v["fecha"], errors="coerce")
    v = v.dropna(subset=["fecha"])
    v["sku"]   = v["sku"].astype(str).str.strip().str.upper()
    v["qty"]   = pd.to_numeric(v["qty"], errors="coerce").fillna(0)

    per = "M" if freq.upper().startswith("M") else "W"
    v["period"] = v["fecha"].dt.to_period(per).dt.to_timestamp()

    # CORRECCIÓN CRÍTICA: Separar ventas positivas de notas de crédito
    # Las notas de crédito por cambios de productos se vuelven a facturar,
    # por lo que su efecto neto en unidades debería ser nulo.
    # ESTRATEGIA: Usar solo ventas positivas (sin sumar negativas) para el modelo.
    # Esto refleja la demanda real de unidades, no los ajustes contables.
    
    ventas_positivas = v[v["qty"] > 0].copy()
    notas_credito = v[v["qty"] < 0].copy()
    
    # DEBUG: Información sobre la separación
    if len(notas_credito) > 0:
        total_ventas_pos = ventas_positivas["qty"].sum()
        total_notas = abs(notas_credito["qty"].sum())
        print(f"[DEBUG _prep_series] Separación ventas/notas de crédito:")
        print(f"   Ventas positivas: {len(ventas_positivas)} registros, total: {total_ventas_pos:.2f}")
        print(f"   Notas de crédito: {len(notas_credito)} registros, total: {total_notas:.2f}")
        print(f"   Diferencia neta: {total_ventas_pos - total_notas:.2f}")
    
    # Agregar por periodo SOLO usando ventas positivas (demanda real)
    # Esto evita que las notas de crédito distorsionen el modelo
    ts = ventas_positivas.groupby(["sku","period"], as_index=False)["qty"].sum()
    
    # Si un período no tiene ventas positivas, no aparecerá en ts (será 0 implícitamente)
    # Esto es correcto: si no hay ventas reales, la demanda es 0
    
    return ts

def _expand_full_periods(hist_df: pd.DataFrame, freq: str) -> pd.Series:
    """Devuelve una serie por periodo CONTIGUA (rellena con 0), requerida para zr/nz/ETS."""
    if hist_df.empty:
        return pd.Series(dtype=float)
    start = hist_df["period"].min()
    end   = hist_df["period"].max()
    idx   = pd.date_range(start, end, freq=_freq_code(freq))
    s = (hist_df.set_index("period")["qty"]
         .reindex(idx, fill_value=0)
         .astype(float))
    s.index.name = "period"
    return s


# ---------------------------
# Modelos simples
# ---------------------------
def _simple_ma6(hist: pd.Series, horizon: int) -> np.ndarray:
    """
    Modelo simple basado en media móvil.
    Usa la media de los últimos períodos como predicción constante.
    """
    arr = hist.values.astype(float)
    if arr.size == 0:
        return np.zeros(horizon, dtype=float)
    k = min(6, arr.size)
    mean = arr[-k:].mean() if arr[-k:].sum() > 0 else 0.0
    return np.full(horizon, mean, dtype=float)

def _robust_forecast(hist: pd.Series, horizon: int) -> np.ndarray:
    """
    Modelo robusto que detecta tendencias y estacionalidad mensual.
    
    Estrategia:
    1. Si hay suficientes datos (≥12 meses), intenta detectar estacionalidad mensual
    2. Si hay suficientes datos (≥12 períodos), detecta tendencia (creciente/decreciente)
    3. Usa la media reciente (últimos períodos) como base
    4. Si hay tendencia clara, la proyecta suavemente
    5. Si hay estacionalidad, aplica factores estacionales mensuales
    6. Si no hay tendencia ni estacionalidad, usa la media reciente constante
    """
    arr = hist.values.astype(float)
    if arr.size == 0:
        return np.zeros(horizon, dtype=float)
    
    # Calcular medias de diferentes ventanas para detectar tendencia
    total_periods = len(arr)
    
    # Media reciente (últimos 6 períodos o últimos 50% si hay menos de 6)
    recent_window = min(6, max(3, total_periods // 2))
    recent_mean = arr[-recent_window:].mean()
    
    # Media histórica global (para normalización de factores estacionales)
    hist_mean = arr.mean() if arr.size > 0 else recent_mean
    
    # DETECCIÓN DE ESTACIONALIDAD MENSUAL (si tenemos al menos 12 meses de datos)
    # Esto requiere que hist tenga índices de tipo DatetimeIndex para poder extraer el mes
    seasonal_factors = None
    if total_periods >= 12:
        print(f"[DEBUG _robust_forecast] Intentando detectar estacionalidad (total_periods={total_periods})")
        try:
            # Verificar que el índice sea DatetimeIndex
            if isinstance(hist.index, pd.DatetimeIndex):
                print(f"[DEBUG _robust_forecast] Índice es DatetimeIndex, tipo: {type(hist.index)}")
                # Agrupar por mes del año y calcular la media de cada mes
                meses_medias = {}
                for idx, val in hist.items():
                    mes = idx.month
                    if mes not in meses_medias:
                        meses_medias[mes] = []
                    meses_medias[mes].append(val)
                
                print(f"[DEBUG _robust_forecast] Meses encontrados en histórico: {sorted(meses_medias.keys())}, total: {len(meses_medias)}")
                
                # Calcular media por mes si tenemos todos los meses representados
                if len(meses_medias) >= 12:  # Tenemos todos los meses del año
                    # Calcular media por mes
                    factores = {}
                    for mes in range(1, 13):
                        if mes in meses_medias and len(meses_medias[mes]) > 0:
                            media_mes = np.mean(meses_medias[mes])
                            factores[mes] = media_mes
                    
                    print(f"[DEBUG _robust_forecast] Factores calculados: {len(factores)} meses")
                    
                    if len(factores) == 12:
                        # Calcular promedio de factores para normalización
                        suma_factores = sum(factores.values())
                        if suma_factores > 0:
                            factor_promedio = suma_factores / 12.0
                            if factor_promedio > 0:
                                # Normalizar factores para que su promedio sea 1.0
                                seasonal_factors = {mes: factores[mes] / factor_promedio for mes in factores}
                                
                                # Verificar si hay variación estacional significativa (CV de factores > 0.15)
                                factores_values = list(seasonal_factors.values())
                                cv_factores = np.std(factores_values) / np.mean(factores_values) if np.mean(factores_values) > 0 else 0
                                
                                print(f"[DEBUG _robust_forecast] CV de factores estacionales: {cv_factores:.3f} (umbral: 0.15)")
                                
                                if cv_factores < 0.15:  # Variación muy baja, no es estacionalidad real
                                    print(f"[DEBUG _robust_forecast] ❌ CV demasiado bajo, no se considera estacionalidad")
                                    seasonal_factors = None
                                else:
                                    print(f"[DEBUG _robust_forecast] ✅ Estacionalidad mensual detectada (CV factores: {cv_factores:.3f})")
                                    # Log de factores para debug
                                    print(f"   Factores estacionales por mes:")
                                    meses_nombres = {1:"Ene", 2:"Feb", 3:"Mar", 4:"Abr", 5:"May", 6:"Jun",
                                                    7:"Jul", 8:"Ago", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dic"}
                                    for mes in sorted(seasonal_factors.keys()):
                                        print(f"     {meses_nombres.get(mes, mes)}: {seasonal_factors[mes]:.3f}")
                            else:
                                print(f"[DEBUG _robust_forecast] ❌ Factor promedio <= 0: {factor_promedio}")
                        else:
                            print(f"[DEBUG _robust_forecast] ❌ Suma de factores <= 0: {suma_factores}")
                    else:
                        print(f"[DEBUG _robust_forecast] ❌ No se tienen los 12 meses, solo {len(factores)}")
                else:
                    print(f"[DEBUG _robust_forecast] ❌ Menos de 12 meses únicos encontrados: {len(meses_medias)}")
            else:
                print(f"[DEBUG _robust_forecast] ❌ Índice NO es DatetimeIndex, tipo: {type(hist.index)}")
        except Exception as e:
            print(f"[DEBUG _robust_forecast] ❌ Error calculando estacionalidad: {e}")
            import traceback
            traceback.print_exc()
            seasonal_factors = None
    else:
        print(f"[DEBUG _robust_forecast] ❌ Muy pocos períodos para estacionalidad: {total_periods} < 12")
    
    # Media de períodos anteriores (para comparar tendencia)
    has_trend = False
    smoothed_trend = 0.0
    if total_periods >= 12:
        # Si hay suficientes datos, comparar primera mitad vs segunda mitad
        mid_point = total_periods // 2
        first_half_mean = arr[:mid_point].mean()
        second_half_mean = arr[mid_point:].mean()
        
        # Detectar tendencia: si la diferencia es significativa (>20%), hay tendencia
        if first_half_mean > 0:
            trend_ratio = (second_half_mean - first_half_mean) / first_half_mean
            has_trend = abs(trend_ratio) > 0.20  # 20% de cambio indica tendencia
            
            if has_trend:
                # Hay tendencia: calcular proyección suavizada
                trend_per_period = (second_half_mean - first_half_mean) / mid_point
                smoothed_trend = trend_per_period * 0.5  # Reducir la tendencia al 50% para suavizar
                print(f"[DEBUG _robust_forecast] Tendencia detectada: {trend_ratio*100:+.1f}%")
    
    # Generar predicciones
    predictions = []
    
    # Determinar el mes de inicio del horizonte de predicción
    start_month = None
    if isinstance(hist.index, pd.DatetimeIndex) and len(hist.index) > 0:
        try:
            # El último período histórico + 1 período
            last_period = hist.index[-1]
            # Calcular el primer mes del horizonte (siguiente mes después del último histórico)
            if isinstance(last_period, pd.Timestamp):
                next_period = last_period + pd.offsets.MonthBegin(1)
                start_month = next_period.month
        except Exception as e:
            print(f"[DEBUG _robust_forecast] Error calculando mes de inicio: {e}")
            pass
    
    for i in range(horizon):
        # Base: media reciente
        pred = recent_mean
        
        # Aplicar tendencia si existe
        if has_trend:
            pred = pred + (smoothed_trend * (i + 1))
        
        # Aplicar factor estacional mensual si existe
        if seasonal_factors is not None and start_month is not None:
            # Calcular el mes correspondiente a esta predicción (i)
            mes_prediccion = ((start_month - 1 + i) % 12) + 1
            factor_estacional = seasonal_factors.get(mes_prediccion, 1.0)
            # Aplicar factor estacional: pred = pred * factor_estacional
            # Pero ajustar para que mantenga la media reciente como base
            pred = recent_mean * factor_estacional
            
            # Si hay tendencia, también aplicarla sobre el valor estacional
            if has_trend:
                pred = pred + (smoothed_trend * (i + 1))
        
        predictions.append(max(0.0, pred))  # No permitir negativos
    
    return np.array(predictions, dtype=float)


# ---------------------------
# Reglas de intermitencia
# ---------------------------
def _needs_croston(
    arr: np.ndarray,
    zr: float,
    nz: int,
    adi: float,
    klass: str,
) -> bool:
    """Regla más robusta para decidir Croston."""
    if arr.size < 6:
        return False

    # Regla original
    if (zr >= 0.38) and (nz >= 3):
        return True

    # Regla por ADI (Syntetos–Boylan)
    if adi >= 1.32:
        return True

    # Regla por clasificación
    if klass in ("intermittent", "lumpy"):
        return True

    return False


def _croston_sba(arr: np.ndarray, horizon: int, alpha: float = 0.1) -> np.ndarray:
    y = np.asarray(arr, dtype=float)
    n = len(y)
    if n == 0 or np.nansum(y) == 0:
        return np.zeros(horizon, dtype=float)
    z = np.zeros(n)  # tamaño de demanda (>0)
    p = np.zeros(n)  # intervalo entre demandas
    q = 1
    first = True
    for t in range(n):
        if y[t] > 0:
            if first:
                z[t] = y[t]
                p[t] = q
                first = False
            else:
                z[t] = z[t-1] + alpha*(y[t] - z[t-1])
                p[t] = p[t-1] + alpha*(q - p[t-1])
            q = 1
        else:
            if t > 0:
                z[t] = z[t-1]
                p[t] = p[t-1]
            q += 1
    z_hat = z[-1] if n else 0.0
    p_hat = p[-1] if n else 1.0
    f = 0.0 if (p_hat is None or p_hat <= 0) else (z_hat / p_hat) * (1 - alpha/2.0)  # SBA
    return np.full(horizon, max(0.0, float(f)), dtype=float)


# ---------------------------
# ETS con guardas
# ---------------------------
def _ets_forecast(arr: np.ndarray, horizon: int, freq: str, full_series: pd.Series = None) -> (np.ndarray, str):
    print(f"\n{'='*60}")
    print(f"[DEBUG _ets_forecast] INICIO - Analizando estacionalidad")
    print(f"{'='*60}")
    print(f"   Longitud de serie histórica: {len(arr)}")
    print(f"   Frecuencia: {freq}")
    print(f"   Horizonte de predicción: {horizon}")
    
    if not _HAS_SM or len(arr) < 3:
        print(f"[DEBUG _ets_forecast] ⚠️ Usando modelo robusto (HAS_SM={_HAS_SM}, len(arr)={len(arr)})")
        # Usar full_series si está disponible para preservar DatetimeIndex
        hist_series = full_series if full_series is not None and isinstance(full_series.index, pd.DatetimeIndex) else pd.Series(arr)
        return _robust_forecast(hist_series, horizon), "Robust-Mean"

    s = _season_length(freq)
    trend = 'add' if len(arr) >= 3 else None
    seasonal = None
    sp = None
    
    print(f"   Longitud de estación requerida: {s}")
    
    # CORRECCIÓN CRÍTICA: Requerir más períodos para estacionalidad confiable
    # Con solo 2 años (24 períodos), la estacionalidad no es confiable y causa predicciones erróneas
    # Requerimos al menos 3 años (36 períodos) para detectar estacionalidad de forma confiable
    min_periods_estacionalidad = 3 * s if s >= 2 else float('inf')
    min_teorico = 2 * s if s >= 2 else 'N/A'
    
    print(f"   Mínimo teórico para estacionalidad: {min_teorico}")
    print(f"   Mínimo RECOMENDADO para estacionalidad confiable: {min_periods_estacionalidad if min_periods_estacionalidad != float('inf') else 'N/A'}")
    
    if s >= 2 and len(arr) >= min_periods_estacionalidad:
        seasonal, sp = 'add', s
        print(f"[DEBUG _ets_forecast] ✅ ESTACIONALIDAD DETECTADA: seasonal='{seasonal}', seasonal_periods={sp}")
        print(f"   La serie tiene {len(arr)} períodos, suficiente para estacionalidad confiable")
    elif s >= 2 and len(arr) >= 2*s:
        # Tiene el mínimo teórico pero no el recomendado - advertir pero permitir
        print(f"[DEBUG _ets_forecast] ⚠️ Estacionalidad detectada con datos limitados")
        print(f"   La serie tiene {len(arr)} períodos (mínimo teórico: {2*s}, recomendado: {min_periods_estacionalidad})")
        print(f"   Se usará modelo SIN estacionalidad para mayor robustez")
        seasonal, sp = None, None
    else:
        print(f"[DEBUG _ets_forecast] ❌ NO se detectó estacionalidad")
        print(f"   Razón: len(arr)={len(arr)} < {min_teorico} (mínimo teórico)")
        print(f"   Se usará modelo sin estacionalidad")
    
    try:
        model = ExponentialSmoothing(
            arr, trend=trend, seasonal=seasonal, seasonal_periods=sp,
            initialization_method='estimated'
        )
        fit = model.fit(optimized=True, use_brute=True)
        
        # DEBUG: Extraer componentes del modelo ajustado si están disponibles
        if hasattr(fit, 'params'):
            print(f"[DEBUG _ets_forecast] Parámetros del modelo ajustado:")
            params = fit.params
            smoothing_params = []
            for key in ['smoothing_level', 'smoothing_trend', 'smoothing_seasonal']:
                if key in params:
                    val = params[key]
                    # Manejar valores nan
                    if pd.isna(val) or np.isnan(val):
                        print(f"   {key}: nan (no aplica)")
                    else:
                        print(f"   {key}: {val:.6f}")
                        smoothing_params.append(val)
            
            # GUARD CRÍTICO: Si todos los parámetros están en 0 o muy cerca de 0, el modelo no se ajustó bien
            # Esto puede pasar con estacionalidad cuando hay pocos datos
            # Filtrar valores válidos (no nan) antes de verificar
            valid_params = [p for p in smoothing_params if not (pd.isna(p) or np.isnan(p))]
            if valid_params and all(abs(p) < 0.001 for p in valid_params):
                print(f"[DEBUG _ets_forecast] ⚠️ ADVERTENCIA CRÍTICA: Todos los parámetros de suavizado están en 0")
                print(f"   Parámetros válidos encontrados: {valid_params}")
                print(f"   Esto indica que el modelo ETS no se ajustó correctamente")
                print(f"   Probablemente debido a pocos datos o configuración inadecuada")
                print(f"   Usando modelo robusto basado en media histórica general (más estable)")
                # Usar full_series si está disponible para preservar DatetimeIndex
                hist_series = full_series if full_series is not None and isinstance(full_series.index, pd.DatetimeIndex) else pd.Series(arr)
                return _robust_forecast(hist_series, horizon), "Robust-Mean (fallback-params-cero)"
        
        # DEBUG: Si hay estacionalidad, intentar mostrar los factores estacionales
        if seasonal == 'add' and sp is not None:
            try:
                # statsmodels puede exponer los componentes de diferentes formas
                if hasattr(fit, 'seasonal'):
                    seasonal_vals = fit.seasonal
                    if len(seasonal_vals) >= sp:
                        print(f"[DEBUG _ets_forecast] Factores estacionales (último ciclo de {sp} períodos):")
                        last_cycle = seasonal_vals[-sp:]
                        for i, val in enumerate(last_cycle, 1):
                            print(f"     Mes {i}: {val:.4f}")
                        print(f"   Media de factores estacionales: {last_cycle.mean():.4f}")
                        print(f"   Mínimo factor estacional: {last_cycle.min():.4f} (mes {last_cycle.argmin() + 1})")
                        print(f"   Máximo factor estacional: {last_cycle.max():.4f} (mes {last_cycle.argmax() + 1})")
            except Exception as e:
                print(f"[DEBUG _ets_forecast] No se pudieron extraer factores estacionales: {e}")
        
        # DEBUG: Mostrar los últimos valores de la serie para contexto
        print(f"[DEBUG _ets_forecast] Últimos {min(12, len(arr))} valores de la serie histórica:")
        for i, val in enumerate(arr[-min(12, len(arr)):], 1):
            print(f"   Período -{min(12, len(arr)) - i + 1}: {val:.2f}")
        
        yhat_raw = np.asarray(fit.forecast(horizon), dtype=float)
        # DEBUG: Log antes de truncar a 0
        print(f"[DEBUG _ets_forecast] yhat_raw (antes de truncar): {yhat_raw}")
        
        # GUARD: Detectar si hay valores negativos (problema común con ETS estacional aditivo)
        negativos = (yhat_raw < 0).sum()
        if negativos > 0:
            print(f"[DEBUG _ets_forecast] ⚠️ {negativos} valores negativos detectados en la predicción")
            print(f"   Valores negativos: {yhat_raw[yhat_raw < 0]}")
            print(f"   Todos los valores yhat_raw: {yhat_raw}")
            
            # Calcular estadísticas del histórico para ajuste inteligente
            hist_mean = arr.mean() if arr.size else 0.0
            hist_std = arr.std() if arr.size and arr.std() > 0 else hist_mean * 0.3
            hist_min_positive = arr[arr > 0].min() if (arr > 0).any() else hist_mean * 0.1
            
            # Verificar si hay valores positivos en la predicción
            valores_positivos = (yhat_raw > 0).sum()
            min_pred = yhat_raw.min()
            max_pred = yhat_raw.max()
            
            print(f"[DEBUG _ets_forecast] Análisis: hist_mean={hist_mean:.2f}, valores_positivos={valores_positivos}, min_pred={min_pred:.2f}, max_pred={max_pred:.2f}")
            
            # Solo usar fallback si TODOS los valores son negativos (no solo algunos)
            if valores_positivos == 0:
                print(f"[DEBUG _ets_forecast] ⚠️ TODOS los valores son negativos, usando fallback MA6")
                # Usar full_series si está disponible para preservar DatetimeIndex
                hist_series = full_series if full_series is not None and isinstance(full_series.index, pd.DatetimeIndex) else pd.Series(arr)
                return _robust_forecast(hist_series, horizon), "Robust-Mean (fallback-todos-negativos)"
            
            # Si hay valores positivos, ajustar los negativos preservando la variación
            # Estrategia: desplazar solo los valores negativos, manteniendo los positivos intactos
            if max_pred > 0:
                # Hay valores positivos: ajustar solo los negativos
                # Calcular un mínimo razonable basado en el histórico
                min_reasonable = max(0.0, min(hist_min_positive * 0.15, hist_mean * 0.1))
                
                # Aplicar ajuste: los valores negativos se desplazan, los positivos se mantienen
                yhat = yhat_raw.copy()
                # Solo ajustar los valores negativos
                mask_negativos = yhat < 0
                if mask_negativos.any():
                    # Calcular el desplazamiento necesario para llevar el mínimo negativo al mínimo razonable
                    min_negativo = yhat[mask_negativos].min()
                    shift = min_reasonable - min_negativo
                    # Aplicar el desplazamiento solo a los valores negativos
                    yhat[mask_negativos] = yhat[mask_negativos] + shift
                    # Asegurar que no queden valores negativos después del ajuste
                    yhat = np.maximum(yhat, 0.0)
                
                print(f"[DEBUG _ets_forecast] Ajuste aplicado: min_reasonable={min_reasonable:.2f}")
                print(f"   yhat antes del ajuste: {yhat_raw}")
                print(f"   yhat después del ajuste: {yhat}")
            else:
                # No debería llegar aquí si valores_positivos > 0, pero por seguridad
                print(f"[DEBUG _ets_forecast] ⚠️ Caso inesperado, usando fallback MA6")
                # Usar full_series si está disponible para preservar DatetimeIndex
                hist_series = full_series if full_series is not None and isinstance(full_series.index, pd.DatetimeIndex) else pd.Series(arr)
                return _robust_forecast(hist_series, horizon), "Robust-Mean (fallback-inesperado)"
        else:
            yhat = np.maximum(yhat_raw, 0.0)
        
        # DEBUG: Log después de truncar
        print(f"[DEBUG _ets_forecast] yhat (después de ajuste): {yhat}")
        tag = f"ETS(A,{ 'A' if trend=='add' else 'N' },{ 'A' if seasonal=='add' else 'N' })"

        # GUARD: si ETS se disparó demasiado vs histórico, bajamos a MA6
        hist_mean = arr.mean() if arr.size else 0.0
        forecast_mean = yhat.mean() if yhat.size else 0.0
        if hist_mean > 0 and forecast_mean > 4 * hist_mean:
            # demasiado optimista, caer a MA6
            # Usar full_series si está disponible para preservar DatetimeIndex
            hist_series = full_series if full_series is not None and isinstance(full_series.index, pd.DatetimeIndex) else pd.Series(arr)
            return _robust_forecast(hist_series, horizon), "Robust-Mean (guard)"

        return yhat, tag
    except Exception:
        # Usar full_series si está disponible para preservar DatetimeIndex
        hist_series = full_series if full_series is not None and isinstance(full_series.index, pd.DatetimeIndex) else pd.Series(arr)
        return _robust_forecast(hist_series, horizon), "Robust-Mean"


# ---------------------------
# Clasificación ADI / CV2
# ---------------------------
def _classify_adi_cv2(full: pd.Series) -> tuple[float, float, str]:
    periods = len(full)
    nz = int((full > 0).sum())
    if nz == 0:
        return float("inf"), 0.0, "zero"

    adi = periods / nz
    nonzero = full[full > 0].values
    if len(nonzero) <= 1 or nonzero.mean() == 0:
        cv2 = 0.0
    else:
        cv2 = (np.std(nonzero, ddof=0) / np.mean(nonzero)) ** 2

    if adi < 1.32 and cv2 < 0.49:
        klass = "smooth"
    elif adi >= 1.32 and cv2 < 0.49:
        klass = "intermittent"
    elif adi < 1.32 and cv2 >= 0.49:
        klass = "erratic"
    else:
        klass = "lumpy"
    return float(adi), float(cv2), klass


# ---------------------------
# Selector de modelo por SKU
# ---------------------------
def _forecast_per_sku(
    full_series: pd.Series,
    horizon: int,
    freq: str,
    zr: float,
    nz: int,
    adi: float,
    klass: str,
) -> (np.ndarray, str):
    arr = np.asarray(full_series.values if isinstance(full_series, pd.Series) else full_series, dtype=float)
    
    # DEBUG: Log de entrada a _forecast_per_sku
    print(f"[DEBUG _forecast_per_sku] arr.size={arr.size}, np.nansum(arr)={np.nansum(arr) if arr.size > 0 else 0}")
    if arr.size > 0:
        print(f"   arr (primeros 10): {arr[:min(10, len(arr))]}")
        print(f"   arr (últimos 10): {arr[-min(10, len(arr)):]}")

    if arr.size == 0 or np.nansum(arr) == 0:
        print(f"[DEBUG _forecast_per_sku] ⚠️ Retornando ceros porque arr.size={arr.size} o suma={np.nansum(arr)}")
        return np.zeros(horizon, dtype=float), "Naive-0"

    # nuestra regla más robusta
    if _needs_croston(arr, zr, nz, adi, klass):
        return _croston_sba(arr, horizon), "Croston-SBA"

    # si no, ETS con guardas (pasar full_series para preservar DatetimeIndex)
    return _ets_forecast(arr, horizon, freq, full_series=full_series)


# ---------------------------
# Inbound normalización
# ---------------------------
def _normalize_inbound(inbound: pd.DataFrame | None) -> pd.DataFrame:
    if inbound is None or len(inbound) == 0:
        return pd.DataFrame(columns=["sku", "qty", "eta", "estado"])

    df = inbound.copy()
    cols = {str(c).lower(): c for c in df.columns}
    sku_c = cols.get("sku")
    qty_c = cols.get("qty") or cols.get("cantidad")
    eta_c = cols.get("eta") or cols.get("fecha") or cols.get("arribo") or cols.get("llegada")
    est_c = cols.get("estado")

    if not sku_c or not qty_c:
        return pd.DataFrame(columns=["sku", "qty", "eta", "estado"])

    out = pd.DataFrame()
    out["sku"] = df[sku_c].astype(str).str.strip().str.upper()

    s = (df[qty_c].astype(str)
         .str.replace('\xa0', '', regex=False)
         .str.replace('.',  '', regex=False)
         .str.replace(',',  '.', regex=False))
    out["qty"] = pd.to_numeric(s, errors="coerce").fillna(0).clip(lower=0)

    out["eta"] = pd.to_datetime(df[eta_c], errors="coerce") if eta_c else pd.NaT
    out["estado"] = (df[est_c].astype(str).str.upper().str.strip() if est_c is not None
                     else pd.Series(["ABIERTA"] * len(df)))

    out = out[out["qty"] > 0]
    closed = {
        "CERRADA","CERRADO","CANCELADA","CANCELADO","ANULADA","ANULADO",
        "RECEPCIONADA","RECEPCIONADO","INGRESADA","INGRESADO",
        "CLOSED","COMPLETE","COMPLETA"
    }
    out = out[~out["estado"].isin(closed)]
    return out[["sku","qty","eta","estado"]]


# ---------------------------
# API principal
# ---------------------------
def forecast_all(
    ventas: pd.DataFrame,
    stock: pd.DataFrame,
    config: pd.DataFrame,
    freq: str = "M",
    horizon_override: int | None = None,
    inbound: pd.DataFrame | None = None,
):
    run_id = f"run_{np.base_repr(np.random.randint(1 << 30), 36).lower()}"

    # Normalizar stock/config
    st = stock.copy()
    st["sku"] = st["sku"].astype(str).str.strip().str.upper()
    st["stock"] = pd.to_numeric(st["stock"], errors="coerce").fillna(0).clip(lower=0)

    cfg = config.copy()
    cfg["sku"] = cfg["sku"].astype(str).str.strip().str.upper()
    for c, d in [("lead_time_dias", 0), ("minimo_compra", 1), ("multiplo", 1), ("seguridad_dias", 0)]:
        if c not in cfg.columns:
            cfg[c] = d
        cfg[c] = pd.to_numeric(cfg[c], errors="coerce").fillna(d)
    if "proveedor" not in cfg.columns:
        cfg["proveedor"] = ""
    if "activo" not in cfg.columns:
        cfg["activo"] = True

    # Series históricas
    ts = _prep_series(ventas, freq)
    
    # DEBUG: Log de series históricas preparadas
    print(f"[DEBUG forecast_all] Ventas recibidas: {len(ventas)} filas")
    print(f"[DEBUG forecast_all] Series históricas (ts) después de _prep_series: {len(ts)} filas")
    if not ts.empty:
        print(f"[DEBUG forecast_all] SKUs únicos en ts: {sorted(ts['sku'].unique().tolist())}")
        print(f"[DEBUG forecast_all] Rango de períodos en ts: {ts['period'].min()} a {ts['period'].max()}")
    
    horizon = 6 if not horizon_override or horizon_override <= 0 else int(horizon_override)
    days_per = _days_per_period(freq)
    total_days = horizon * days_per

    # Inbound normalizado y acotado
    inbound_n = _normalize_inbound(inbound)
    today = pd.Timestamp.today().normalize()
    horizon_end = today + pd.Timedelta(days=total_days)
    if not inbound_n.empty:
        inbound_win = inbound_n[
            (inbound_n["eta"].notna())
            & (inbound_n["eta"] >= today)
            & (inbound_n["eta"] <= horizon_end)
        ]
    else:
        inbound_win = inbound_n

    # Universo de SKUs
    skus = set(ts["sku"]).union(st["sku"]).union(cfg["sku"])
    if not inbound_n.empty:
        skus = skus.union(inbound_n["sku"])
    skus = sorted(skus)
    
    # DEBUG: Log de universo de SKUs
    print(f"[DEBUG forecast_all] Universo de SKUs: {skus}")
    print(f"[DEBUG forecast_all] SKUs en ts (historial): {sorted(set(ts['sku'])) if not ts.empty else '[]'}")
    print(f"[DEBUG forecast_all] SKUs en stock: {sorted(set(st['sku'])) if not st.empty else '[]'}")
    print(f"[DEBUG forecast_all] SKUs en config: {sorted(set(cfg['sku'])) if not cfg.empty else '[]'}")

    rows_det, rows_res, rows_prop = [], [], []

    for sku in skus:
        hist = ts.loc[ts["sku"] == sku, ["period","qty"]].sort_values("period")
        full = _expand_full_periods(hist, freq)
        periods = len(full)
        nz = int((full > 0).sum())
        zr = float(((full == 0).sum() / periods) if periods > 0 else 0.0)
        total_qty_hist = float(full.sum())
        adi, cv2, klass = _classify_adi_cv2(full)
        
        # DEBUG: Log específico para cada SKU
        print(f"[DEBUG forecast_all] Procesando SKU: {sku}")
        print(f"   hist (raw): {len(hist)} filas, períodos: {hist['period'].tolist() if not hist.empty else '[]'}")
        if not hist.empty:
            print(f"   Valores históricos (últimos 12 períodos):")
            hist_last12 = hist.tail(12) if len(hist) > 12 else hist
            for _, row in hist_last12.iterrows():
                print(f"     {row['period'].strftime('%Y-%m')}: {row['qty']:.2f}")
        print(f"   full (expandido): {len(full)} períodos, suma total: {total_qty_hist}")
        print(f"   nz (períodos con demanda>0): {nz}, zr (tasa de ceros): {zr}")
        print(f"   total_qty_hist: {total_qty_hist}")
        if len(full) > 0:
            print(f"   Media histórica: {full.mean():.2f}, Mediana: {full.median():.2f}")
            print(f"   Últimos 6 períodos históricos: {full.tail(6).tolist()}")
            print(f"   Valores por mes (si es mensual):")
            if len(full) >= 12:
                # Agrupar por mes del año para ver estacionalidad
                meses = {}
                for idx, val in full.items():
                    mes = idx.month
                    if mes not in meses:
                        meses[mes] = []
                    meses[mes].append(val)
                for mes in sorted(meses.keys()):
                    print(f"     Mes {mes}: media={np.mean(meses[mes]):.2f}, valores={meses[mes]}")

        # aquí va el modelo, ahora más robusto
        yhat, model_tag = _forecast_per_sku(
            full_series=full,
            horizon=horizon,
            freq=freq,
            zr=zr,
            nz=nz,
            adi=adi,
            klass=klass,
        )
        
        # DEBUG: Log del resultado del modelo
        print(f"   Modelo usado: {model_tag}")
        print(f"   yhat (predicción): {yhat}, suma: {np.sum(yhat)}")

        # períodos futuros
        if full.empty:
            start = today.to_period("M" if freq.upper().startswith("M") else "W").to_timestamp()
        else:
            last = full.index.max()
            start = (last + pd.offsets.MonthBegin(1)) if freq.upper().startswith("M") else (last + pd.offsets.Week(1))
        
        # DEBUG: Log de cálculo de períodos futuros
        print(f"   Último período histórico: {last if not full.empty else 'N/A'}")
        print(f"   Start para períodos futuros: {start}")
        print(f"   Horizon: {horizon}")
        
        fut_idx = pd.date_range(start, periods=horizon, freq=_freq_code(freq))
        print(f"   Períodos futuros generados: {[dt.strftime('%Y-%m') for dt in fut_idx]}")
        print(f"   Valores yhat asignados: {yhat.tolist()}")
        
        for i, dt in enumerate(fut_idx):
            rows_det.append([run_id, sku, dt.strftime("%Y-%m"), dt, float(yhat[i])])

        demanda_H = float(np.sum(yhat))
        demanda_diaria = (demanda_H / total_days) if total_days > 0 else 0.0

        # config por SKU
        if sku in set(cfg["sku"]):
            row_cfg = cfg[cfg["sku"] == sku].iloc[0]
        else:
            row_cfg = pd.Series({
                "lead_time_dias":0, "minimo_compra":1, "multiplo":1,
                "proveedor":"", "seguridad_dias":0, "activo":True
            })

        seg_dias = int(row_cfg.get("seguridad_dias", 0) or 0)
        ss_qty = float(demanda_diaria * seg_dias)

        stock_actual = float(st.loc[st["sku"] == sku, "stock"].sum()) if (sku in set(st["sku"])) else 0.0
        inbound_qty = float(inbound_win.loc[inbound_win["sku"] == sku, "qty"].sum()) if not inbound_win.empty else 0.0

        minimo = int(row_cfg.get("minimo_compra", 1) or 1)
        multip = int(row_cfg.get("multiplo", 1) or 1)
        proveedor = str(row_cfg.get("proveedor", ""))

        propuesta_raw = max(0.0, demanda_H + ss_qty - (stock_actual + inbound_qty))
        propuesta_qty = 0 if propuesta_raw <= 0 else _ceil_to_multiple(propuesta_raw, multip)
        if propuesta_qty > 0 and minimo > 1:
            propuesta_qty = max(propuesta_qty, minimo)

        rows_res.append([
            run_id, sku, model_tag,
            round(demanda_H, 4), round(ss_qty, 4), round(stock_actual, 4),
            int(propuesta_qty), int(row_cfg.get("lead_time_dias", 0) or 0),
            minimo, multip, proveedor, int(seg_dias),
            periods, nz, round(zr, 2), round(adi, 4), round(cv2, 4), klass, round(total_qty_hist, 4)
        ])

        comentario = (
            f"Modelo={model_tag}; SS={round(ss_qty,1)}; stock={round(stock_actual,1)}; "
            f"inbound={round(inbound_qty,1)}; zr={round(zr,2)}; nz={nz}"
        )
        rows_prop.append([run_id, sku, int(propuesta_qty), proveedor, comentario, round(zr,2), nz, klass])

    pred_detalle = pd.DataFrame(
        rows_det,
        columns=["run_id","sku","period","fecha_periodo","demanda_predicha"]
    )

    pred_resumen = pd.DataFrame(
        rows_res,
        columns=[
            "run_id","sku","modelo","demanda_H","safety_stock","stock_actual",
            "propuesta_qty","lead_time_dias","minimo_compra","multiplo","proveedor","seguridad_dias",
            "periods","nz","zr","ADI","CV2","class","total_qty_hist"
        ]
    )

    propuesta = pd.DataFrame(
        rows_prop,
        columns=["run_id","sku","qty_sugerida","proveedor","comentario","zr","nz","class"]
    )

    return pred_detalle, pred_resumen, propuesta






