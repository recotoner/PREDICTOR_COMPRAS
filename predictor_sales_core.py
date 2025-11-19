# predictor_sales_core.py — Core de predicción de VENTAS (monto $)

import pandas as pd
import numpy as np

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except Exception:  # statsmodels puede no estar instalado; usamos fallback
    ExponentialSmoothing = None


# ============================================================
# Helpers de series
# ============================================================

def _build_series(df_sku: pd.DataFrame, freq: str) -> pd.Series:
    """
    Recibe un DF filtrado a un SKU, con columnas ['fecha', 'venta_neta'].
    Devuelve una serie temporal agregada por periodo (freq).
    """
    if df_sku is None or df_sku.empty:
        return pd.Series(dtype=float)

    s = (
        df_sku
        .dropna(subset=["fecha"])
        .set_index("fecha")
        .sort_index()["venta_neta"]
    )
    if s.empty:
        return s

    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    s_agg = s.resample(freq).sum()
    # quitamos períodos completamente vacíos
    s_agg = s_agg[s_agg.notna()]
    return s_agg.astype(float)


def _calc_basic_stats(s: pd.Series) -> dict:
    """Estadísticos básicos para clasificación y diagnóstico."""
    if s is None or s.empty:
        return {
            "n": 0,
            "zero_ratio": 1.0,
            "periodos_con_venta": 0,
            "mean_hist": 0.0,
            "mean_last6": 0.0,
        }

    n = len(s)
    zeros = float((s <= 0).sum())
    zero_ratio = zeros / n
    periodos_con_venta = int((s > 0).sum())

    mean_hist = float(s.mean())
    window = min(6, n)
    mean_last6 = float(s.tail(window).mean()) if window > 0 else 0.0

    return {
        "n": n,
        "zero_ratio": zero_ratio,
        "periodos_con_venta": periodos_con_venta,
        "mean_hist": mean_hist,
        "mean_last6": mean_last6,
    }


# ============================================================
# Clasificación de demanda (ADI + CV2)
# ============================================================

def _classify_demand(s: pd.Series) -> tuple:
    """
    Clasifica la demanda usando ADI (Average Demand Interval) y CV2 (Coefficient of Variation).
    
    Clasificación según Syntetos & Boylan (2005):
    - Smooth: ADI <= 1.32 y CV2 <= 0.49
    - Intermittent: ADI > 1.32 y CV2 <= 0.49
    - Erratic: ADI <= 1.32 y CV2 > 0.49
    - Lumpy: ADI > 1.32 y CV2 > 0.49
    
    Parameters
    ----------
    s : pd.Series
        Serie temporal de demanda.
    
    Returns
    -------
    tuple : (adi, cv2, klass)
        adi: Average Demand Interval
        cv2: Coefficient of Variation squared
        klass: str, una de ['smooth', 'intermittent', 'erratic', 'lumpy']
    """
    if s is None or s.empty:
        return 0.0, 0.0, "smooth"
    
    # Solo considerar valores positivos para el cálculo
    s_pos = s[s > 0]
    
    if len(s_pos) == 0:
        return float('inf'), 0.0, "intermittent"
    
    if len(s_pos) == 1:
        return float(len(s)), 0.0, "intermittent"
    
    # ADI: Average Demand Interval (períodos entre demandas)
    n_periods = len(s)
    n_demands = len(s_pos)
    adi = float(n_periods / n_demands) if n_demands > 0 else float('inf')
    
    # CV2: Coefficient of Variation squared
    mean_demand = float(s_pos.mean())
    if mean_demand > 0:
        cv2 = float((s_pos.std() / mean_demand) ** 2)
    else:
        cv2 = 0.0
    
    # Clasificación
    if adi <= 1.32 and cv2 <= 0.49:
        klass = "smooth"
    elif adi > 1.32 and cv2 <= 0.49:
        klass = "intermittent"
    elif adi <= 1.32 and cv2 > 0.49:
        klass = "erratic"
    else:  # adi > 1.32 and cv2 > 0.49
        klass = "lumpy"
    
    return adi, cv2, klass


# ============================================================
# Métricas de evaluación
# ============================================================

def _calculate_metrics(hist: pd.Series, fc: pd.Series, validation_periods: int = 6, 
                       modelo_usado: str = None, freq: str = "M") -> dict:
    """
    Calcula métricas de evaluación del forecast usando backtesting real con el mismo modelo.
    
    Métricas calculadas:
    - MAPE: Mean Absolute Percentage Error
    - RMSE: Root Mean Squared Error
    - MAE: Mean Absolute Error
    
    Hace backtesting real: usa el mismo modelo (ETS, Croston, etc.) que se usó para el forecast
    para predecir los períodos de validación y comparar con valores reales.
    
    Parameters
    ----------
    hist : pd.Series
        Serie histórica completa.
    fc : pd.Series
        Serie de forecast (usado para referencia, no para cálculo de métricas).
    validation_periods : int
        Número de períodos históricos a usar para validación (default: 6).
    modelo_usado : str
        Modelo que se usó para el forecast ('ETS', 'CROSTON_SBA', 'PROM6', etc.).
    freq : str
        Frecuencia de la serie ('M' o 'W').
    
    Returns
    -------
    dict : Diccionario con métricas {'mape', 'rmse', 'mae', 'n_validation'}
    """
    if hist is None or hist.empty or fc is None or fc.empty:
        return {
            "mape": None,
            "rmse": None,
            "mae": None,
            "n_validation": 0,
        }
    
    # Usar los últimos N períodos para validación (hold-out)
    if len(hist) < validation_periods + 2:
        # No hay suficientes datos para validación
        return {
            "mape": None,
            "rmse": None,
            "mae": None,
            "n_validation": 0,
        }
    
    # Separar datos en train y validation
    hist_train = hist.iloc[:-validation_periods]
    hist_val = hist.tail(validation_periods)
    
    if len(hist_train) < 2 or len(hist_val) < 2:
        return {
            "mape": None,
            "rmse": None,
            "mae": None,
            "n_validation": len(hist_val),
        }
    
    actual = hist_val.values
    
    # Hacer backtesting real usando el mismo modelo que se usó para el forecast
    predicted = None
    
    if modelo_usado == "ETS" and ExponentialSmoothing is not None and len(hist_train) >= 4:
        try:
            # Usar ETS para predecir los períodos de validación
            model = ExponentialSmoothing(
                hist_train,
                trend="add",
                seasonal=None,
                initialization_method="estimated",
            )
            fit = model.fit(optimized=True)
            fc_val = fit.forecast(len(hist_val))
            predicted = fc_val.values
        except Exception:
            predicted = None
    
    elif modelo_usado == "CROSTON_SBA" and len(hist_train) >= 3:
        try:
            # Usar Croston SBA para predecir los períodos de validación
            predicted_array = _croston_sba(hist_train, alpha=0.1, horizon=len(hist_val))
            predicted = predicted_array
        except Exception:
            predicted = None
    
    elif modelo_usado in ("PROM6_SHORT", "PROM6_FALLBACK", "PROM6"):
        # Para modelos de promedio, usar promedio de últimos períodos del train
        window = min(6, len(hist_train))
        predicted_value = float(hist_train.tail(window).mean()) if window > 0 else float(hist_train.mean())
        predicted = np.full(len(actual), predicted_value)
    
    # Fallback: si no se pudo usar el modelo, usar promedio simple
    if predicted is None:
        predicted_value = float(hist_train.tail(min(6, len(hist_train))).mean()) if len(hist_train) > 0 else 0.0
        predicted = np.full(len(actual), predicted_value)
    
    # Calcular promedio del train para filtros (necesario para el mask)
    train_mean_total = float(hist_train.mean()) if len(hist_train) > 0 else 0.0
    
    # Filtrar valores cero para MAPE (evitar división por cero)
    # También filtrar valores muy pequeños que pueden inflar el MAPE artificialmente
    mask = actual > (train_mean_total * 0.1)  # Solo considerar valores > 10% del promedio
    n_valid = int(mask.sum())
    
    if n_valid == 0:
        mape = None
    else:
        # Calcular MAPE solo sobre valores significativos
        errors_pct = np.abs((actual[mask] - predicted[mask]) / actual[mask]) * 100
        # Limitar errores extremos (más de 500%) para evitar distorsiones
        errors_pct = np.clip(errors_pct, 0, 500)
        mape = float(np.mean(errors_pct))
    
    # RMSE y MAE se calculan sobre todos los valores
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    mae = float(np.mean(np.abs(actual - predicted)))
    
    return {
        "mape": mape,
        "rmse": rmse,
        "mae": mae,
        "n_validation": len(hist_val),
    }


# ============================================================
# Ajuste estacional manual (parametrizable)
# ============================================================

def _apply_manual_seasonality(fc: pd.Series, freq: str, factors=None) -> pd.Series:
    """
    Aplica factores estacionales manuales por mes SOLO para frecuencia mensual.

    Parameters
    ----------
    fc : pd.Series
        Serie forecast con índice datetime.
    freq : str
        Frecuencia ('M' normalmente).
    factors : dict | None
        Diccionario {mes_int: factor}, por ejemplo:
        {12: 0.7, 1: 0.9, 2: 0.8, 3: 1.10}.
        - Se aceptan claves int o str ("12").
        - Si es None o vacío, no se aplica ajuste.
    """
    if fc is None or fc.empty:
        return fc

    # Solo tiene sentido para frecuencia mensual
    if not str(freq).upper().startswith("M"):
        return fc

    if not factors:
        return fc

    # --- Normalizamos el diccionario de factores: claves 1..12 en int ---
    factors_clean = {}
    try:
        items = factors.items()
    except Exception:
        items = []

    for k, v in items:
        try:
            mes = int(k)
            if 1 <= mes <= 12:
                fv = float(v)
                if np.isfinite(fv):
                    factors_clean[mes] = fv
        except Exception:
            # si algo viene raro, lo ignoramos
            continue

    if not factors_clean:
        # si después de limpiar no hay nada, devolvemos tal cual
        return fc

    ajustados = []
    for ts, val in fc.items():
        mes = getattr(ts, "month", None)
        factor = 1.0
        if mes is not None:
            factor = float(factors_clean.get(mes, 1.0))
        ajustados.append(float(val) * factor)

    return pd.Series(ajustados, index=fc.index, name=fc.name)


# ============================================================
# Croston SBA (intermitente)
# ============================================================

def _croston_sba(s: pd.Series, alpha: float, horizon: int) -> np.ndarray:
    """
    Implementación sencilla de Croston-SBA sobre una serie de demanda agregada.
    Devuelve un array de tamaño `horizon` con demanda POR PERIODO.

    Referencia: Croston (1972) + ajuste SBA (Syntetos & Boylan).
    """
    y = np.asarray(s, dtype=float)
    n = len(y)
    if n == 0:
        return np.zeros(horizon, dtype=float)

    # índices de demanda distinta de cero
    nz_idx = np.where(y > 0)[0]
    if len(nz_idx) == 0:
        return np.zeros(horizon, dtype=float)

    # demandas y tiempos entre demandas
    z = y[nz_idx]
    inter = np.diff(
        np.concatenate(([-1], nz_idx))
    )  # distancia en índices; primer inter incluye desde 0

    inter[0] = nz_idx[0] + 1  # corregimos el primer intervalo

    # inicialización simple
    z_hat = z[0]
    p_hat = float(inter[0])

    for k in range(1, len(z)):
        z_hat = alpha * z[k] + (1.0 - alpha) * z_hat
        p_hat = alpha * inter[k] + (1.0 - alpha) * p_hat

    if p_hat <= 0:
        demand_rate = 0.0
    else:
        # SBA: (1 - alpha / 2) * (z_hat / p_hat)
        demand_rate = (1.0 - alpha / 2.0) * (z_hat / p_hat)

    if not np.isfinite(demand_rate) or demand_rate < 0:
        demand_rate = 0.0

    return np.full(horizon, demand_rate, dtype=float)


# ============================================================
# Forecast de una sola serie
# ============================================================

def _forecast_one_series(s: pd.Series, freq: str, horizon: int, seasonality_factors=None):
    """
    Hace el forecast de una serie agregada.
    Devuelve (serie_hist, serie_forecast, modelo_usado, stats)
    donde stats incluye zero_ratio, periodos_con_venta, etc.

    seasonality_factors: dict {mes_int: factor} o None.
    """
    stats = _calc_basic_stats(s)

    if stats["n"] == 0:
        # sin historia, forecast = 0
        idx_future = pd.date_range(
            start=pd.Timestamp.today().normalize(),
            periods=horizon,
            freq=freq,
        )
        fc = pd.Series(0.0, index=idx_future, name="forecast")
        # también aplicamos estacionalidad por consistencia (aunque es todo 0)
        fc = _apply_manual_seasonality(fc, freq=freq, factors=seasonality_factors)
        return s, fc, "NO_DATA", stats

    s = s.astype(float).fillna(0.0)

    # Índice futuro (continuación natural de la serie)
    last_date = s.index.max()
    # Calcular el siguiente período de manera compatible con pandas moderno
    # Usar pd.DateOffset en lugar de to_offset para evitar el error de Timestamp
    if freq.upper().startswith("M"):
        # Frecuencia mensual: agregar 1 mes
        next_date = last_date + pd.DateOffset(months=1)
    elif freq.upper().startswith("W"):
        # Frecuencia semanal: agregar 1 semana
        next_date = last_date + pd.DateOffset(weeks=1)
    else:
        # Para otras frecuencias, intentar con to_offset pero usando el método correcto
        try:
            offset = pd.tseries.frequencies.to_offset(freq)
            # En pandas moderno, necesitamos usar el offset de manera diferente
            if hasattr(offset, 'delta'):
                next_date = last_date + offset.delta
            else:
                next_date = last_date + pd.DateOffset(days=1)  # fallback conservador
        except Exception:
            # Fallback: usar 1 día si todo falla
            next_date = last_date + pd.DateOffset(days=1)
    
    idx_future = pd.date_range(
        start=next_date,
        periods=horizon,
        freq=freq,
    )

    n = stats["n"]
    zero_ratio = stats["zero_ratio"]
    periodos_con_venta = stats["periodos_con_venta"]

    # Clasificación ADI+CV2 solo para información (no se usa en selección de modelos)
    adi, cv2, demand_class = _classify_demand(s)

    modelo = "PROM6"
    fc_values = None

    # ---------- Clasificación original (restaurada) ----------
    # 1) series muy cortas -> PROM6_SHORT
    if n < 6:
        mean_last = stats["mean_last6"]
        fc_values = np.full(horizon, mean_last, dtype=float)
        modelo = "PROM6_SHORT"

    # 2) series intermitentes -> Croston SBA (lógica original restaurada)
    elif (zero_ratio >= 0.40) and (periodos_con_venta >= 3):
        fc_values = _croston_sba(s, alpha=0.1, horizon=horizon)
        modelo = "CROSTON_SBA"

    # 3) resto -> ETS (tendencia), fallback PROM6
    else:
        if ExponentialSmoothing is not None and n >= 4:
            try:
                model = ExponentialSmoothing(
                    s,
                    trend="add",
                    seasonal=None,
                    initialization_method="estimated",
                )
                fit = model.fit(optimized=True)
                fc = fit.forecast(horizon)
                fc.index = idx_future
                fc_values = fc.to_numpy(dtype=float)
                modelo = "ETS"
            except Exception:
                fc_values = None

        if fc_values is None:
            mean_last = stats["mean_last6"]
            fc_values = np.full(horizon, mean_last, dtype=float)
            modelo = "PROM6_FALLBACK"

    # ---------- Guardrails para evitar explosiones ----------
    # Si el promedio del forecast se dispara >> promedio histórico,
    # recortamos a 1.5x el máximo entre promedio histórico y últimos 6.
    fc_values = np.asarray(fc_values, dtype=float)
    fc_mean = float(fc_values.mean()) if len(fc_values) else 0.0

    ref = max(stats["mean_hist"], stats["mean_last6"])
    if ref < 0:
        ref = 0.0

    max_factor = 1.5  # conservador; no queremos 10x

    if ref > 0 and fc_mean > ref * max_factor:
        target_mean = ref * max_factor
        if fc_mean > 0:
            scale = target_mean / fc_mean
            fc_values = fc_values * scale
            fc_mean = target_mean  # solo informativo

    # Construimos la serie forecast con el índice futuro
    fc_series = pd.Series(fc_values, index=idx_future, name="forecast")

    # Ajuste estacional manual (si hay factores configurados)
    fc_series = _apply_manual_seasonality(
        fc_series,
        freq=freq,
        factors=seasonality_factors,
    )

    # Calcular métricas de evaluación usando el mismo modelo que se usó para el forecast
    metrics = _calculate_metrics(s, fc_series, validation_periods=min(6, n), 
                                  modelo_usado=modelo, freq=freq)

    # Agregar clasificación y métricas a stats
    stats_enhanced = stats.copy()
    stats_enhanced["adi"] = adi
    stats_enhanced["cv2"] = cv2
    stats_enhanced["demand_class"] = demand_class
    stats_enhanced.update(metrics)

    return s, fc_series, modelo, stats_enhanced


# ============================================================
# Core público
# ============================================================

def forecast_sales(
    ventas: pd.DataFrame,
    freq: str = "M",
    horizon: int = 6,
    seasonality_factors=None,
):
    """
    Core de predicción de ventas.

    Parámetros
    ----------
    ventas : DataFrame
        Debe venir de normalize_ventas_for_sales y tener al menos
        columnas: ['fecha', 'sku', 'venta_neta'].
    freq : str
        Frecuencia 'M' (mensual) o 'W' (semanal).
    horizon : int
        Número de períodos futuros a proyectar.
    seasonality_factors : dict | None
        Diccionario opcional {mes_int: factor} para ajustar
        manualmente el forecast mensual (solo se aplica si freq es 'M').

    Returns
    -------
    det : DataFrame
        Detalle con columnas ['sku', 'fecha', 'venta_neta', 'tipo']
        donde tipo ∈ {'hist', 'forecast'}.
    res : DataFrame
        Resumen por SKU con columnas clave:
        ['sku', 'periodos_hist', 'venta_hist_total',
         'venta_hist_promedio', 'venta_hist_prom_ult6', 'venta_forecast_total', 'modelo',
         'zero_ratio', 'periodos_con_venta',
         'adi', 'cv2', 'demand_class', 'mape', 'rmse', 'mae']
        
        Donde:
        - venta_hist_promedio: Promedio histórico total (todos los períodos)
        - venta_hist_prom_ult6: Promedio de los últimos 6 meses
        - adi: Average Demand Interval
        - cv2: Coefficient of Variation squared
        - demand_class: Clasificación de demanda ('smooth', 'intermittent', 'erratic', 'lumpy')
        - mape: Mean Absolute Percentage Error (%)
        - rmse: Root Mean Squared Error
        - mae: Mean Absolute Error
    """
    base_cols = ["fecha", "sku", "venta_neta"]
    for c in base_cols:
        if c not in ventas.columns:
            raise ValueError(
                f"El DataFrame de ventas debe contener la columna '{c}'."
            )

    if ventas is None or ventas.empty:
        det_empty = pd.DataFrame(columns=["sku", "fecha", "venta_neta", "tipo"])
        res_empty = pd.DataFrame(
            columns=[
                "sku",
                "periodos_hist",
                "venta_hist_total",
                "venta_hist_promedio",
                "venta_hist_prom_ult6",
                "venta_forecast_total",
                "modelo",
                "zero_ratio",
                "periodos_con_venta",
                "adi",
                "cv2",
                "demand_class",
                "mape",
                "rmse",
                "mae",
            ]
        )
        return det_empty, res_empty

    ventas2 = ventas.copy()
    ventas2["fecha"] = pd.to_datetime(ventas2["fecha"], errors="coerce")
    ventas2 = ventas2.dropna(subset=["fecha"])
    ventas2["venta_neta"] = pd.to_numeric(
        ventas2["venta_neta"], errors="coerce"
    ).fillna(0.0)
    ventas2["sku"] = (
        ventas2["sku"].astype(str).str.strip().str.upper()
    )

    det_rows = []
    res_rows = []

    for sku, df_sku in ventas2.groupby("sku"):
        serie = _build_series(df_sku, freq=freq)
        hist, fc, modelo, stats = _forecast_one_series(
            serie,
            freq=freq,
            horizon=horizon,
            seasonality_factors=seasonality_factors,
        )

        # detalle histórico
        for fecha, val in hist.items():
            det_rows.append(
                {
                    "sku": sku,
                    "fecha": fecha,
                    "venta_neta": float(val),
                    "tipo": "hist",
                }
            )
        # detalle forecast
        for fecha, val in fc.items():
            det_rows.append(
                {
                    "sku": sku,
                    "fecha": fecha,
                    "venta_neta": float(val),
                    "tipo": "forecast",
                }
            )

        venta_hist_total = float(hist.sum()) if len(hist) else 0.0
        # Promedio total histórico (todos los períodos)
        venta_hist_prom_total = float(hist.mean()) if len(hist) > 0 else 0.0
        # Promedio de últimos 6 meses (para comparar con tendencia reciente)
        venta_hist_prom_ult6 = float(
            hist.tail(min(6, len(hist))).mean()
        ) if len(hist) else 0.0
        venta_fc_total = float(fc.sum()) if len(fc) else 0.0

        res_rows.append(
            {
                "sku": sku,
                "periodos_hist": int(len(hist)),
                "venta_hist_total": venta_hist_total,
                "venta_hist_promedio": venta_hist_prom_total,  # Cambiado: ahora es promedio total
                "venta_hist_prom_ult6": venta_hist_prom_ult6,  # Nuevo: promedio últimos 6 meses
                "venta_forecast_total": venta_fc_total,
                "modelo": modelo,
                "zero_ratio": stats["zero_ratio"],
                "periodos_con_venta": stats["periodos_con_venta"],
                # Nuevas métricas y clasificación
                "adi": stats.get("adi", None),
                "cv2": stats.get("cv2", None),
                "demand_class": stats.get("demand_class", "unknown"),
                "mape": stats.get("mape", None),
                "rmse": stats.get("rmse", None),
                "mae": stats.get("mae", None),
            }
        )

    det = pd.DataFrame(det_rows)
    if not det.empty:
        det = det.sort_values(["sku", "fecha", "tipo"])

    res = pd.DataFrame(res_rows)
    if not res.empty:
        res = res.sort_values("venta_forecast_total", ascending=False)

    return det, res





