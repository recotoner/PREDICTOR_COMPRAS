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

def _pandas_offset_freq(freq: str) -> str:
    """
    Convierte alias de la app ('M' mensual, 'W' semanal) al offset de pandas vigente.
    'M' está deprecado desde pandas 2.2+ → usar 'ME' (fin de mes), mismo comportamiento histórico.
    """
    f = str(freq).strip().upper()
    if f in ("M", "BM"):
        return "ME"
    if f == "MS":
        return "MS"
    return str(freq).strip()


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
    s_agg = s.resample(_pandas_offset_freq(freq)).sum()
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
        "meses_winsorizados": 0,
    }


def _winsorize_series(s: pd.Series, iqr_factor: float = 2.5) -> tuple[pd.Series, int]:
    """
    Recorta solo picos altos: valores > Q3 + iqr_factor × IQR pasan a ese tope.
    No modifica valores bajos. Si IQR <= 0, devuelve la serie sin cambios.
    """
    if s is None or s.empty:
        return s, 0
    s = pd.to_numeric(s, errors="coerce").astype(float)
    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr <= 0:
        return s.copy(), 0
    cap = q3 + iqr_factor * iqr
    out = s.copy()
    mask = out > cap
    n_adj = int(mask.sum())
    if n_adj > 0:
        out.loc[mask] = cap
    return out, n_adj


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

    pd_freq = _pandas_offset_freq(freq)

    if stats["n"] == 0:
        # sin historia, forecast = 0
        idx_future = pd.date_range(
            start=pd.Timestamp.today().normalize(),
            periods=horizon,
            freq=pd_freq,
        )
        fc = pd.Series(0.0, index=idx_future, name="forecast")
        # también aplicamos estacionalidad por consistencia (aunque es todo 0)
        fc = _apply_manual_seasonality(fc, freq=freq, factors=seasonality_factors)
        return s, fc, "NO_DATA", stats

    s = s.astype(float).fillna(0.0)

    # Índice futuro (continuación natural de la serie)
    last_date = s.index.max()
    idx_future = pd.date_range(
        start=last_date + pd.tseries.frequencies.to_offset(pd_freq),
        periods=horizon,
        freq=pd_freq,
    )

    n = stats["n"]
    zero_ratio = stats["zero_ratio"]
    periodos_con_venta = stats["periodos_con_venta"]

    modelo = "PROM6"
    fc_values = None

    # ---------- Clasificación muy simple ----------
    # 1) series muy cortas -> PROM6_SHORT
    if n < 6:
        mean_last = stats["mean_last6"]
        fc_values = np.full(horizon, mean_last, dtype=float)
        modelo = "PROM6_SHORT"

    # 2) series intermitentes -> Croston SBA
    elif (zero_ratio >= 0.40) and (periodos_con_venta >= 3):
        fc_values = _croston_sba(s, alpha=0.1, horizon=horizon)
        modelo = "CROSTON_SBA"

    # 3) resto -> ETS (tendencia), fallback PROM6
    else:
        stats["meses_winsorizados"] = 0
        if ExponentialSmoothing is not None and n >= 4:
            try:
                s_ets = s
                if n >= 6:
                    s_ets, n_win = _winsorize_series(s)
                    stats["meses_winsorizados"] = n_win
                model = ExponentialSmoothing(
                    s_ets,
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

    # Ventas proyectadas no pueden ser negativas (ETS aditivo puede cruzar cero).
    fc_values = np.clip(fc_values, 0, None)

    # Construimos la serie forecast con el índice futuro
    fc_series = pd.Series(fc_values, index=idx_future, name="forecast")

    # Ajuste estacional manual (si hay factores configurados)
    fc_series = _apply_manual_seasonality(
        fc_series,
        freq=freq,
        factors=seasonality_factors,
    )

    return s, fc_series, modelo, stats


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
         'venta_hist_promedio', 'venta_forecast_total', 'modelo',
         'zero_ratio', 'periodos_con_venta', 'meses_winsorizados']
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
                "venta_forecast_total",
                "modelo",
                "zero_ratio",
                "periodos_con_venta",
                "meses_winsorizados",
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

        meses_win = int(stats.get("meses_winsorizados", 0) or 0)
        if meses_win > 0:
            print(
                f"[predictor_sales] Winsorización ETS: {sku} — "
                f"{meses_win} mes(es) ajustados (tope Q3 + 2.5×IQR)."
            )

        # detalle histórico (valores brutos del Excel, no la serie winsorizada)
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
        venta_hist_prom = float(
            hist.tail(min(6, len(hist))).mean()
        ) if len(hist) else 0.0
        venta_fc_total = float(fc.sum()) if len(fc) else 0.0

        res_rows.append(
            {
                "sku": sku,
                "periodos_hist": int(len(hist)),
                "venta_hist_total": venta_hist_total,
                "venta_hist_promedio": venta_hist_prom,
                "venta_forecast_total": venta_fc_total,
                "modelo": modelo,
                "zero_ratio": stats["zero_ratio"],
                "periodos_con_venta": stats["periodos_con_venta"],
                "meses_winsorizados": meses_win,
            }
        )

    det = pd.DataFrame(det_rows)
    if not det.empty:
        det = det.sort_values(["sku", "fecha", "tipo"])

    res = pd.DataFrame(res_rows)
    if not res.empty:
        res = res.sort_values("venta_forecast_total", ascending=False)

    return det, res





