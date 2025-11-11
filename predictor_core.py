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

    # Agregar por periodo (sin períodos faltantes aún)
    ts = v.groupby(["sku","period"], as_index=False)["qty"].sum()
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
    arr = hist.values.astype(float)
    if arr.size == 0:
        return np.zeros(horizon, dtype=float)
    k = min(6, arr.size)
    mean = arr[-k:].mean() if arr[-k:].sum() > 0 else 0.0
    return np.full(horizon, mean, dtype=float)


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
def _ets_forecast(arr: np.ndarray, horizon: int, freq: str) -> (np.ndarray, str):
    if not _HAS_SM or len(arr) < 3:
        return _simple_ma6(pd.Series(arr), horizon), "Naive-MA6"

    s = _season_length(freq)
    trend = 'add' if len(arr) >= 3 else None
    seasonal = None
    sp = None
    if s >= 2 and len(arr) >= 2*s:
        seasonal, sp = 'add', s
    try:
        model = ExponentialSmoothing(
            arr, trend=trend, seasonal=seasonal, seasonal_periods=sp,
            initialization_method='estimated'
        )
        fit = model.fit(optimized=True, use_brute=True)
        yhat = np.asarray(fit.forecast(horizon), dtype=float)
        yhat = np.maximum(yhat, 0.0)
        tag = f"ETS(A,{ 'A' if trend=='add' else 'N' },{ 'A' if seasonal=='add' else 'N' })"

        # GUARD: si ETS se disparó demasiado vs histórico, bajamos a MA6
        hist_mean = arr.mean() if arr.size else 0.0
        forecast_mean = yhat.mean() if yhat.size else 0.0
        if hist_mean > 0 and forecast_mean > 4 * hist_mean:
            # demasiado optimista, caer a MA6
            return _simple_ma6(pd.Series(arr), horizon), "Naive-MA6 (guard)"

        return yhat, tag
    except Exception:
        return _simple_ma6(pd.Series(arr), horizon), "Naive-MA6"


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

    if arr.size == 0 or np.nansum(arr) == 0:
        return np.zeros(horizon, dtype=float), "Naive-0"

    # nuestra regla más robusta
    if _needs_croston(arr, zr, nz, adi, klass):
        return _croston_sba(arr, horizon), "Croston-SBA"

    # si no, ETS con guardas
    return _ets_forecast(arr, horizon, freq)


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

    rows_det, rows_res, rows_prop = [], [], []

    for sku in skus:
        hist = ts.loc[ts["sku"] == sku, ["period","qty"]].sort_values("period")
        full = _expand_full_periods(hist, freq)
        periods = len(full)
        nz = int((full > 0).sum())
        zr = float(((full == 0).sum() / periods) if periods > 0 else 0.0)
        total_qty_hist = float(full.sum())
        adi, cv2, klass = _classify_adi_cv2(full)

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

        # períodos futuros
        if full.empty:
            start = today.to_period("M" if freq.upper().startswith("M") else "W").to_timestamp()
        else:
            last = full.index.max()
            start = (last + pd.offsets.MonthBegin(1)) if freq.upper().startswith("M") else (last + pd.offsets.Week(1))
        fut_idx = pd.date_range(start, periods=horizon, freq=_freq_code(freq))
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






