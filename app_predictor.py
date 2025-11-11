# app_predictor.py — Predictor de Compras (ONLINE Sheets + OFFLINE CSV)
# Ejecuta:
#   streamlit run app_predictor.py

import os, time, requests
import pandas as pd
import streamlit as st
from urllib.parse import quote
from typing import Optional
from predictor_core import forecast_all  # tu core

# ======================================
# CONFIG / MODOS
# ======================================
OFFLINE = False               # True: lee CSV locales
BASE = "templates_csv"

# Google Sheets (valores por defecto / modo single-tenant)
DEFAULT_SHEET_ID   = "1Pbjxy_V-NuTbfnN_SLpexkYx_w62Umsg7eBr2qrQJrI"
TAB_VENTAS         = "ventas_raw"
TAB_STOCK          = "stock_snapshot"
TAB_STOCK_TRANS    = "stock_transición"   # la seguimos leyendo, pero NO se suma
TAB_CONFIG         = "config"
TAB_INBOUND        = "inbound_po"
TAB_CLIENTES_CONF  = "clientes_config"

# Webhooks Make por defecto (3 escenarios)
DEFAULT_MAKE_WEBHOOK_S1_URL = "https://hook.us1.make.com/1pdchxe8cl7qg2oo7byqi4u5x4p9cc4n"
DEFAULT_MAKE_WEBHOOK_S2_URL = "https://hook.us1.make.com/vdj87rfcjpmeuccds9vieu45410tnsug"
DEFAULT_MAKE_WEBHOOK_S3_URL = "https://hook.us1.make.com/k50t6u1rtrswqd6vl4s8mqf2ndu6noa3"

# URLs de otras apps (para navegación)
REPORT_APP_URL = "http://localhost:8504"   # cámbiala en server

# ======================================
# HELPERS DE LECTURA
# ======================================
def read_gsheets(sheet_id: str, tab: str) -> pd.DataFrame:
    """Lee una pestaña de Google Sheets como CSV."""
    if OFFLINE:
        return pd.read_csv(os.path.join(BASE, f"{tab}.csv"))
    sheet_param = quote(tab, safe="")
    url = (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?"
        f"tqx=out:csv&sheet={sheet_param}"
    )
    return pd.read_csv(url)


@st.cache_data
def load_clientes_config() -> Optional[pd.DataFrame]:
    """Intenta leer la pestaña clientes_config del sheet por defecto."""
    try:
        df = read_gsheets(DEFAULT_SHEET_ID, TAB_CLIENTES_CONF)
        if "activo" in df.columns:
            df = df[df["activo"].astype(str).str.upper().isin(["TRUE", "1", "SI"])]
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception:
        return None


# ======================================
# NORMALIZADORES (los tuyos)
# ======================================
def _is_numeric_col(s: pd.Series) -> bool:
    return pd.to_numeric(s, errors="coerce").notna().sum() > 0


def normalize_ventas_sheet(df: pd.DataFrame) -> pd.DataFrame:
    cols_lc = {c.lower(): c for c in df.columns}
    fecha_col = cols_lc.get("fecha")
    sku_col   = cols_lc.get("sku")
    qty_col   = cols_lc.get("cantidad") or cols_lc.get("qty")
    if not fecha_col or not sku_col or not qty_col:
        raise ValueError("ventas_raw debe tener columnas 'fecha', 'sku' y 'cantidad'.")
    out = pd.DataFrame()
    out["fecha"] = pd.to_datetime(df[fecha_col], errors="coerce")
    out["sku"]   = df[sku_col].astype(str).str.strip().str.upper()
    out["qty"]   = pd.to_numeric(df[qty_col], errors="coerce").fillna(0)
    out = out.dropna(subset=["fecha", "sku"])
    return out


def _guess_sku_col(df: pd.DataFrame) -> str:
    prefer = ["sku", "SKU", "codigo", "producto", "Producto"]
    for c in prefer:
        if c in df.columns and not _is_numeric_col(df[c]):
            return c
    for c in df.columns:
        if not _is_numeric_col(df[c]):
            return c
    return df.columns[0]


def _guess_stock_col(df: pd.DataFrame) -> str:
    prefer = ["stock", "cantidad", "qty", "disponible", "on_hand"]
    for c in prefer:
        if c in df.columns and _is_numeric_col(df[c]):
            return c
    for c in df.columns:
        if _is_numeric_col(df[c]):
            return c
    raise ValueError("No encontré columna numérica de stock.")


def normalize_stock_sheet(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["sku", "stock"])
    sku_col = _guess_sku_col(df)
    qty_col = _guess_stock_col(df)
    out = pd.DataFrame()
    out["sku"]   = df[sku_col].astype(str).str.strip().str.upper()
    out["stock"] = pd.to_numeric(df[qty_col], errors="coerce").fillna(0)
    return out


def normalize_config_sheet(df: pd.DataFrame,
                           ventas_n: pd.DataFrame,
                           stock_n: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        skus = sorted(set(ventas_n["sku"]).union(set(stock_n["sku"])))
        return pd.DataFrame(
            {
                "sku": skus,
                "proveedor": "",
                "lead_time_dias": 0,
                "minimo_compra": 1,
                "multiplo": 1,
                "alias": None,
                "activo": True,
                "seguridad_dias": 0,
            }
        )

    df2 = df.copy()
    df2.columns = [str(c).strip().lower() for c in df2.columns]

    if "sku" in df2.columns:
        rename_map = {
            "min_lote": "minimo_compra",
            "minimo_lote": "minimo_compra",
            "multiplo_lote": "multiplo",
        }
        df2 = df2.rename(columns={k: v for k, v in rename_map.items() if k in df2.columns})
        for c in ["lead_time_dias", "minimo_compra", "multiplo", "seguridad_dias"]:
            if c in df2.columns:
                df2[c] = pd.to_numeric(df2[c], errors="coerce").fillna(0)
        df2["sku"] = df2["sku"].astype(str).str.strip().str.upper()
        for c, default in [
            ("proveedor", ""),
            ("minimo_compra", 1),
            ("multiplo", 1),
            ("alias", None),
            ("activo", True),
            ("seguridad_dias", 0),
        ]:
            if c not in df2.columns:
                df2[c] = default
        return df2

    # config global
    lead_time = int(pd.to_numeric(df2.get("lead_time_dias", pd.Series([0])).iloc[0], errors="coerce") or 0)
    seg_dias  = int(pd.to_numeric(df2.get("seguridad_dias", pd.Series([0])).iloc[0], errors="coerce") or 0)
    min_lote  = int(pd.to_numeric(df2.get("min_lote", pd.Series([1])).iloc[0], errors="coerce") or 1)

    skus = sorted(set(ventas_n["sku"]).union(set(stock_n["sku"])))
    cfg = pd.DataFrame(
        {
            "sku": skus,
            "proveedor": "",
            "lead_time_dias": lead_time,
            "minimo_compra": min_lote,
            "multiplo": 1,
            "alias": None,
            "activo": True,
            "seguridad_dias": seg_dias,
        }
    )
    return cfg


def normalize_inbound_sheet(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["sku", "qty", "eta", "estado"])
    cols = {c.lower(): c for c in df.columns}
    sku_c = cols.get("sku")
    qty_c = cols.get("qty") or cols.get("cantidad")
    eta_c = cols.get("eta") or cols.get("fecha")
    est_c = cols.get("estado") or cols.get("status")
    if not sku_c or not qty_c:
        return pd.DataFrame(columns=["sku", "qty", "eta", "estado"])
    out = pd.DataFrame()
    out["sku"]    = df[sku_c].astype(str).str.strip().str.upper()
    out["qty"]    = pd.to_numeric(df[qty_c], errors="coerce").fillna(0)
    out["eta"]    = pd.to_datetime(df[eta_c], errors="coerce") if eta_c else pd.NaT
    out["estado"] = df[est_c].astype(str).upper().str.strip() if est_c else "ABIERTA"
    out = out[out["qty"] > 0]
    return out


def prepare_inbound_for_core(inbound: pd.DataFrame) -> pd.DataFrame:
    if inbound is None or inbound.empty:
        return pd.DataFrame(columns=["sku", "qty", "eta", "estado"])
    df = inbound.copy()
    df = df[pd.to_numeric(df["qty"], errors="coerce").fillna(0) > 0]
    today = pd.Timestamp.today().normalize()
    df["eta"] = pd.to_datetime(df["eta"], errors="coerce")
    df.loc[df["eta"].isna(), "eta"] = today
    df.loc[df["eta"] < today, "eta"] = today
    df_grp = df.groupby("sku", as_index=False)["qty"].sum()
    df_grp["eta"] = today
    df_grp["estado"] = "PENDIENTE"
    return df_grp


# ======================================
# UTIL WEBHOOK
# ======================================
def trigger_make(url: str, payload: dict) -> dict:
    if not url:
        return {"ok": False, "error": "webhook no configurado"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        return {"ok": r.ok, "status": r.status_code, "text": r.text[:500]}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ======================================
# UI
# ======================================
st.set_page_config(page_title="Predictor de Compras", layout="wide")

# multi-tenant (opcional)
clientes_df = load_clientes_config()

CURRENT_SHEET_ID = DEFAULT_SHEET_ID
MAKE_WEBHOOK_S1_URL = DEFAULT_MAKE_WEBHOOK_S1_URL
MAKE_WEBHOOK_S2_URL = DEFAULT_MAKE_WEBHOOK_S2_URL
MAKE_WEBHOOK_S3_URL = DEFAULT_MAKE_WEBHOOK_S3_URL
CURRENT_TENANT_ID   = "default"
KAME_CLIENT_ID      = ""
KAME_CLIENT_SECRET  = ""
USE_STOCK_TOTAL     = False  # está en la hoja igual, pero al final no sumamos transición

if clientes_df is not None and len(clientes_df):
    tenant_ids = clientes_df["tenant_id"].tolist()
    tenant_sel = st.sidebar.selectbox("Cliente", tenant_ids, index=0)
    row = clientes_df[clientes_df["tenant_id"] == tenant_sel].iloc[0]

    CURRENT_TENANT_ID  = row["tenant_id"]
    CURRENT_SHEET_ID   = row.get("sheet_id", DEFAULT_SHEET_ID)
    KAME_CLIENT_ID     = row.get("kame_client_id", "")
    KAME_CLIENT_SECRET = row.get("kame_client_secret", "")
    USE_STOCK_TOTAL    = str(row.get("use_stock_total", "FALSE")).upper() in ["TRUE", "1", "SI"]

    MAKE_WEBHOOK_S1_URL = row.get("webhook_s1", DEFAULT_MAKE_WEBHOOK_S1_URL)
    MAKE_WEBHOOK_S2_URL = row.get("webhook_s2", DEFAULT_MAKE_WEBHOOK_S2_URL)
    MAKE_WEBHOOK_S3_URL = row.get("webhook_s3", DEFAULT_MAKE_WEBHOOK_S3_URL)

# --- navegación lateral ---
st.sidebar.markdown("### Navegación")
st.sidebar.markdown(
    f"[📊 Reportería de ventas]({REPORT_APP_URL}?tenant={CURRENT_TENANT_ID})"
)
st.sidebar.markdown("---")

modo_datos = "ONLINE (KAME ERP)" if not OFFLINE else "OFFLINE (CSV)"
st.title("🧠 Predictor de Compras")
st.caption(f"Fuente de datos: **{modo_datos}** — Tenant: **{CURRENT_TENANT_ID}**")

# Botones (3)
bt1, bt2, bt3 = st.columns(3)
with bt1:
    if st.button("📘 Actualizar ventas (S1)", use_container_width=True):
        st.json(trigger_make(MAKE_WEBHOOK_S1_URL, {"reason": "ui_run", "tenant_id": CURRENT_TENANT_ID}))
with bt2:
    if st.button("📦 Actualizar stock total (S2)", use_container_width=True):
        st.json(
            trigger_make(
                MAKE_WEBHOOK_S2_URL,
                {
                    "reason": "ui_run",
                    "tenant_id": CURRENT_TENANT_ID,
                    "use_stock_total": True,
                },
            )
        )
with bt3:
    if st.button("🧾 Actualizar inbound (S3)", use_container_width=True):
        st.json(trigger_make(MAKE_WEBHOOK_S3_URL, {"reason": "ui_run", "tenant_id": CURRENT_TENANT_ID}))

st.markdown("")

# filtros
colA, colB, colC = st.columns(3)
freq    = colA.selectbox("Frecuencia", ["Mensual (M)", "Semanal (W)"], index=0)
horizon = colB.slider("Horizonte (períodos)", 2, 24, 6, 1)
modo    = colC.selectbox("Modo", ["Global", "Por SKU"], index=1)
sku_q   = st.text_input("SKU (opcional)") if modo == "Por SKU" else None

# opciones avanzadas
with st.expander("Opciones avanzadas (Make / Debug)"):
    disparar        = st.checkbox("Disparar S1/S2/S3 antes de predecir", value=False)
    mostrar_debug   = st.checkbox("Mostrar debug de columnas/valores config", value=False)
    mostrar_inbound = st.checkbox("Mostrar inbound agrupado", value=True)
    mostrar_stocks  = st.checkbox("Mostrar stocks (informativos)", value=True)

# botón principal
if st.button("Ejecutar predicción", type="primary", use_container_width=True):
    # opcional: disparar escenarios
    if disparar:
        st.info("Disparando S1/S2/S3…")
        st.write("S1:", trigger_make(MAKE_WEBHOOK_S1_URL, {"reason": "ui_run", "tenant_id": CURRENT_TENANT_ID}))
        st.write("S2:", trigger_make(MAKE_WEBHOOK_S2_URL, {"reason": "ui_run", "tenant_id": CURRENT_TENANT_ID, "use_stock_total": True}))
        st.write("S3:", trigger_make(MAKE_WEBHOOK_S3_URL, {"reason": "ui_run", "tenant_id": CURRENT_TENANT_ID}))
        st.write("Esperando 5s para que Make actualice las hojas…")
        time.sleep(5)

    # leer datos
    with st.spinner("Leyendo datos de Sheets…"):
        ventas_raw   = read_gsheets(CURRENT_SHEET_ID, TAB_VENTAS)
        stock_raw    = read_gsheets(CURRENT_SHEET_ID, TAB_STOCK)
        stock_tr_raw = read_gsheets(CURRENT_SHEET_ID, TAB_STOCK_TRANS)
        config_raw   = read_gsheets(CURRENT_SHEET_ID, TAB_CONFIG)
        inbound_raw  = read_gsheets(CURRENT_SHEET_ID, TAB_INBOUND)

        ventas  = normalize_ventas_sheet(ventas_raw)
        stock_p = normalize_stock_sheet(stock_raw)
        stock_t = normalize_stock_sheet(stock_tr_raw)
        config  = normalize_config_sheet(config_raw, ventas, stock_p)
        inbound = normalize_inbound_sheet(inbound_raw)

        # stock al core = solo stock_snapshot
        stock_total = stock_p.copy()

    # filtrar por SKU
    if modo == "Por SKU" and sku_q:
        f = str(sku_q).strip().lower()
        ventas      = ventas[ventas["sku"].str.lower() == f]
        stock_p     = stock_p[stock_p["sku"].str.lower() == f]
        stock_t     = stock_t[stock_t["sku"].str.lower() == f]
        stock_total = stock_total[stock_total["sku"].str.lower() == f]
        config      = config[config["sku"].str.lower() == f]
        inbound     = inbound[inbound["sku"].str.lower() == f]

    inbound_core = prepare_inbound_for_core(inbound)

    # ========================
    # PANEL RESUMEN ARRIBA
    # ========================
    sku_mostrar = sku_q.upper() if sku_q else "(varios)"
    stock_env = float(stock_total["stock"].sum()) if not stock_total.empty else 0
    inbound_env = int(inbound_core["qty"].sum()) if not inbound_core.empty else 0

    colm1, colm2, colm3, colm4 = st.columns(4)
    colm1.metric("SKU", sku_mostrar)
    colm2.metric("Stock enviado al core", stock_env)
    colm3.metric("Inbound detectado", inbound_env)
    placeholder_propuesta = colm4.empty()

    # ========================
    # SECCIÓN TÉCNICA EN EXPANDERS
    # ========================
    with st.expander("Estado de los Servicios 📄", expanded=False):
        st.write(f"Ventas: {'✅ OK' if not ventas_raw.empty else '⚠️ Vacío'}")
        st.write(f"Stock (stock_snapshot): {'✅ OK' if not stock_raw.empty else '⚠️ Vacío'}")
        st.write(f"Stock transición (solo informativo): {'✅ OK' if not stock_tr_raw.empty else '⚠️ Vacío'}")
        st.write(f"Inbound: {'✅ OK' if not inbound_raw.empty else '⚠️ Vacío'}")

    if mostrar_stocks:
        with st.expander("Stocks leídos", expanded=False):
            st.subheader("Stock (stock_snapshot)")
            st.dataframe(stock_p, use_container_width=True, hide_index=True)

            st.subheader("Stock transición (informativo, NO se usa en el cálculo)")
            st.dataframe(stock_t, use_container_width=True, hide_index=True)

            st.subheader("Stock TOTAL enviado al core")
            st.dataframe(stock_total, use_container_width=True, hide_index=True)

    if mostrar_inbound:
        with st.expander("Inbound (crudo de Sheets) / agrupado", expanded=False):
            st.subheader("Inbound (crudo de Sheets)")
            st.dataframe(inbound, use_container_width=True)
            st.subheader("Inbound agrupado que se envía al core")
            st.dataframe(inbound_core, use_container_width=True)

    if mostrar_debug:
        with st.expander("DEBUG de Config", expanded=False):
            st.code(list(config.columns), language="python")
            st.dataframe(config.head(), use_container_width=True)

    # ========================
    # SI NO HAY VENTAS → avisar
    # ========================
    if ventas.empty:
        st.warning("No hay ventas para los filtros dados.")
    else:
        with st.spinner("Calculando pronóstico…"):
            freq_code = "M" if freq.startswith("Mensual") else "W"
            det, res, prop = forecast_all(
                ventas=ventas,
                stock=stock_total,
                config=config,
                inbound=inbound_core,
                freq=freq_code,
                horizon_override=horizon,
            )

        st.success("Listo ✅")

        if not prop.empty:
            total_prop = int(prop["qty_sugerida"].sum())
            placeholder_propuesta.metric("Propuesta sugerida", total_prop)
        else:
            placeholder_propuesta.metric("Propuesta sugerida", 0)

        st.subheader("Propuesta de compra ↪")
        st.dataframe(prop, use_container_width=True)
        st.download_button(
            "Descargar propuesta (CSV)",
            prop.to_csv(index=False).encode("utf-8"),
            "propuesta.csv",
            "text/csv",
        )

        st.subheader("Resumen por SKU ↪")
        st.dataframe(res, use_container_width=True)
        st.download_button(
            "Descargar resumen (CSV)",
            res.to_csv(index=False).encode("utf-8"),
            "pred_resumen.csv",
            "text/csv",
        )

        st.subheader("Detalle por período")
        st.dataframe(det, use_container_width=True)
        st.download_button(
            "Descargar detalle (CSV)",
            det.to_csv(index=False).encode("utf-8"),
            "pred_detalle.csv",
            "text/csv",
        )


















