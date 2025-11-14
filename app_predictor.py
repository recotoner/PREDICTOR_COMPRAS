# app_predictor.py — Predictor de Compras (ONLINE Sheets + OFFLINE CSV)
# Ejecuta:
#   streamlit run app_predictor.py

import os, time, requests, re, unicodedata
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

# URL de reportería (respaldo; la real se lee del sheet del tenant)
REPORT_APP_URL = "http://localhost:8504"

# HENRY: tiempos de espera recomendados después de S1/S2/S3 (en segundos)
COOLDOWN_S1 = 60   # ventas
COOLDOWN_S2 = 90   # stock total
COOLDOWN_S3 = 30   # inbound

# ======================================
# STREAMLIT BASE
# ======================================
st.set_page_config(page_title="Predictor de Compras", layout="wide")

# CSS
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: #0f172a !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        color: #ffffff !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] .stSelectbox label { color: #ffffff !important; }
    [data-testid="stSidebar"] .stSelectbox svg { color: #ffffff !important; }
    .stTextInput > div > div,
    .stSelectbox:not([data-testid="stSidebar"] .stSelectbox) > div > div {
        border: 1px solid rgba(15, 23, 42, 0.15) !important;
        border-radius: 10px !important;
    }
    .stButton > button {
        background-color: #0f766e !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        height: 46px;
        font-weight: 600;
    }
    .stButton > button:hover { background-color: #115e57 !important; }
    .hero-card{
        max-width: 760px; margin: 32px auto; padding: 28px 32px;
        background:#fff; border-radius:18px; box-shadow:0 10px 30px rgba(2,6,23,.08);
    }
    .hero-badge{display:inline-block; font-size:12px; padding:6px 10px; border-radius:999px;
        background:#e6f7f5; color:#065f5b; border:1px solid #b7ece7; margin-bottom:12px;}
    .hero-title{font-size:36px; font-weight:800; margin:6px 0 10px 0;}
    .hero-sub{color:#334155; margin-bottom:24px;}
    .hero-chart{width:100%; height:140px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================================
# HELPERS / AUTH
# ======================================
def slugify(name: str) -> str:
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()
    return s or "default"

def _truthy(v) -> bool:
    return str(v).replace("\u00a0"," ").strip().upper() in ("TRUE","1","SI","YES")

def _clean(s: str) -> str:
    return str(s).replace("\u00a0"," ").strip()

def ensure_clientes_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    d.columns = [str(c).replace("\u00a0"," ").strip() for c in d.columns]
    if "tenant_name" not in d.columns and "tenant" in d.columns:
        d = d.rename(columns={"tenant": "tenant_name"})
    if "is_active" not in d.columns and "activo" in d.columns:
        d["is_active"] = d["activo"]
    for col in [
        "tenant_id","tenant_name","sheet_id",
        "webhook_s1","webhook_s2","webhook_s3",
        "kame_client_id","kame_client_secret",
        "use_stock_total","login_email","login_pin","is_active"
    ]:
        if col not in d.columns:
            d[col] = "" if col not in ("use_stock_total","is_active") else True
    for col in ["tenant_id","tenant_name","sheet_id","webhook_s1","webhook_s2","webhook_s3",
                "kame_client_id","kame_client_secret","login_email","login_pin"]:
        if col in d.columns:
            d[col] = d[col].astype(str).map(_clean)
    if "login_email" in d.columns:
        d["login_email"] = d["login_email"].str.lower()
    d["is_active"] = d["is_active"].apply(_truthy)
    d["use_stock_total"] = d["use_stock_total"].apply(_truthy)
    d["tenant_name"] = d["tenant_name"].replace({pd.NA:"", None:""}).astype(str)
    d.loc[d["tenant_id"].eq("") | d["tenant_id"].isna(), "tenant_id"] = d["tenant_name"].apply(slugify)
    return d

def auth_tenant_row(cfg_df: pd.DataFrame, tenant_key: str, email: str, pin: str) -> Optional[pd.Series]:
    """Autentica permitiendo dejar email/pin en blanco en el sheet."""
    if cfg_df is None or cfg_df.empty:
        return None
    m = cfg_df[
        (cfg_df["tenant_name"].astype(str) == tenant_key) |
        (cfg_df["tenant_id"].astype(str) == tenant_key)
    ]
    if m.empty:
        return None
    row = m.iloc[0]
    if not _truthy(row.get("is_active", True)):
        return None
    le = _clean(row.get("login_email","")).lower()
    lp = _clean(row.get("login_pin",""))
    email_ok = (le == "") or (le == _clean(email).lower())
    pin_ok   = (lp == "") or (lp == _clean(pin))
    return row if (email_ok and pin_ok) else None

# ======================================
# HELPERS DE LECTURA
# ======================================
def read_gsheets(sheet_id: str, tab: str) -> pd.DataFrame:
    """Lee una pestaña de Google Sheets como CSV."""
    if OFFLINE:
        return pd.read_csv(os.path.join(BASE, f"{tab}.csv"))
    sheet_param = quote(tab, safe="")
    url = (f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?"
           f"tqx=out:csv&sheet={sheet_param}")
    return pd.read_csv(url)

@st.cache_data
def load_clientes_config() -> Optional[pd.DataFrame]:
    """Intenta leer la pestaña clientes_config del sheet por defecto."""
    try:
        df = read_gsheets(DEFAULT_SHEET_ID, TAB_CLIENTES_CONF)
        df = ensure_clientes_columns(df)
        return df[df["is_active"] == True].reset_index(drop=True)
    except Exception:
        return None

@st.cache_data
def load_global_urls(sheet_id: str) -> dict:
    """Lee la hoja config y devuelve predictor_url y reporteria_url (fila 1)."""
    try:
        df = read_gsheets(sheet_id, TAB_CONFIG)
        if df.empty:
            return {}
        df.columns = [str(c).strip().lower() for c in df.columns]
        row = df.iloc[0]
        return {
            "predictor_url": str(row.get("predictor_url", "")).strip(),
            "reporteria_url": str(row.get("reporteria_url", "")).strip(),
        }
    except Exception:
        return {}

# ======================================
# NORMALIZADORES (definidos ANTES de uso)
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
                "predictor_url": None,
                "reporteria_url": None,
            }
        )

    df2 = df.copy()
    df2.columns = [str(c).strip().lower() for c in df2.columns]

    # ----- caso 1: por SKU -----
    if "sku" in df2.columns:
        rename_map = {"min_lote": "minimo_compra", "minimo_lote": "minimo_compra", "multiplo_lote": "multiplo"}
        df2 = df2.rename(columns={k: v for k, v in rename_map.items() if k in df2.columns})
        for c in ["lead_time_dias", "minimo_compra", "multiplo", "seguridad_dias"]:
            if c in df2.columns:
                df2[c] = pd.to_numeric(df2[c], errors="coerce").fillna(0)
        df2["sku"] = df2["sku"].astype(str).str.strip().str.upper()
        for c, default in [
            ("proveedor", ""), ("minimo_compra", 1), ("multiplo", 1), ("alias", None),
            ("activo", True), ("seguridad_dias", 0), ("predictor_url", None), ("reporteria_url", None),
        ]:
            if c not in df2.columns:
                df2[c] = default
        return df2

    # ----- caso 2: global -----
    lead_time = int(pd.to_numeric(df2.get("lead_time_dias", pd.Series([0])).iloc[0], errors="coerce") or 0)
    seg_dias  = int(pd.to_numeric(df2.get("seguridad_dias", pd.Series([0])).iloc[0], errors="coerce") or 0)
    min_lote  = int(pd.to_numeric(df2.get("min_lote", pd.Series([1])).iloc[0], errors="coerce") or 1)
    predictor_url  = df2.get("predictor_url",  pd.Series([None])).iloc[0]
    reporteria_url = df2.get("reporteria_url", pd.Series([None])).iloc[0]
    skus = sorted(set(ventas_n["sku"]).union(set(stock_n["sku"])))
    cfg = pd.DataFrame(
        {
            "sku": skus, "proveedor": "", "lead_time_dias": lead_time, "minimo_compra": min_lote,
            "multiplo": 1, "alias": None, "activo": True, "seguridad_dias": seg_dias,
            "predictor_url": predictor_url, "reporteria_url": reporteria_url,
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
    out["sku"] = df[sku_c].astype(str).str.strip().str.upper()
    out["qty"] = pd.to_numeric(df[qty_c], errors="coerce").fillna(0)
    out["eta"] = pd.to_datetime(df[eta_c], errors="coerce") if eta_c else pd.NaT
    out["estado"] = (df[est_c].astype(str).str.upper().str.strip() if est_c else "ABIERTA")
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

# HENRY: helper para manejar el cooldown del botón de predicción
def start_cooldown(seconds: int):
    """Marca en sesión un tiempo mínimo antes de ejecutar la predicción."""
    if seconds <= 0:
        return
    now_ts = time.time()
    current = st.session_state.get("cooldown_until", 0)
    st.session_state["cooldown_until"] = max(current, now_ts + seconds)

# ======================================
# CARGA CONFIG CLIENTES
# ======================================
clientes_df = load_clientes_config()
if clientes_df is not None and not clientes_df.empty:
    clientes_df = ensure_clientes_columns(clientes_df)

# Botón Salir SIEMPRE visible (limpia sesión)
if st.sidebar.button("Salir", type="secondary", use_container_width=True):
    for k in ["TENANT_NAME","TENANT_ID","TENANT_ROW","cooldown_until","batch_scenarios_disparados"]:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

# ======================================
# LOGIN (si hay clientes_config)
# ======================================
def show_hero():
    st.markdown(
        """
        <div class="hero-card">
          <span class="hero-badge">Streamlit · Compras Inteligentes</span>
          <div class="hero-title">Tu copiloto de compras</div>
          <div class="hero-sub">Pronostica la demanda, descuenta inbound y propone cantidades con un clic.</div>
          <svg class="hero-chart" viewBox="0 0 600 160" preserveAspectRatio="none">
            <path d="M20 120 C140 40, 260 80, 380 90 S 560 120, 580 110" fill="none" stroke="#0f766e" stroke-width="4"/>
            <circle cx="220" cy="70" r="5" fill="#0f766e"/>
            <circle cx="420" cy="100" r="5" fill="#0f766e"/>
          </svg>
        </div>
        """,
        unsafe_allow_html=True
    )

if clientes_df is not None and not clientes_df.empty and "TENANT_ID" not in st.session_state:
    st.sidebar.header("Acceso")
    # lista con placeholder (sin preselección)
    if "tenant_name" in clientes_df.columns and clientes_df["tenant_name"].str.strip().ne("").any():
        tenant_list = clientes_df["tenant_name"].tolist()
        id_map = dict(zip(clientes_df["tenant_name"], clientes_df["tenant_id"]))
    else:
        tenant_list = clientes_df["tenant_id"].tolist()
        id_map = {t: t for t in tenant_list}
    options   = ["— Selecciona un cliente —"] + tenant_list
    tenant_sel = st.sidebar.selectbox("Cliente", options, index=0)
    email_in   = st.sidebar.text_input("Email")
    pin_in     = st.sidebar.text_input("PIN", type="password")
    go = st.sidebar.button("Entrar", use_container_width=True)

    show_hero()

    if go:
        if tenant_sel == "— Selecciona un cliente —":
            st.sidebar.error("Selecciona un cliente.")
            st.stop()
        # refrescamos cache por si cambió el sheet
        st.cache_data.clear()
        cfg_live = load_clientes_config()
        if cfg_live is None or getattr(cfg_live, "empty", True):
            cfg_live = clientes_df.copy()
        row = auth_tenant_row(cfg_live, id_map.get(tenant_sel, tenant_sel), email_in, pin_in)
        if row is None:
            st.sidebar.error("Credenciales inválidas o tenant inactivo.")
        else:
            tname = row.get("tenant_name") or row.get("tenant_id") or tenant_sel
            tid   = row.get("tenant_id") or slugify(tname)
            st.session_state["TENANT_NAME"] = str(tname)
            st.session_state["TENANT_ID"]   = str(tid)
            st.session_state["TENANT_ROW"]  = row.to_dict()
            st.rerun()
    st.stop()

# ======================================
# APP (ya logueado o sin clientes_config)
# ======================================
TENANT_NAME = st.session_state.get("TENANT_NAME", "default")
TENANT_ID   = st.session_state.get("TENANT_ID", "default")
TENANT_ROW  = st.session_state.get("TENANT_ROW", {})

CURRENT_SHEET_ID   = TENANT_ROW.get("sheet_id", DEFAULT_SHEET_ID) if TENANT_ROW else DEFAULT_SHEET_ID
KAME_CLIENT_ID     = TENANT_ROW.get("kame_client_id", "")
KAME_CLIENT_SECRET = TENANT_ROW.get("kame_client_secret", "")
USE_STOCK_TOTAL    = _truthy(TENANT_ROW.get("use_stock_total", False)) if TENANT_ROW else False

MAKE_WEBHOOK_S1_URL = TENANT_ROW.get("webhook_s1", DEFAULT_MAKE_WEBHOOK_S1_URL)
MAKE_WEBHOOK_S2_URL = TENANT_ROW.get("webhook_s2", DEFAULT_MAKE_WEBHOOK_S2_URL)
MAKE_WEBHOOK_S3_URL = TENANT_ROW.get("webhook_s3", DEFAULT_MAKE_WEBHOOK_S3_URL)

# URLs de ese sheet
urls_cfg = load_global_urls(CURRENT_SHEET_ID)
report_url_from_sheet = urls_cfg.get("reporteria_url", "").strip()
target_report_url = report_url_from_sheet or REPORT_APP_URL
if TENANT_ID:
    sep = "&" if "?" in target_report_url else "?"
    target_report_url = f"{target_report_url}{sep}tenant={TENANT_ID}"

# --- navegación lateral ---
st.sidebar.markdown("### Navegación")
st.sidebar.markdown(f"""<a href="{target_report_url}" target="_blank">📊 Reportería de ventas</a>""",
                    unsafe_allow_html=True)
st.sidebar.markdown("---")

modo_datos = "ONLINE (KAME ERP)" if not OFFLINE else "OFFLINE (CSV)"
st.title("🧠 Predictor de Compras ↪")
st.caption(f"Fuente de datos: **{modo_datos}** — Tenant: **{TENANT_NAME}**")

# --------------------------------------
# BOTONES SUPERIORES
# --------------------------------------
bt1, bt2, bt3 = st.columns(3)
with bt1:
    if st.button("📘 Actualizar ventas (S1)", use_container_width=True):
        resp = trigger_make(MAKE_WEBHOOK_S1_URL, {"reason": "ui_run", "tenant_id": TENANT_ID})
        if resp.get("ok"):
            st.success("Actualización de ventas enviada a Make. Espera ~60 segundos antes de ejecutar la predicción.")
            start_cooldown(COOLDOWN_S1)
        else:
            st.error(f"No se pudo actualizar ventas (S1). Código: {resp.get('status')}, detalle: {resp.get('error') or resp.get('text')}")
with bt2:
    if st.button("📦 Actualizar stock total (S2)", use_container_width=True):
        resp = trigger_make(MAKE_WEBHOOK_S2_URL, {
            "reason": "ui_run", "tenant_id": TENANT_ID, "use_stock_total": True
        })
        if resp.get("ok"):
            st.success("Actualización de stock total enviada. Espera ~90 segundos antes de ejecutar la predicción.")
            start_cooldown(COOLDOWN_S2)
        else:
            st.error(f"No se pudo actualizar stock (S2). Código: {resp.get('status')}, detalle: {resp.get('error') or resp.get('text')}")
with bt3:
    if st.button("🧾 Actualizar inbound (S3)", use_container_width=True):
        resp = trigger_make(MAKE_WEBHOOK_S3_URL, {"reason": "ui_run", "tenant_id": TENANT_ID})
        if resp.get("ok"):
            st.success("Actualización de inbound enviada. Espera ~30 segundos antes de ejecutar la predicción.")
            start_cooldown(COOLDOWN_S3)
        else:
            st.error(f"No se pudo actualizar inbound (S3). Código: {resp.get('status')}, detalle: {resp.get('error') or resp.get('text')}")

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

# --------------------------------------
# BOTÓN PRINCIPAL (con cooldown)
# --------------------------------------
now_ts = time.time()
cooldown_until = st.session_state.get("cooldown_until", 0)
remaining = max(0, int(cooldown_until - now_ts))

# HENRY: placeholder para mostrar un contador en vivo
cooldown_placeholder = st.empty()

if remaining > 0:
    # Contador regresivo en vivo (máx 90s según los COOLDOWN_* que definimos)
    for sec in range(remaining, 0, -1):
        cooldown_placeholder.warning(
            f"Se están actualizando datos desde KAME/Make. "
            f"Espera aproximadamente {sec} segundos antes de ejecutar la predicción."
        )
        time.sleep(1)
    # Al terminar el contador limpiamos estado y mensaje
    cooldown_placeholder.empty()
    st.session_state["cooldown_until"] = 0
    remaining = 0

# Si ya no hay espera pendiente, el botón queda habilitado
if remaining > 0:
    main_label = f"Ejecutar predicción (espera {remaining}s)"
    main_disabled = True
else:
    main_label = "Ejecutar predicción"
    main_disabled = False

if st.button(main_label, type="primary", use_container_width=True, disabled=main_disabled):
    # HENRY: modo "disparar S1/S2/S3 antes de predecir"
    if disparar and not st.session_state.get("batch_scenarios_disparados", False):
        st.info("Disparando S1 (ventas), S2 (stock total) y S3 (inbound) desde la app…")

        s1 = trigger_make(MAKE_WEBHOOK_S1_URL, {"reason": "ui_run", "tenant_id": TENANT_ID})
        s2 = trigger_make(MAKE_WEBHOOK_S2_URL, {
            "reason": "ui_run", "tenant_id": TENANT_ID, "use_stock_total": True
        })
        s3 = trigger_make(MAKE_WEBHOOK_S3_URL, {"reason": "ui_run", "tenant_id": TENANT_ID})

        if s1.get("ok") and s2.get("ok") and s3.get("ok"):
            # un solo cooldown largo para los tres (usa los mismos tiempos definidos arriba)
            start_cooldown(max(COOLDOWN_S1, COOLDOWN_S2, COOLDOWN_S3))
            st.session_state["batch_scenarios_disparados"] = True
        else:
            st.error(
                "Alguno de los escenarios S1/S2/S3 devolvió error. "
                "Revisa el escenario en Make antes de ejecutar la predicción."
            )
        st.rerun()  # volvemos a correr el script para que aparezca el contador


    # Si llegamos aquí:
    # - disparar == False  → predicción normal
    # - disparar == True y batch_scenarios_disparados == True → ya esperamos el cooldown, ahora sí predecimos

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

        stock_total = stock_p.copy()  # solo stock_snapshot

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

    # panel resumen
    sku_mostrar = sku_q.upper() if sku_q else "(varios)"
    stock_env   = float(stock_total["stock"].sum()) if not stock_total.empty else 0
    inbound_env = int(inbound_core["qty"].sum()) if not inbound_core.empty else 0

    colm1, colm2, colm3, colm4 = st.columns(4)
    colm1.metric("SKU", sku_mostrar)
    colm2.metric("Stock enviado al core", stock_env)
    colm3.metric("Inbound detectado", inbound_env)
    placeholder_propuesta = colm4.empty()

    # secciones técnicas
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

    # si no hay ventas
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

        # al terminar una corrida correcta, limpiamos el cooldown y el batch
        st.session_state["cooldown_until"] = 0
        st.session_state["batch_scenarios_disparados"] = False

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
































