# app_predictor.py — Predictor de Compras (ONLINE Sheets + OFFLINE CSV)
# Ejecuta:
#   streamlit run app_predictor.py

import os, time, requests, re, unicodedata, hashlib, json
import pandas as pd
import streamlit as st
from urllib.parse import quote
from typing import Optional
from predictor_core import forecast_all  # tu core de COMPRAS

# Nuevo: core de VENTAS (monto $)
try:
    from predictor_sales_core import forecast_sales
except ImportError:
    forecast_sales = None

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
DEFAULT_MAKE_WEBHOOK_S1_URL = "https://hook.us1.make.com/qfr459tm0yth3xjbsjl3ef7vq3m44hwv"
DEFAULT_MAKE_WEBHOOK_S2_URL = "https://hook.us1.make.com/vdj87rfcjpmeuccds9vieu45410tnsug"
DEFAULT_MAKE_WEBHOOK_S3_URL = "https://hook.us1.make.com/k50t6u1rtrswqd6vl4s8mqf2ndu6noa3"

# URL de reportería (respaldo; la real se lee del sheet del tenant)
REPORT_APP_URL = "http://localhost:8504"

# HENRY: tiempos de espera recomendados después de S1/S2/S3 (en segundos)
COOLDOWN_S1 = 180   # ventas
COOLDOWN_S2 = 240   # stock total
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
    [data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #475569 0%, #64748b 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        height: 42px;
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(71, 85, 105, 0.2) !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(135deg, #64748b 0%, #475569 100%) !important;
        box-shadow: 0 4px 12px rgba(71, 85, 105, 0.3) !important;
        transform: translateY(-1px) !important;
    }
    /* Estilo mejorado para el link de navegación */
    [data-testid="stSidebar"] a {
        display: inline-block !important;
        padding: 12px 16px !important;
        background: linear-gradient(135deg, rgba(15, 118, 110, 0.1) 0%, rgba(20, 184, 166, 0.1) 100%) !important;
        border: 1.5px solid rgba(15, 118, 110, 0.3) !important;
        border-radius: 10px !important;
        color: #14b8a6 !important;
        text-decoration: none !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        margin: 4px 0 !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }
    [data-testid="stSidebar"] a:hover {
        background: linear-gradient(135deg, rgba(15, 118, 110, 0.2) 0%, rgba(20, 184, 166, 0.2) 100%) !important;
        border-color: rgba(15, 118, 110, 0.5) !important;
        color: #0d9488 !important;
        transform: translateX(4px) !important;
        box-shadow: 0 2px 8px rgba(15, 118, 110, 0.2) !important;
    }
    /* Estilo mejorado para radio buttons del sidebar */
    [data-testid="stSidebar"] .stRadio > div {
        background: rgba(15, 23, 42, 0.3) !important;
        border-radius: 12px !important;
        padding: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #e2e8f0 !important;
        font-weight: 500 !important;
        padding: 8px 12px !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
        display: inline-block !important;
        margin: 2px 0 !important;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(15, 118, 110, 0.15) !important;
        color: #14b8a6 !important;
    }
    [data-testid="stSidebar"] .stRadio input[type="radio"]:checked + label,
    [data-testid="stSidebar"] .stRadio [data-baseweb="radio"] [aria-checked="true"] ~ label {
        background: linear-gradient(135deg, #0f766e 0%, #14b8a6 100%) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 8px rgba(15, 118, 110, 0.3) !important;
    }
    [data-testid="stSidebar"] .stRadio [data-baseweb="radio"] {
        background: transparent !important;
    }
    .stTextInput > div > div,
    .stSelectbox:not([data-testid="stSidebar"] .stSelectbox) > div > div {
        border: 1px solid rgba(15, 23, 42, 0.15) !important;
        border-radius: 10px !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #0f766e 0%, #14b8a6 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        height: 48px;
        font-weight: 600;
        font-size: 15px;
        box-shadow: 0 4px 12px rgba(15, 118, 110, 0.3) !important;
        transition: all 0.3s ease !important;
        padding: 0 24px !important;
    }
    .stButton > button:hover { 
        background: linear-gradient(135deg, #115e57 0%, #0f766e 100%) !important;
        box-shadow: 0 6px 16px rgba(15, 118, 110, 0.4) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button:active {
        transform: translateY(0px) !important;
        box-shadow: 0 2px 8px rgba(15, 118, 110, 0.3) !important;
    }
    .hero-card{
        max-width: 760px; margin: 32px auto; padding: 28px 32px;
        background:#fff; border-radius:18px; box-shadow:0 10px 30px rgba(2,6,23,.08);
    }
    .hero-badge{display:inline-block; font-size:12px; padding:6px 10px; border-radius:999px;
        background:#e6f7f5; color:#065f5b; border:1px solid #b7ece7; margin-bottom:12px;}
    .hero-title{font-size:36px; font-weight:800; margin:6px 0 10px 0;}
    .hero-sub{color:#334155; margin-bottom:24px;}
    .hero-chart{width:100%; height:140px;}
    h1 { 
        font-weight: 700 !important;
        color: #0f172a !important;
        margin-bottom: 0.5rem !important;
    }
    h2 { 
        font-weight: 600 !important;
        color: #1e293b !important;
    }
    .stSelectbox > div > div {
        border-radius: 10px !important;
        transition: all 0.2s ease !important;
    }
    .stSelectbox > div > div:hover {
        border-color: #0f766e !important;
    }
    .stSlider > div > div {
        border-radius: 10px !important;
    }
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

    # columnas requeridas, incluyendo mod_compras / mod_ventas
    for col in [
        "tenant_id","tenant_name","sheet_id",
        "webhook_s1","webhook_s2","webhook_s3",
        "kame_client_id","kame_client_secret",
        "use_stock_total","login_email","login_pin","is_active",
        "mod_compras","mod_ventas",
    ]:
        if col not in d.columns:
            if col in ("use_stock_total", "is_active"):
                d[col] = True
            elif col == "mod_compras":
                # por defecto: módulo Compras habilitado
                d[col] = True
            elif col == "mod_ventas":
                # por defecto: módulo Ventas deshabilitado
                d[col] = False
            else:
                d[col] = ""

    # normalizamos columnas string
    for col in [
        "tenant_id","tenant_name","sheet_id",
        "webhook_s1","webhook_s2","webhook_s3",
        "kame_client_id","kame_client_secret",
        "login_email","login_pin",
    ]:
        if col in d.columns:
            d[col] = d[col].astype(str).map(_clean)

    if "login_email" in d.columns:
        d["login_email"] = d["login_email"].str.lower()

    d["tenant_name"] = d["tenant_name"].replace({pd.NA:"", None:""}).astype(str)
    d.loc[d["tenant_id"].eq("") | d["tenant_id"].isna(), "tenant_id"] = d["tenant_name"].apply(slugify)

    # booleanos base
    d["is_active"] = d["is_active"].apply(_truthy)
    d["use_stock_total"] = d["use_stock_total"].apply(_truthy)

    # mod_compras: vacío => True; texto truthy => True; otro => False
    raw_mc = d["mod_compras"].astype(str).replace("\u00a0", " ").str.strip()
    d["mod_compras"] = raw_mc.apply(lambda x: True if x == "" else _truthy(x))

    # mod_ventas: vacío => False; texto truthy => True; otro => False
    raw_mv = d["mod_ventas"].astype(str).replace("\u00a0", " ").str.strip()
    d["mod_ventas"] = raw_mv.apply(lambda x: False if x == "" else _truthy(x))

    d["tenant_name"] = d["tenant_name"].astype(str)
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
    
    # IMPORTANTE: Google Sheets a veces no exporta correctamente valores con corchetes [2A] cuando están en formato "Automático"
    # Solución: Leer el CSV directamente con requests para ver qué está devolviendo realmente
    # Agregamos un parámetro de tiempo para evitar cache del servidor/proxy
    url = (f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?"
           f"tqx=out:csv&sheet={sheet_param}&t={int(time.time())}")
    
    try:
        # Intentar leer directamente con requests para debug
        response = requests.get(url, timeout=30)
        response.encoding = 'utf-8'
        csv_content = response.text
        
        # Leer desde el contenido CSV en memoria
        from io import StringIO
        df = pd.read_csv(
            StringIO(csv_content),
            dtype=str, 
            na_filter=False, 
            keep_default_na=False,
            on_bad_lines='skip',
            engine='python'
        )
    except Exception as e:
        # Fallback: método directo si falla
        df = pd.read_csv(
            url, 
            dtype=str, 
            na_filter=False, 
            keep_default_na=False,
            encoding='utf-8',
            on_bad_lines='skip',
            engine='python'
        )
    
    # Limpiar: reemplazar representaciones de NaN/None por string vacío
    df = df.replace(['nan', 'NaN', 'None', '<NA>', 'NaT', 'null', 'NULL'], '')
    
    return df

@st.cache_data(ttl=300)
def load_clientes_config() -> Optional[pd.DataFrame]:
    """Intenta leer la pestaña clientes_config del sheet por defecto."""
    try:
        df = read_gsheets(DEFAULT_SHEET_ID, TAB_CLIENTES_CONF)
        df = ensure_clientes_columns(df)
        return df[df["is_active"] == True].reset_index(drop=True)
    except Exception:
        return None

@st.cache_data(ttl=300)
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
    # Normalizar nombres de columnas: eliminar espacios y caracteres invisibles
    cols_normalized = {str(c).replace("\u00a0", " ").strip().lower(): c for c in df.columns}
    fecha_col = cols_normalized.get("fecha")
    sku_col   = cols_normalized.get("sku")
    qty_col   = cols_normalized.get("cantidad") or cols_normalized.get("qty")
    if not fecha_col or not sku_col or not qty_col:
        raise ValueError("ventas_raw debe tener columnas 'fecha', 'sku' y 'cantidad'.")
    out = pd.DataFrame()
    out["fecha"] = pd.to_datetime(df[fecha_col], errors="coerce")
    # Convertir SKU a string, manejando números y valores NaN correctamente
    # Si el SKU es numérico (ej: 3010100111114.0), convertirlo a string sin el .0
    # Si es NaN, convertirlo a string vacío y luego filtrarlo
    sku_series = df[sku_col].astype(str)
    # Reemplazar representaciones de NaN/None por string vacío
    sku_series = sku_series.replace(['nan', 'NaN', 'None', '<NA>', 'NaT'], '')
    # Si el SKU es un número con .0 al final (ej: "3010100111114.0"), remover el .0
    sku_series = sku_series.str.replace(r'\.0$', '', regex=True)
    out["sku"] = sku_series.str.strip().str.upper()
    out["qty"] = pd.to_numeric(df[qty_col], errors="coerce").fillna(0)
    # Filtrar filas con SKU vacío o fecha inválida
    out = out[(out["sku"] != "") & (out["fecha"].notna())]
    return out

# === NUEVO: normalizador extendido para Módulo Ventas ===
def normalize_ventas_for_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizador extendido para el Módulo de Ventas.
    NO se usa en el core de Compras, por lo que no altera el comportamiento actual.
    """
    base_cols = [
        "fecha", "sku", "qty", "venta_neta", "precio_unitario",
        "rut", "razon_social", "tipo_documento", "folio", "glosa",
        "sucursal", "unidad_negocio", "familia", "vendedor",
        "lista_precio", "producto",
    ]

    if df is None or df.empty:
        return pd.DataFrame(columns=base_cols)

    # Normalizar nombres de columnas: eliminar espacios y caracteres invisibles
    cols_lc = {str(c).replace("\u00a0", " ").strip().lower(): c for c in df.columns}

    def _col(*candidates):
        for key in candidates:
            key_lc = key.lower().strip()
            if key_lc in cols_lc:
                return cols_lc[key_lc]
        return None

    fecha_col = _col("fecha")
    sku_col   = _col("sku", "sku2")
    qty_col   = _col("cantidad", "qty", "unidades")
    # monto de la línea: intentamos varios nombres
    venta_col = _col("venta_neta", "total_linea", "total línea", "total_line")

    if not fecha_col or not sku_col or not qty_col:
        # si faltan columnas mínimas, devolvemos un DF vacío "bien formado"
        return pd.DataFrame(columns=base_cols)

    out = pd.DataFrame()
    out["fecha"] = pd.to_datetime(df[fecha_col], errors="coerce")
    out["sku"]   = df[sku_col].astype(str).str.strip().str.upper()
    out["qty"]   = pd.to_numeric(df[qty_col], errors="coerce").fillna(0)

    if venta_col:
        out["venta_neta"] = pd.to_numeric(df[venta_col], errors="coerce").fillna(0.0)
    else:
        out["venta_neta"] = 0.0

    precio_col = _col("precio_unitario", "precio un.", "precio_un", "precio unitario")
    if precio_col:
        out["precio_unitario"] = pd.to_numeric(df[precio_col], errors="coerce").fillna(0.0)
    else:
        out["precio_unitario"] = 0.0

    text_map = {
        "rut": ["rut"],
        "razon_social": ["razon_social", "razón social"],
        "tipo_documento": ["tipo_documento", "documento"],
        "folio": ["folio"],
        "glosa": ["glosa"],
        "sucursal": ["sucursal"],
        "unidad_negocio": ["unidad_negocio", "unidad de negocio"],
        "familia": ["familia"],
        "vendedor": ["vendedor"],
        "lista_precio": ["lista_precio", "lista precio"],
        "producto": ["producto", "descripcion", "descripción"],
    }

    for logical, candidates in text_map.items():
        c = _col(*candidates)
        if c:
            out[logical] = df[c].astype(str).fillna("").str.strip()
        else:
            out[logical] = ""

    out = out.dropna(subset=["fecha", "sku"])
    return out
# === FIN normalizador ventas ===

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
# HELPERS PARA GRÁFICOS
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

# HENRY: helper para calcular cooldown dinámico basado en cantidad de SKUs
def calculate_dynamic_cooldown(update_type: str, default_cooldown: int) -> int:
    """
    Calcula un cooldown dinámico basado en la cantidad de SKUs del snapshot.
    
    Relación observada:
    - Hasta 500 SKUs: 2-3 minutos (180 segundos)
    - 1000 SKUs: ~7 minutos (420 segundos)
    
    Args:
        update_type: Tipo de actualización ("S1" o "S2")
        default_cooldown: Cooldown por defecto si no hay snapshot o SKUs
    
    Returns:
        Segundos de cooldown calculados
    """
    snapshot = st.session_state.get(f"snapshot_{update_type}")
    if not snapshot or snapshot.get("unique_skus") is None:
        return default_cooldown
    
    sku_count = snapshot.get("unique_skus", 0)
    
    # Fórmula basada en observaciones:
    # - 500 SKUs → 180 segundos (3 minutos)
    # - 1000 SKUs → 420 segundos (7 minutos)
    # - Interpolación lineal con mínimo de 120 segundos (2 minutos)
    
    if sku_count <= 500:
        # Hasta 500 SKUs: usar 3 minutos (180 segundos)
        return 180
    elif sku_count >= 1000:
        # 1000+ SKUs: usar 7 minutos (420 segundos)
        return 420
    else:
        # Entre 500 y 1000 SKUs: interpolación lineal
        # Pendiente: (420 - 180) / (1000 - 500) = 0.48 segundos por SKU adicional
        cooldown = 180 + (sku_count - 500) * 0.48
        return int(cooldown)

# HENRY: helper para manejar el cooldown del botón de predicción
def start_cooldown(seconds: int):
    """Marca en sesión un tiempo mínimo antes de ejecutar la predicción."""
    if seconds <= 0:
        return
    now_ts = time.time()
    current = st.session_state.get("cooldown_until", 0)
    st.session_state["cooldown_until"] = max(current, now_ts + seconds)

# HENRY: helpers para detectar fallos en actualizaciones
def _calculate_data_hash(df: pd.DataFrame, columns: list = None) -> str:
    """
    Calcula un hash MD5 de los datos para detectar cambios precisos.
    Si se especifican columnas, solo usa esas columnas.
    """
    if df is None or df.empty:
        return hashlib.md5(b"empty").hexdigest()
    
    # Si se especifican columnas, usar solo esas
    if columns:
        available_cols = [c for c in columns if c in df.columns]
        if available_cols:
            df_to_hash = df[available_cols].copy()
        else:
            df_to_hash = df.copy()
    else:
        df_to_hash = df.copy()
    
    # Convertir a string y calcular hash
    # Usar valores ordenados para que el hash sea consistente
    try:
        # Ordenar por todas las columnas para consistencia
        df_sorted = df_to_hash.sort_values(by=list(df_to_hash.columns)).reset_index(drop=True)
        data_str = df_sorted.to_csv(index=False).encode('utf-8')
    except Exception:
        # Fallback: convertir a string sin ordenar
        data_str = df_to_hash.to_csv(index=False).encode('utf-8')
    
    return hashlib.md5(data_str).hexdigest()

def _get_last_update_timestamp(sheet_id: str, update_type: str) -> Optional[pd.Timestamp]:
    """
    Lee un timestamp de última actualización desde el sheet DEL TENANT ACTUAL.
    
    SEGURIDAD MULTI-TENANT CRÍTICA:
    - Esta función SIEMPRE lee del sheet específico del tenant (sheet_id)
    - Cada tenant tiene su propio Google Sheet y su propia pestaña 'config'
    
    REQUISITOS DETERMINÍSTICOS:
    - S1 (Ventas) → config!H2 (columna index 7)
    - S2 (Stock)  → config!I2 (columna index 8)
    - S3 (Inbound)→ config!K2 (columna index 10)
    
    Args:
        sheet_id: ID del Google Sheet del TENANT ACTUAL
        update_type: Tipo de actualización ("S1", "S2" o "S3")
    """
    if not sheet_id or sheet_id.strip() == "":
        return None
    
    try:
        # Leer pestaña config
        config_df = read_gsheets(sheet_id, TAB_CONFIG)
        if config_df.empty:
            return None
            
        # Determinar columna según el tipo (0-indexed, Row 2 del sheet = Row 0 del DF)
        col_idx = -1
        search_name = ""
        if update_type == "S1":
            col_idx = 7  # Columna H
            search_name = "last_update_s1"
        elif update_type == "S2":
            col_idx = 8  # Columna I
            search_name = "last_update_s2"
        elif update_type == "S3":
            col_idx = 10 # Columna K
            search_name = "last_update_s3"
            
        ts_found = None
        
        # 1. Intentar por índice fijo (REQUISITO EXPLÍCITO)
        # HENRY: Buscamos en toda la columna (no solo fila 1) para encontrar el más reciente
        if col_idx != -1 and len(config_df.columns) > col_idx:
            series = pd.to_datetime(config_df.iloc[:, col_idx], errors="coerce")
            ts_found = series.max()
        
        # 2. Fallback por nombre si el índice no funcionó o no encontró nada
        if ts_found is None and search_name:
            cols_lc = {str(c).lower().strip(): c for c in config_df.columns}
            if search_name in cols_lc:
                series = pd.to_datetime(config_df[cols_lc[search_name]], errors="coerce")
                ts_found = series.max()
        
        # 3. Otros fallbacks genéricos
        if ts_found is None:
            for col_name in ["ultima_actualizacion", "timestamp"]:
                cols_lc = {str(c).lower().strip(): c for c in config_df.columns}
                if col_name in cols_lc:
                    series = pd.to_datetime(config_df[cols_lc[col_name]], errors="coerce")
                    ts_found = series.max()
                    if ts_found is not None:
                        break

        if ts_found is not None and pd.notna(ts_found):
            # HENRY: Asegurar que el timestamp retornado sea consciente de la zona horaria de Chile
            # Si es naive (lo normal desde Sheets), lo localizamos. Si ya tiene TZ, lo convertimos.
            try:
                tz_chile = "America/Santiago"
                if ts_found.tzinfo is None:
                    return ts_found.tz_localize(tz_chile, ambiguous='infer')
                else:
                    return ts_found.tz_convert(tz_chile)
            except Exception:
                return ts_found # Fallback al original si falla la localización
                    
    except Exception:
        pass
        
    return None

def _get_update_status(sheet_id: str, update_type: str) -> Optional[str]:
    """
    Lee el estado de actualización (STATUS_FINAL) desde el sheet DEL TENANT ACTUAL.
    
    SEGURIDAD MULTI-TENANT CRÍTICA:
    - Esta función SIEMPRE lee del sheet específico del tenant (sheet_id)
    - NUNCA lee de un sheet global o de otro tenant
    
    Make.com debe escribir el estado en la pestaña 'config' del sheet del tenant:
    - "RUNNING" cuando comienza la actualización
    - "DONE" cuando realmente termina
    
    Args:
        sheet_id: ID del Google Sheet del TENANT ACTUAL (debe venir de CURRENT_SHEET_ID)
        update_type: Tipo de actualización ("S1", "S2" o "S3")
    
    Busca en la pestaña 'config' del sheet del tenant:
    - Columna: 'STATUS_FINAL', 'status_final', 'status', 'estado'
    
    Retorna:
    - "RUNNING" si está en ejecución
    - "DONE" si terminó
    - None si no existe o hay error
    """
    # Validación de seguridad: asegurar que sheet_id no esté vacío
    if not sheet_id or sheet_id.strip() == "":
        return None
    
    try:
        config_df = read_gsheets(sheet_id, TAB_CONFIG)
        if not config_df.empty:
            cols_lc = {str(c).lower(): c for c in config_df.columns}
            status_col = None
            
            # Buscar columnas de estado
            for col_name in ["status_final", "status", "estado", "estado_final"]:
                if col_name in cols_lc:
                    status_col = cols_lc[col_name]
                    break
            
            if status_col and len(config_df) > 0:
                status_str = str(config_df.iloc[0][status_col]).strip().upper()
                if status_str in ["RUNNING", "DONE"]:
                    return status_str
                # También aceptar variaciones
                if "RUNNING" in status_str or "EJECUTANDO" in status_str:
                    return "RUNNING"
                if "DONE" in status_str or "COMPLETO" in status_str or "TERMINADO" in status_str:
                    return "DONE"
    except Exception:
        pass
    
    return None

def _get_elapsed_time_info(ts: Optional[pd.Timestamp]) -> str:
    """
    Calcula el tiempo transcurrido desde un timestamp hasta ahora,
    forzando el uso de la zona horaria de Chile (America/Santiago).
    
    Esta función está diseñada para ser consistente tanto en local 
    como en producción (Render/UTC).
    """
    if ts is None or pd.isna(ts):
        return "No disponible"
    
    try:
        tz_chile = "America/Santiago"
        
        # 1. Asegurar objeto Timestamp
        ts = pd.Timestamp(ts)
        
        # 2. Localizar timestamp del sheet (si es naive, es Chile; si aware, convertir a Chile)
        if ts.tzinfo is None:
            ts_chile_aware = ts.tz_localize(tz_chile, ambiguous='infer')
        else:
            ts_chile_aware = ts.tz_convert(tz_chile)
            
        # 3. Obtener "ahora" explícitamente en Chile
        ahora_chile = pd.Timestamp.now(tz=tz_chile)
        
        # 4. Calcular diferencia
        diff = ahora_chile - ts_chile_aware
        total_seconds = diff.total_seconds()
        
        # Manejo de casos de borde (relojes ligeramente desincronizados)
        if total_seconds < -60:
            return "recién actualizado"
        elif total_seconds < 0:
            return "hace unos momentos"
            
        horas = int(total_seconds // 3600)
        minutos = int((total_seconds % 3600) // 60)
        
        if horas > 0:
            return f"{horas}h {minutos}m"
        elif minutos > 0:
            return f"{minutos}m"
        else:
            return "hace unos momentos"
            
    except Exception:
        # Fallback ultra-seguro: intentar resta naive asumiendo que ambos son la misma zona
        try:
            # Si falló lo anterior, intentamos una comparación sin zonas horarias
            # para evitar el desfase de horas del servidor
            ts_naive = ts.replace(tzinfo=None) if hasattr(ts, 'replace') else ts
            # En Render, pd.Timestamp.now() es UTC. Si el sheet es Chile (UTC-3), 
            # la resta naive daría error de ~3h.
            # Por eso, para el fallback, tratamos de forzar la resta local.
            import datetime
            import pytz
            chile_tz = pytz.timezone("America/Santiago")
            now_chile = datetime.datetime.now(chile_tz).replace(tzinfo=None)
            
            diff = now_chile - ts_naive
            total_seconds = diff.total_seconds()
            horas = int(total_seconds // 3600)
            minutos = int((total_seconds % 3600) // 60)
            if horas > 0: return f"{horas}h {minutos}m"
            if minutos > 0: return f"{minutos}m"
            return "hace unos momentos"
        except:
            return "No disponible"

# MODO BATCH: Función para verificar si el batch está listo (STATUS_FINAL == "DONE" y timestamp estable)
def check_batch_ready(sheet_id: str) -> dict:
    """
    Verifica si el batch está listo para ejecutar predicción.
    
    Requisitos:
    1. STATUS_FINAL == "DONE"
    2. Timestamp de stock no cambia en 4 lecturas consecutivas
    
    Retorna:
    - ready: bool
    - message: str (mensaje para mostrar al usuario)
    - status: str ("DONE", "RUNNING", "UNKNOWN")
    """
    if not sheet_id or not sheet_id.strip():
        return {
            "ready": False,
            "message": "❌ Sheet ID no configurado",
            "status": "UNKNOWN"
        }
    
    # Si ya confirmamos que el batch está listo, verificar rápidamente que sigue listo
    if st.session_state.get("batch_ready_confirmed", False):
        current_timestamp = _get_last_update_timestamp(sheet_id, "S2")
        confirmed_timestamp = st.session_state.get("batch_ready_timestamp")
        
        # Si el timestamp es None (batch reiniciado) o cambió, resetear confirmación
        if current_timestamp is None or current_timestamp != confirmed_timestamp:
            st.session_state["batch_ready_confirmed"] = False
            st.session_state["batch_ready_timestamp"] = None
            st.session_state["batch_timestamp_readings"] = []
        else:
            # El timestamp sigue siendo el mismo, verificar que el status siga siendo DONE
            batch_status = _get_update_status(sheet_id, "S2")
            if batch_status == "DONE":
                return {
                    "ready": True,
                    "message": "✅ Batch completado y verificado. Puedes ejecutar la predicción.",
                    "status": "DONE"
                }
            else:
                # El status cambió (ej: a RUNNING), resetear
                st.session_state["batch_ready_confirmed"] = False
                st.session_state["batch_ready_timestamp"] = None
                st.session_state["batch_timestamp_readings"] = []

    # Verificar STATUS_FINAL
    batch_status = _get_update_status(sheet_id, "S2")  # Usamos S2 para verificar el batch completo
    
    if batch_status != "DONE":
        if batch_status == "RUNNING":
            return {
                "ready": False,
                "message": "⏳ Actualización batch en curso (STATUS: RUNNING). Espera a que termine.",
                "status": "RUNNING"
            }
        else:
            return {
                "ready": False,
                "message": "⏳ Actualización batch en curso o no confirmada. STATUS_FINAL no disponible.",
                "status": "UNKNOWN"
            }
    
    # STATUS_FINAL == "DONE", ahora verificar estabilización del timestamp
    # Leer timestamp actual de stock
    current_timestamp = _get_last_update_timestamp(sheet_id, "S2")
    
    if current_timestamp is None:
        return {
            "ready": False,
            "message": "⏳ Actualización batch en curso. Timestamp de stock no disponible.",
            "status": "DONE"
        }
    
    # Guardar lecturas en session_state para verificar estabilización
    timestamp_readings_key = "batch_timestamp_readings"
    if timestamp_readings_key not in st.session_state:
        st.session_state[timestamp_readings_key] = []
    
    readings = st.session_state[timestamp_readings_key]
    
    # Limpiar lecturas muy antiguas (más de 1 minuto)
    now = time.time()
    readings = [r for r in readings if (now - r["time"]) < 60]
    st.session_state[timestamp_readings_key] = readings
    
    # Solo agregar una nueva lectura si han pasado al menos 3 segundos desde la última
    should_add_reading = True
    if len(readings) > 0:
        last_reading_time = readings[-1]["time"]
        if (now - last_reading_time) < 3:
            should_add_reading = False
    
    if should_add_reading:
        # Agregar lectura actual
        readings.append({
            "timestamp": current_timestamp,
            "time": now
        })
        st.session_state[timestamp_readings_key] = readings
    
    # Mantener solo las últimas 4 lecturas
    if len(readings) > 4:
        readings = readings[-4:]
        st.session_state[timestamp_readings_key] = readings
    
    # Verificar si las últimas 4 lecturas son iguales
    if len(readings) >= 4:
        # Verificar que todos los timestamps sean iguales
        first_ts = readings[0]["timestamp"]
        all_same = all(r["timestamp"] == first_ts for r in readings)
        
        if all_same:
            # Timestamp estabilizado, listo para ejecutar
            # Guardar confirmación en session_state para evitar revalidar cuando se presiona el botón
            st.session_state["batch_ready_confirmed"] = True
            st.session_state["batch_ready_timestamp"] = current_timestamp
            return {
                "ready": True,
                "message": "✅ Batch completado y verificado. Puedes ejecutar la predicción.",
                "status": "DONE"
            }
        else:
            # Timestamp aún está cambiando, reiniciar lecturas
            st.session_state[timestamp_readings_key] = [readings[-1]]  # Mantener solo la última
            return {
                "ready": False,
                "message": "⏳ Actualización batch en curso. Timestamp de stock aún se está actualizando.",
                "status": "DONE"
            }
    else:
        # Aún no tenemos 4 lecturas, esperar más
        return {
            "ready": False,
            "message": f"⏳ Verificando estabilización del timestamp ({len(readings)}/4 lecturas)...",
            "status": "DONE",
            "needs_rerun": len(readings) > 0 and should_add_reading  # Indicar si necesita rerun
        }

def save_data_snapshot(sheet_id: str, update_type: str):
    """
    Guarda un snapshot de los datos antes de disparar una actualización.
    
    SEGURIDAD MULTI-TENANT: Esta función SIEMPRE debe recibir el sheet_id del tenant actual.
    NUNCA debe leer de un sheet global o de otro tenant.
    
    Args:
        sheet_id: ID del Google Sheet del TENANT ACTUAL (debe venir de CURRENT_SHEET_ID)
        update_type: Tipo de actualización ("S1" o "S2")
    """
    # Validación de seguridad: asegurar que sheet_id no esté vacío
    if not sheet_id or sheet_id.strip() == "":
        st.error(f"❌ **ERROR DE SEGURIDAD**: sheet_id está vacío para {update_type}. No se puede guardar snapshot.")
        return
    
    try:
        if update_type == "S1":  # Ventas
            ventas_raw = read_gsheets(sheet_id, TAB_VENTAS)
            ventas = normalize_ventas_sheet(ventas_raw)
            # Guardar estado inicial de STATUS_FINAL antes de disparar
            initial_status = _get_update_status(sheet_id, update_type)
            snapshot = {
                "timestamp": time.time(),
                "row_count": len(ventas_raw),
                "last_date": None,
                "total_qty": 0.0,
                "data_hash": None,  # NUEVO: hash de los datos
                "unique_skus": None,  # NUEVO: número de SKUs únicos para detectar actualizaciones incompletas
                "initial_status": initial_status,  # Estado inicial de STATUS_FINAL antes de disparar
            }
            if not ventas.empty and "fecha" in ventas.columns:
                snapshot["last_date"] = ventas["fecha"].max()
                snapshot["total_qty"] = float(ventas["qty"].sum())
                snapshot["unique_skus"] = int(ventas["sku"].nunique())  # Guardar número de SKUs únicos
                # Calcular hash de los datos (usar columnas clave: fecha, sku, qty)
                snapshot["data_hash"] = _calculate_data_hash(ventas, ["fecha", "sku", "qty"])
            elif not ventas_raw.empty:
                # Si hay datos pero no se pudieron normalizar, calcular hash del raw
                snapshot["data_hash"] = _calculate_data_hash(ventas_raw)
            st.session_state[f"snapshot_{update_type}"] = snapshot
        elif update_type == "S2":  # Stock
            stock_raw = read_gsheets(sheet_id, TAB_STOCK)
            stock = normalize_stock_sheet(stock_raw)
            # Guardar estado inicial de STATUS_FINAL antes de disparar
            initial_status = _get_update_status(sheet_id, update_type)
            snapshot = {
                "timestamp": time.time(),
                "row_count": len(stock_raw),
                "total_stock": 0.0,
                "data_hash": None,  # NUEVO: hash de los datos
                "unique_skus": None,  # NUEVO: número de SKUs únicos para detectar actualizaciones incompletas
                "initial_status": initial_status,  # Estado inicial de STATUS_FINAL antes de disparar
            }
            if not stock.empty and "stock" in stock.columns:
                snapshot["total_stock"] = float(stock["stock"].sum())
                snapshot["unique_skus"] = int(stock["sku"].nunique())  # Guardar número de SKUs únicos
                # Calcular hash de los datos (usar columnas clave: sku, stock)
                snapshot["data_hash"] = _calculate_data_hash(stock, ["sku", "stock"])
            elif not stock_raw.empty:
                # Si hay datos pero no se pudieron normalizar, calcular hash del raw
                snapshot["data_hash"] = _calculate_data_hash(stock_raw)
            st.session_state[f"snapshot_{update_type}"] = snapshot
    except Exception as e:
        # Si falla, guardamos un snapshot mínimo
        st.session_state[f"snapshot_{update_type}"] = {
            "timestamp": time.time(),
            "error": str(e),
        }

def check_update_success(sheet_id: str, update_type: str, cooldown_seconds: int) -> dict:
    """
    Verifica si una actualización fue exitosa comparando con el snapshot anterior.
    
    SEGURIDAD MULTI-TENANT: Esta función SIEMPRE debe recibir el sheet_id del tenant actual.
    NUNCA debe leer de un sheet global o de otro tenant.
    
    Usa múltiples métodos:
    1. Hash de datos (más preciso)
    2. Timestamp de última actualización (si existe en el sheet del tenant)
    3. Comparación de métricas (fallback)
    
    Args:
        sheet_id: ID del Google Sheet del TENANT ACTUAL (debe venir de CURRENT_SHEET_ID)
        update_type: Tipo de actualización ("S1" o "S2")
        cooldown_seconds: Segundos de cooldown esperados
    
    Retorna un dict con:
    - success: bool (True si parece exitosa)
    - warnings: list de mensajes de advertencia
    - errors: list de mensajes de error
    """
    # Validación de seguridad: asegurar que sheet_id no esté vacío
    if not sheet_id or sheet_id.strip() == "":
        return {
            "success": False,
            "warnings": [],
            "errors": [f"❌ **ERROR DE SEGURIDAD**: sheet_id está vacío para {update_type}. No se puede verificar actualización."],
        }
    
    result = {
        "success": True,
        "warnings": [],
        "errors": [],
    }
    
    snapshot = st.session_state.get(f"snapshot_{update_type}")
    if not snapshot:
        result["warnings"].append("No se pudo verificar la actualización (no hay snapshot previo).")
        return result
    
    # Si el snapshot tiene un error, no podemos verificar
    if "error" in snapshot:
        result["warnings"].append(f"No se pudo crear snapshot previo: {snapshot.get('error')}. La verificación se omitirá.")
        return result
    
    try:
        elapsed = time.time() - snapshot.get("timestamp", time.time())
        snapshot_time = snapshot.get("timestamp", time.time())
        
        # PRIORIDAD 1: Verificar STATUS_FINAL (más confiable que timestamp)
        # Make.com escribe "RUNNING" cuando comienza y "DONE" cuando realmente termina
        # IMPORTANTE: Solo considerar "DONE" si el estado cambió desde el estado inicial
        initial_status = snapshot.get("initial_status")
        update_status = _get_update_status(sheet_id, update_type)
        
        if update_status == "DONE":
            # Solo considerar completado si el estado cambió desde el inicial
            # Si el estado inicial ya era "DONE", significa que es de una ejecución anterior
            if initial_status != "DONE":
                # El estado cambió a "DONE", pero verificar que el timestamp se haya estabilizado
                # Si el timestamp sigue cambiando, Make.com aún está ejecutando
                last_update_ts = _get_last_update_timestamp(sheet_id, update_type)
                
                # Guardar el timestamp cuando detectamos "DONE" por primera vez
                done_timestamp_key = f"done_timestamp_{update_type}"
                previous_done_timestamp = st.session_state.get(done_timestamp_key)
                
                if last_update_ts is not None:
                    if previous_done_timestamp is None:
                        # Primera vez que detectamos "DONE", guardar el timestamp
                        st.session_state[done_timestamp_key] = last_update_ts
                        result["warnings"].append(
                            "⏳ Make.com reporta 'DONE', pero verificando que el timestamp se haya estabilizado. "
                            "Esperando 15 segundos para confirmar..."
                        )
                        # No retornar todavía, continuar verificando
                    else:
                        # Ya habíamos detectado "DONE" antes, verificar si el timestamp cambió
                        time_since_previous = (last_update_ts - previous_done_timestamp).total_seconds()
                        
                        if time_since_previous < 15:
                            # El timestamp cambió recientemente (menos de 15 segundos), Make.com aún está ejecutando
                            result["warnings"].append(
                                f"⏳ Make.com reporta 'DONE', pero el timestamp sigue actualizándose "
                                f"(último cambio hace {time_since_previous:.1f}s). "
                                "Esperando a que se estabilice..."
                            )
                            # Actualizar el timestamp guardado
                            st.session_state[done_timestamp_key] = last_update_ts
                            # No retornar todavía, continuar verificando
                        else:
                            # El timestamp se estabilizó (no cambió en los últimos 15 segundos), realmente terminó
                            result["success"] = True
                            # Limpiar el timestamp guardado
                            if done_timestamp_key in st.session_state:
                                del st.session_state[done_timestamp_key]
                            # Detener el cooldown ya que Make.com realmente terminó
                            st.session_state["cooldown_until"] = 0
                            return result
                else:
                    # No hay timestamp disponible, confiar en el estado "DONE" pero esperar un poco más
                    if previous_done_timestamp is None:
                        # Primera vez que detectamos "DONE", guardar el tiempo actual
                        st.session_state[done_timestamp_key] = time.time()
                        result["warnings"].append(
                            "⏳ Make.com reporta 'DONE', pero no hay timestamp disponible. "
                            "Esperando 15 segundos para confirmar..."
                        )
                    else:
                        # Verificar si pasaron al menos 15 segundos desde que detectamos "DONE"
                        time_since_done = time.time() - previous_done_timestamp
                        if time_since_done >= 15:
                            # Pasaron 15 segundos, considerar completado
                            result["success"] = True
                            if done_timestamp_key in st.session_state:
                                del st.session_state[done_timestamp_key]
                            st.session_state["cooldown_until"] = 0
                            return result
            else:
                # El estado ya era "DONE" antes de presionar el botón
                # Esto es normal al inicio: Make.com puede tardar unos segundos en cambiar a "RUNNING"
                # Solo mostrar warning si ha pasado suficiente tiempo (más de 15 segundos)
                if elapsed > 15:
                    result["warnings"].append(
                        "ℹ️ El estado STATUS_FINAL aún muestra 'DONE' de una ejecución anterior. "
                        "Make.com debería cambiar a 'RUNNING' pronto. Verificando datos..."
                    )
                # Si es muy reciente (menos de 15 segundos), no mostrar warning, es normal
        elif update_status == "RUNNING":
            # Make.com está ejecutando, esperar a que termine
            result["warnings"].append(
                "⏳ Make.com está ejecutando la actualización (STATUS: RUNNING). "
                "Esperando a que termine antes de verificar los datos."
            )
            # No retornar todavía, continuar con otras verificaciones pero no marcar como exitoso
        
        # MEJORA #2: Verificar timestamp de última actualización desde el sheet (si Make.com lo escribe)
        last_update_ts = _get_last_update_timestamp(sheet_id, update_type)
        if last_update_ts is not None:
            # Si existe timestamp en el sheet, verificar que sea más reciente que el snapshot
            snapshot_datetime = pd.Timestamp.fromtimestamp(snapshot_time)
            if last_update_ts > snapshot_datetime:
                # La actualización fue exitosa según el timestamp del sheet
                result["success"] = True
                # No agregar warnings adicionales si el timestamp confirma éxito
                return result
            else:
                # El timestamp no es más reciente, puede haber un problema
                result["warnings"].append(
                    f"⚠️ El timestamp de última actualización en el sheet ({last_update_ts.strftime('%Y-%m-%d %H:%M:%S')}) "
                    f"no es más reciente que cuando se inició la actualización. "
                    f"Es posible que la actualización no se haya completado."
                )
        
        if update_type == "S1":  # Ventas
            ventas_raw = read_gsheets(sheet_id, TAB_VENTAS)
            ventas = normalize_ventas_sheet(ventas_raw)
            
            # Definir variables de conteo ANTES de usarlas
            current_rows = len(ventas_raw)
            previous_rows = snapshot.get("row_count", 0)
            
            # MEJORA #2: Comparar hash de datos (más preciso que comparar totales)
            previous_hash = snapshot.get("data_hash")
            if previous_hash:
                if not ventas.empty:
                    current_hash = _calculate_data_hash(ventas, ["fecha", "sku", "qty"])
                elif not ventas_raw.empty:
                    current_hash = _calculate_data_hash(ventas_raw)
                else:
                    current_hash = None
                
                if current_hash == previous_hash:
                    # Los datos son idénticos
                    if elapsed > cooldown_seconds * 0.8:  # Pasó al menos 80% del cooldown
                        result["warnings"].append(
                            "⚠️ Los datos de ventas son idénticos a antes de la actualización (verificado por hash). "
                            "Es muy probable que la actualización no se haya completado correctamente."
                        )
                else:
                    # Los datos cambiaron, pero verificar si es una actualización completa o parcial
                    # Si hay menos filas que antes, podría ser una actualización incompleta
                    if previous_rows > 0 and current_rows < previous_rows:
                        reduction_pct = ((previous_rows - current_rows) / previous_rows) * 100
                        if reduction_pct > 5.0:
                            result["warnings"].append(
                                f"⚠️ **ATENCIÓN**: Los datos cambiaron (hash diferente), pero el número de filas "
                                f"disminuyó desde {previous_rows} a {current_rows} ({reduction_pct:.1f}% de reducción). "
                                f"Esto podría indicar una actualización parcial o incompleta (ej: error 429 en Make.com)."
                            )
                    else:
                        # Los datos cambiaron y no hay reducción significativa, actualización probablemente exitosa
                        result["success"] = True
            
            # Verificar cambios en número de filas
            
            # DETECCIÓN DE ACTUALIZACIÓN INCOMPLETA: Reducción significativa de filas
            # Si hay menos filas después de una actualización, podría ser un problema
            if previous_rows > 0 and current_rows < previous_rows:
                reduction_pct = ((previous_rows - current_rows) / previous_rows) * 100
                # Si se redujo más del 5%, podría ser una actualización incompleta
                if reduction_pct > 5.0 and elapsed > cooldown_seconds * 0.8:
                    result["warnings"].append(
                        f"⚠️ **POSIBLE ACTUALIZACIÓN INCOMPLETA**: El número de filas disminuyó significativamente "
                        f"desde {previous_rows} a {current_rows} ({reduction_pct:.1f}% de reducción). "
                        f"Esto podría indicar que la actualización se detuvo antes de completarse (ej: error 429 en Make.com)."
                    )
            
            # Verificar número de SKUs únicos (otra señal de actualización incompleta)
            if not ventas.empty and "sku" in ventas.columns:
                current_skus = ventas["sku"].nunique()
                # Si tenemos el snapshot anterior con SKUs, comparar
                previous_skus = snapshot.get("unique_skus")
                if previous_skus and current_skus < previous_skus:
                    reduction_skus = ((previous_skus - current_skus) / previous_skus) * 100
                    if reduction_skus > 5.0 and elapsed > cooldown_seconds * 0.8:
                        result["warnings"].append(
                            f"⚠️ **POSIBLE ACTUALIZACIÓN INCOMPLETA**: El número de SKUs únicos disminuyó "
                            f"desde {previous_skus} a {current_skus} ({reduction_skus:.1f}% de reducción). "
                            f"Esto podría indicar que faltan datos de algunos SKUs."
                        )
            
            # Verificar última fecha
            current_last_date = None
            if not ventas.empty and "fecha" in ventas.columns:
                current_last_date = ventas["fecha"].max()
                previous_last_date = snapshot.get("last_date")
                
                # Verificar antigüedad de datos (última fecha debe ser reciente)
                if current_last_date is not None:
                    hoy = pd.Timestamp.today().normalize()
                    dias_desde_ultima = (hoy - current_last_date).days
                    if dias_desde_ultima > 7:  # Más de 7 días sin datos nuevos
                        result["warnings"].append(
                            f"⚠️ Los datos de ventas parecen antiguos. "
                            f"Última fecha registrada: {current_last_date.strftime('%Y-%m-%d')} "
                            f"(hace {dias_desde_ultima} días)."
                        )
                
                # Comparar con snapshot anterior
                if previous_last_date is not None and current_last_date is not None:
                    if current_last_date <= previous_last_date:
                        result["warnings"].append(
                            f"⚠️ La última fecha de ventas no ha cambiado desde la actualización. "
                            f"Última fecha: {current_last_date.strftime('%Y-%m-%d')}."
                        )
            
            # Verificar cambios significativos en cantidad total (fallback si no hay hash)
            if not previous_hash:
                current_total = float(ventas["qty"].sum()) if not ventas.empty else 0.0
                previous_total = snapshot.get("total_qty", 0.0)
                
                # Si no hay cambios y pasó suficiente tiempo, puede ser un problema
                if abs(current_total - previous_total) < 0.01 and current_rows == previous_rows:
                    if elapsed > cooldown_seconds * 0.8:  # Pasó al menos 80% del cooldown
                        result["warnings"].append(
                            "⚠️ No se detectaron cambios en los datos de ventas después de la actualización. "
                            "Es posible que la actualización no se haya completado correctamente."
                        )
            
        elif update_type == "S2":  # Stock
            stock_raw = read_gsheets(sheet_id, TAB_STOCK)
            stock = normalize_stock_sheet(stock_raw)
            
            # Definir variables de conteo ANTES de usarlas
            current_rows = len(stock_raw)
            previous_rows = snapshot.get("row_count", 0)
            
            # MEJORA #2: Comparar hash de datos (más preciso que comparar totales)
            previous_hash = snapshot.get("data_hash")
            if previous_hash:
                if not stock.empty:
                    current_hash = _calculate_data_hash(stock, ["sku", "stock"])
                elif not stock_raw.empty:
                    current_hash = _calculate_data_hash(stock_raw)
                else:
                    current_hash = None
                
                if current_hash == previous_hash:
                    # Los datos son idénticos
                    if elapsed > cooldown_seconds * 0.8:  # Pasó al menos 80% del cooldown
                        result["warnings"].append(
                            "⚠️ Los datos de stock son idénticos a antes de la actualización (verificado por hash). "
                            "Es muy probable que la actualización no se haya completado correctamente."
                        )
                else:
                    # Los datos cambiaron, pero verificar si es una actualización completa o parcial
                    # Si hay menos filas que antes, podría ser una actualización incompleta
                    if previous_rows > 0 and current_rows < previous_rows:
                        reduction_pct = ((previous_rows - current_rows) / previous_rows) * 100
                        if reduction_pct > 5.0:
                            result["warnings"].append(
                                f"⚠️ **ATENCIÓN**: Los datos cambiaron (hash diferente), pero el número de filas "
                                f"disminuyó desde {previous_rows} a {current_rows} ({reduction_pct:.1f}% de reducción). "
                                f"Esto podría indicar una actualización parcial o incompleta (ej: error 429 en Make.com)."
                            )
                    else:
                        # Los datos cambiaron y no hay reducción significativa, actualización probablemente exitosa
                        result["success"] = True
            
            # Verificar cambios en número de filas
            
            # DETECCIÓN DE ACTUALIZACIÓN INCOMPLETA: Reducción significativa de filas
            # Si hay menos filas después de una actualización, podría ser un problema
            if previous_rows > 0 and current_rows < previous_rows:
                reduction_pct = ((previous_rows - current_rows) / previous_rows) * 100
                # Si se redujo más del 5%, podría ser una actualización incompleta
                if reduction_pct > 5.0 and elapsed > cooldown_seconds * 0.8:
                    result["warnings"].append(
                        f"⚠️ **POSIBLE ACTUALIZACIÓN INCOMPLETA**: El número de filas de stock disminuyó significativamente "
                        f"desde {previous_rows} a {current_rows} ({reduction_pct:.1f}% de reducción). "
                        f"Esto podría indicar que la actualización se detuvo antes de completarse (ej: error 429 en Make.com)."
                    )
            
            # Verificar número de SKUs únicos en stock (otra señal de actualización incompleta)
            if not stock.empty and "sku" in stock.columns:
                current_skus = stock["sku"].nunique()
                previous_skus = snapshot.get("unique_skus")
                if previous_skus and current_skus < previous_skus:
                    reduction_skus = ((previous_skus - current_skus) / previous_skus) * 100
                    if reduction_skus > 5.0 and elapsed > cooldown_seconds * 0.8:
                        result["warnings"].append(
                            f"⚠️ **POSIBLE ACTUALIZACIÓN INCOMPLETA**: El número de SKUs únicos en stock disminuyó "
                            f"desde {previous_skus} a {current_skus} ({reduction_skus:.1f}% de reducción). "
                            f"Esto podría indicar que faltan datos de algunos SKUs."
                        )
            
            # Verificar cambios en stock total (fallback si no hay hash)
            if not previous_hash:
                current_total = float(stock["stock"].sum()) if not stock.empty else 0.0
                previous_total = snapshot.get("total_stock", 0.0)
                
                # Si no hay cambios y pasó suficiente tiempo, puede ser un problema
                if abs(current_total - previous_total) < 0.01 and current_rows == previous_rows:
                    if elapsed > cooldown_seconds * 0.8:  # Pasó al menos 80% del cooldown
                        result["warnings"].append(
                            "⚠️ No se detectaron cambios en los datos de stock después de la actualización. "
                            "Es posible que la actualización no se haya completado correctamente."
                        )
            
            # Verificar si el stock está vacío (puede ser un error)
            if stock.empty:
                result["errors"].append(
                    "❌ ERROR: Los datos de stock están vacíos. La actualización puede haber fallado."
                )
                result["success"] = False
        
        # Si hay errores, marcar como no exitoso
        if result["errors"]:
            result["success"] = False
            
    except Exception as e:
        result["errors"].append(f"❌ Error al verificar actualización: {str(e)}")
        result["success"] = False
    
    return result

# ======================================
# CARGA CONFIG CLIENTES
# ======================================
clientes_df = load_clientes_config()
if clientes_df is not None and not clientes_df.empty:
    clientes_df = ensure_clientes_columns(clientes_df)

# Botón Salir SIEMPRE visible (limpia sesión)
if st.sidebar.button("Salir", type="secondary", use_container_width=True):
    for k in [
        "TENANT_NAME","TENANT_ID","TENANT_ROW",
        "cooldown_until","batch_scenarios_disparados",
        "MODULO_ACTIVO",
    ]:
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

# SEGURIDAD MULTI-TENANT: CURRENT_SHEET_ID siempre debe venir del tenant actual
# NUNCA usar DEFAULT_SHEET_ID como fallback en producción multi-tenant
if TENANT_ROW and TENANT_ROW.get("sheet_id"):
    CURRENT_SHEET_ID = str(TENANT_ROW.get("sheet_id")).strip()
    if not CURRENT_SHEET_ID:
        # Si sheet_id está vacío, usar default solo en modo desarrollo
        CURRENT_SHEET_ID = DEFAULT_SHEET_ID
        st.warning("⚠️ **ADVERTENCIA DE SEGURIDAD**: El tenant no tiene sheet_id configurado. Usando sheet por defecto.")
else:
    # Solo usar DEFAULT_SHEET_ID si no hay tenant configurado (modo desarrollo/single-tenant)
    CURRENT_SHEET_ID = DEFAULT_SHEET_ID
    if TENANT_ROW:
        st.error("❌ **ERROR DE SEGURIDAD**: El tenant está configurado pero no tiene sheet_id. No se pueden leer datos.")

KAME_CLIENT_ID     = TENANT_ROW.get("kame_client_id", "") if TENANT_ROW else ""
KAME_CLIENT_SECRET = TENANT_ROW.get("kame_client_secret", "") if TENANT_ROW else ""
USE_STOCK_TOTAL    = _truthy(TENANT_ROW.get("use_stock_total", False)) if TENANT_ROW else False

# Flags de módulos (Compras / Ventas) leídos desde clientes_config
MOD_COMPRAS = bool(TENANT_ROW.get("mod_compras", True)) if TENANT_ROW else True
MOD_VENTAS  = bool(TENANT_ROW.get("mod_ventas", False)) if TENANT_ROW else False

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

# Módulos disponibles según clientes_config
module_options = []
if MOD_COMPRAS:
    module_options.append("Compras")
if MOD_VENTAS:
    module_options.append("Ventas")
if not module_options:
    module_options = ["Compras"]

if "MODULO_ACTIVO" not in st.session_state or st.session_state["MODULO_ACTIVO"] not in module_options:
    st.session_state["MODULO_ACTIVO"] = module_options[0]

# --- navegación lateral ---
st.sidebar.markdown("### Navegación")
st.sidebar.markdown(
    f"""<div style="margin-top: 8px;"><a href="{target_report_url}" target="_blank" style="display: inline-block; padding: 12px 16px; background: linear-gradient(135deg, rgba(15, 118, 110, 0.1) 0%, rgba(20, 184, 166, 0.1) 100%); border: 1.5px solid rgba(15, 118, 110, 0.3); border-radius: 10px; color: #14b8a6; text-decoration: none; font-weight: 500; transition: all 0.3s ease; width: 100%; box-sizing: border-box;">📊 Reportería de ventas</a></div>""",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

st.sidebar.markdown("### Módulo")
modulo_activo = st.sidebar.radio(
    "Selecciona el módulo",
    module_options,
    index=module_options.index(st.session_state["MODULO_ACTIVO"])
)
st.session_state["MODULO_ACTIVO"] = modulo_activo

modo_datos = "ONLINE (KAME ERP)" if not OFFLINE else "OFFLINE (CSV)"

# ======================================
# MÓDULO COMPRAS (ACTUAL)
# ======================================
if modulo_activo == "Compras":
    st.title("🧠 Predictor de Compras ↪")
    st.caption(f"Fuente de datos: **{modo_datos}** — Tenant: **{TENANT_NAME}**")

    # --------------------------------------
    # BOTONES SUPERIORES (DESHABILITADOS - MODO BATCH)
    # --------------------------------------
    # MODO BATCH: Los botones S1, S2, S3 están deshabilitados
    # Las actualizaciones se ejecutan automáticamente por batch
    bt1, bt2, bt3 = st.columns(3)
    with bt1:
        st.button("📘 Actualizar ventas (S1)", use_container_width=True, disabled=True, help="Modo BATCH: Las actualizaciones se ejecutan automáticamente")
        # CÓDIGO COMENTADO - MODO BATCH
        # if st.button("📘 Actualizar ventas (S1)", use_container_width=True):
        #     if not CURRENT_SHEET_ID or not CURRENT_SHEET_ID.strip():
        #         st.error("❌ **ERROR DE SEGURIDAD**: No se puede actualizar ventas. CURRENT_SHEET_ID no está configurado correctamente.")
        #     else:
        #         save_data_snapshot(CURRENT_SHEET_ID, "S1")
        #         resp = trigger_make(MAKE_WEBHOOK_S1_URL, {"reason": "ui_run", "tenant_id": TENANT_ID})
        #         if resp.get("ok"):
        #             st.success(f"Actualización de ventas enviada a Make. Espera ~{COOLDOWN_S1} segundos antes de ejecutar la predicción.")
        #             start_cooldown(COOLDOWN_S1)
        #         else:
        #             st.error(
        #                 f"No se pudo actualizar ventas (S1). "
        #                 f"Código: {resp.get('status')}, detalle: {resp.get('error') or resp.get('text')}"
        #             )
    with bt2:
        st.button("📦 Actualizar stock total (S2)", use_container_width=True, disabled=True, help="Modo BATCH: Las actualizaciones se ejecutan automáticamente")
        # CÓDIGO COMENTADO - MODO BATCH
        # if st.button("📦 Actualizar stock total (S2)", use_container_width=True):
        #     if not CURRENT_SHEET_ID or not CURRENT_SHEET_ID.strip():
        #         st.error("❌ **ERROR DE SEGURIDAD**: No se puede actualizar stock. CURRENT_SHEET_ID no está configurado correctamente.")
        #     else:
        #         save_data_snapshot(CURRENT_SHEET_ID, "S2")
        #         resp = trigger_make(
        #             MAKE_WEBHOOK_S2_URL,
        #             {"reason": "ui_run", "tenant_id": TENANT_ID, "use_stock_total": True},
        #         )
        #         if resp.get("ok"):
        #             dynamic_cooldown = calculate_dynamic_cooldown("S2", COOLDOWN_S2)
        #             minutos = dynamic_cooldown // 60
        #             segundos = dynamic_cooldown % 60
        #             if minutos > 0:
        #                 tiempo_str = f"{minutos} minuto{'s' if minutos > 1 else ''}"
        #                 if segundos > 0:
        #                     tiempo_str += f" {segundos} segundo{'s' if segundos > 1 else ''}"
        #             else:
        #                 tiempo_str = f"{segundos} segundo{'s' if segundos > 1 else ''}"
        #             st.success(f"Actualización de stock total enviada. Espera ~{tiempo_str} antes de ejecutar la predicción.")
        #             start_cooldown(dynamic_cooldown)
        #         else:
        #             st.error(
        #                 f"No se pudo actualizar stock (S2). "
        #                 f"Código: {resp.get('status')}, detalle: {resp.get('error') or resp.get('text')}"
        #             )
    with bt3:
        st.button("🧾 Actualizar inbound (S3)", use_container_width=True, disabled=True, help="Modo BATCH: Las actualizaciones se ejecutan automáticamente")
        # CÓDIGO COMENTADO - MODO BATCH
        # if st.button("🧾 Actualizar inbound (S3)", use_container_width=True):
        #     resp = trigger_make(MAKE_WEBHOOK_S3_URL, {"reason": "ui_run", "tenant_id": TENANT_ID})
        #     if resp.get("ok"):
        #         st.success("Actualización de inbound enviada. Espera ~30 segundos antes de ejecutar la predicción.")
        #         start_cooldown(COOLDOWN_S3)
        #     else:
        #         st.error(
        #             f"No se pudo actualizar inbound (S3). "
        #             f"Código: {resp.get('status')}, detalle: {resp.get('error') or resp.get('text')}"
        #         )

    st.markdown("")
    
    # HENRY: Mostrar última actualización de S1, S2 y S3
    if CURRENT_SHEET_ID and CURRENT_SHEET_ID.strip():
        last_update_s1 = _get_last_update_timestamp(CURRENT_SHEET_ID, "S1")
        last_update_s2 = _get_last_update_timestamp(CURRENT_SHEET_ID, "S2")
        last_update_s3 = _get_last_update_timestamp(CURRENT_SHEET_ID, "S3")
        
        # Siempre mostrar la fila de información si tenemos un sheet ID
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            if last_update_s1:
                tiempo_str = _get_elapsed_time_info(last_update_s1)
                st.caption(f"📘 **Última actualización de ventas (S1):** {last_update_s1.strftime('%Y-%m-%d %H:%M:%S')} ({tiempo_str} atrás)")
            else:
                st.caption("📘 **Última actualización de ventas (S1):** No disponible")
        
        with col_info2:
            if last_update_s2:
                tiempo_str = _get_elapsed_time_info(last_update_s2)
                st.caption(f"📦 **Última actualización de stock (S2):** {last_update_s2.strftime('%Y-%m-%d %H:%M:%S')} ({tiempo_str} atrás)")
            else:
                st.caption("📦 **Última actualización de stock (S2):** No disponible")

        with col_info3:
            if last_update_s3:
                tiempo_str = _get_elapsed_time_info(last_update_s3)
                st.caption(f"🧾 **Última actualización inbound (S3):** {last_update_s3.strftime('%Y-%m-%d %H:%M:%S')} ({tiempo_str} atrás)")
            else:
                st.caption("🧾 **Última actualización inbound (S3):** No disponible")
        
        st.markdown("---")

    # filtros
    colA, colB, colC = st.columns(3)
    freq    = colA.selectbox("Frecuencia", ["Mensual (M)", "Semanal (W)"], index=0)
    horizon = colB.slider("Horizonte (períodos)", 2, 24, 6, 1)
    modo    = colC.selectbox("Modo", ["Global", "Por SKU"], index=1)
    sku_q   = st.text_input("SKU (opcional)") if modo == "Por SKU" else None

    # opciones avanzadas
    with st.expander("Opciones avanzadas (Make / Debug)"):
        # MODO BATCH: Opción "Disparar S1/S2/S3" deshabilitada
        # disparar = st.checkbox("Disparar S1/S2/S3 antes de predecir", value=False)
        disparar = False  # Siempre False en modo BATCH
        mostrar_debug   = st.checkbox("Mostrar debug de columnas/valores config", value=False)
        mostrar_inbound = st.checkbox("Mostrar inbound agrupado", value=True)
        mostrar_stocks  = st.checkbox("Mostrar stocks (informativos)", value=True)
        
        if mostrar_debug:
            st.write(f"**Debug BATCH Mode:**")
            st.write(f"- TENANT_ID: `{TENANT_ID}`")
            st.write(f"- CURRENT_SHEET_ID: `{CURRENT_SHEET_ID}`")
            st.write(f"- Chile Now: `{pd.Timestamp.now(tz='America/Santiago')}`")
            try:
                test_cfg = read_gsheets(CURRENT_SHEET_ID, TAB_CONFIG)
                st.write(f"- Lectura `{TAB_CONFIG}`: {'✅ OK' if not test_cfg.empty else '❌ Vacía/Error'}")
                if not test_cfg.empty:
                    st.write("- Columnas encontradas (índice: nombre):")
                    st.write({i: col for i, col in enumerate(test_cfg.columns)})
                    st.write("- Primera fila (valores crudos):", test_cfg.iloc[0].to_dict())
                    
                    # Probar cálculos de tiempo en debug
                    for ut in ["S1", "S2", "S3"]:
                        ts_test = _get_last_update_timestamp(CURRENT_SHEET_ID, ut)
                        info_test = _get_elapsed_time_info(ts_test)
                        st.write(f"- Test {ut}: TS=`{ts_test}`, Info=`{info_test}`")
            except Exception as e:
                st.error(f"Error en debug de config: {e}")

    # --------------------------------------
    # BOTÓN PRINCIPAL (MODO BATCH - verificación de estado)
    # --------------------------------------
    # MODO BATCH: Verificar si el batch está listo en lugar de usar cooldown
    batch_check = check_batch_ready(CURRENT_SHEET_ID) if CURRENT_SHEET_ID else {"ready": False, "message": "Sheet ID no configurado", "status": "UNKNOWN"}
    
    # CÓDIGO COMENTADO - LÓGICA DE COOLDOWN ELIMINADA EN MODO BATCH
    # now_ts = time.time()
    # cooldown_until = st.session_state.get("cooldown_until", 0)
    # remaining = max(0, int(cooldown_until - now_ts))
    remaining = 0  # Siempre 0 en modo BATCH
    
    # Mostrar estado del batch
    if not batch_check["ready"]:
        st.info(batch_check["message"])
        # Si necesita rerun para continuar verificando, hacerlo después de 3 segundos
        if batch_check.get("needs_rerun", False):
            time.sleep(3)
            st.rerun()
    
    # CÓDIGO ELIMINADO - LÓGICA DE COOLDOWN NO USADA EN MODO BATCH
    # Todo el bloque de cooldown fue eliminado y reemplazado por check_batch_ready()
    
    # MODO BATCH: Eliminar verificación de snapshots (ya no se crean desde la UI)
    # Las verificaciones ahora se hacen solo con check_batch_ready()
    
    # CÓDIGO ELIMINADO: Todo el bloque de cooldown y verificación de snapshots fue eliminado
    # En modo BATCH, solo usamos check_batch_ready() para verificar el estado
    
    # HENRY: Verificar si las actualizaciones batch están listas (ANTES del botón)
    # MODO BATCH: Ya no verificamos snapshots, solo el estado batch
    # CÓDIGO ELIMINADO: Todo el bloque de verificación de snapshots y cooldown fue eliminado
    # En modo BATCH, solo usamos check_batch_ready() que ya se ejecutó arriba
    
    # Verificar estado batch (ya hecho arriba con check_batch_ready)
    # batch_check ya contiene el estado y mensaje
    
    # CÓDIGO ELIMINADO: Todo el bloque de verificación de snapshots y cooldown fue eliminado
    # El código siguiente estaba mal indentado y fue eliminado completamente
    # (Más de 200 líneas de código de cooldown fueron eliminadas)
    
    # MODO BATCH: La verificación de batch ya se hizo arriba con check_batch_ready()
    # No necesitamos verificar snapshots ni cooldowns
    # TODO EL CÓDIGO DE COOLDOWN FUE ELIMINADO (más de 200 líneas)
    # El bloque completo de verificación de snapshots y cooldown fue eliminado
    # En modo BATCH, solo usamos check_batch_ready() que ya se ejecutó arriba
    
    # CÓDIGO ELIMINADO: Todo el bloque de cooldown (más de 200 líneas) fue eliminado
    # El código siguiente estaba mal indentado y causaba errores de sintaxis
    # Fue completamente eliminado en modo BATCH
    
    # MODO BATCH: Verificación de actualizaciones (simplificada - sin snapshots)
    # Ya no verificamos snapshots porque las actualizaciones son automáticas por batch
    # TODO EL CÓDIGO DE COOLDOWN (más de 200 líneas) FUE ELIMINADO
    # Todo el bloque de código mal indentado fue eliminado (más de 200 líneas)
    
    # MODO BATCH: Verificación simplificada - solo verificamos batch_check que ya se hizo arriba
    # TODO EL CÓDIGO DE COOLDOWN (más de 200 líneas) FUE ELIMINADO COMPLETAMENTE

    # MODO BATCH: Verificación simplificada - no verificamos snapshots
    # Las actualizaciones son automáticas por batch, solo verificamos batch_check
    verification_messages = []  # Vacío en modo BATCH (no hay snapshots)
    # CÓDIGO COMENTADO - MODO BATCH
    # if remaining == 0:
    #     verification_messages = []
    #     if "snapshot_S1" in st.session_state:
    #         if not CURRENT_SHEET_ID or not CURRENT_SHEET_ID.strip():
    #             if TENANT_ROW:
    #                 st.error("❌ **ERROR DE SEGURIDAD**: Se intentó verificar actualización pero CURRENT_SHEET_ID no está configurado correctamente.")
    #         check_s1 = check_update_success(CURRENT_SHEET_ID, "S1", COOLDOWN_S1)
    #         if check_s1["warnings"] or check_s1["errors"]:
    #             verification_messages.append({...})
    #         elif check_s1["success"]:
    #             verification_messages.append({...})
    #     if "snapshot_S2" in st.session_state:
    #         if not CURRENT_SHEET_ID or not CURRENT_SHEET_ID.strip():
    #             if TENANT_ROW:
    #                 st.error("...")
    #         cooldown_s2_verificacion = calculate_dynamic_cooldown("S2", COOLDOWN_S2)
    #         check_s2 = check_update_success(CURRENT_SHEET_ID, "S2", cooldown_s2_verificacion)
    #         if check_s2["warnings"] or check_s2["errors"]:
    #             verification_messages.append({...})
    #         elif check_s2["success"]:
    #             verification_messages.append({...})
    #     if verification_messages:
    #         st.markdown("---")
    #         for msg in verification_messages:
    #             st.subheader(msg["title"])
    #             if msg.get("success_msg"):
    #                 st.success(msg["success_msg"])
    #             for warning in msg["warnings"]:
    #                 st.warning(warning)
    #             for error in msg["errors"]:
    #                 st.error(error)
    #             if msg["success"] and not msg.get("success_msg") and (msg["warnings"] or msg["errors"]):
    #                 st.info("ℹ️ Los datos se leyeron correctamente, pero se detectaron posibles problemas con la actualización.")
    #         st.markdown("---")

    # HENRY: Verificar si hay errores críticos que bloqueen la ejecución
    critical_errors = []
    critical_warnings = []
    
    # MODO BATCH: No verificamos snapshots (las actualizaciones son automáticas)
    # CÓDIGO COMENTADO - MODO BATCH
    # if remaining == 0:
    #     # Re-ejecutar verificaciones si hay snapshots
    #     if "snapshot_S1" in st.session_state or "snapshot_S2" in st.session_state:
    #         if "snapshot_S1" in st.session_state:
    #             check_s1 = check_update_success(CURRENT_SHEET_ID, "S1", COOLDOWN_S1)
    #             if check_s1["errors"]:
    #                 critical_errors.extend([f"S1: {e}" for e in check_s1["errors"]])
    #             for w in check_s1["warnings"]:
    #                 if "ACTUALIZACIÓN INCOMPLETA" in w or "incompleta" in w.lower():
    #                     critical_warnings.append(f"S1: {w}")
    #         if "snapshot_S2" in st.session_state:
    #             cooldown_s2_verificacion = calculate_dynamic_cooldown("S2", COOLDOWN_S2)
    #             check_s2 = check_update_success(CURRENT_SHEET_ID, "S2", cooldown_s2_verificacion)
    #             if check_s2["errors"]:
    #                 critical_errors.extend([f"S2: {e}" for e in check_s2["errors"]])
    #             for w in check_s2["warnings"]:
    #                 if "ACTUALIZACIÓN INCOMPLETA" in w or "incompleta" in w.lower():
    #                     critical_warnings.append(f"S2: {w}")
    
    # Guardar estado de verificación para mostrar antes de ejecutar
    st.session_state["critical_errors"] = critical_errors
    st.session_state["critical_warnings"] = critical_warnings
    
    # MODO BATCH: El botón se habilita solo si el batch está listo
    # Verificar estado batch (ya hecho arriba con check_batch_ready)
    if not batch_check["ready"]:
        main_label = f"Ejecutar predicción (BLOQUEADO: {batch_check['message']})"
        main_disabled = True
    elif critical_errors:
        main_label = "❌ Ejecutar predicción (BLOQUEADO: errores críticos)"
        main_disabled = True
    elif critical_warnings:
        main_label = "⚠️ Ejecutar predicción (advertencias críticas)"
        main_disabled = False
    else:
        main_label = "Ejecutar predicción"
        main_disabled = False

    # Mostrar advertencias críticas antes del botón
    if critical_errors:
        st.error("🚫 **EJECUCIÓN BLOQUEADA**: Se detectaron errores críticos en las actualizaciones. No se puede ejecutar la predicción con datos incompletos o erróneos.")
        for err in critical_errors:
            st.error(f"  • {err}")
        st.info("💡 **Recomendación**: Reintenta la actualización (S1/S2) y espera a que se complete correctamente antes de ejecutar la predicción.")
    
    if critical_warnings and not critical_errors:
        st.warning("⚠️ **ADVERTENCIA CRÍTICA**: Se detectaron posibles actualizaciones incompletas. Se recomienda reintentar antes de ejecutar la predicción.")
        for warn in critical_warnings:
            st.warning(f"  • {warn}")
        st.info("💡 **Recomendación**: Espera a que el batch complete la actualización. Las actualizaciones se ejecutan automáticamente.")

    if st.button(main_label, type="primary", use_container_width=True, disabled=main_disabled):
        # MODO BATCH: Verificación rápida final (sin revalidar completamente)
        # Si el batch ya estaba confirmado, verificar rápidamente que sigue listo
        if not batch_check["ready"]:
            st.error(f"❌ **EJECUCIÓN CANCELADA**: {batch_check['message']}")
            st.stop()
        
        # BLOQUEO FINAL: No ejecutar si hay errores críticos
        if critical_errors:
            st.error("❌ **EJECUCIÓN CANCELADA**: No se puede ejecutar la predicción con errores críticos en los datos.")
            st.stop()
        # MODO BATCH: Opción "disparar S1/S2/S3" deshabilitada
        # CÓDIGO COMENTADO - MODO BATCH
        # if disparar and not st.session_state.get("batch_scenarios_disparados", False):
        # CÓDIGO COMENTADO - MODO BATCH
        # if disparar and not st.session_state.get("batch_scenarios_disparados", False):
        #     st.info("Disparando S1 (ventas), S2 (stock total) y S3 (inbound) desde la app…")
        #     if CURRENT_SHEET_ID and CURRENT_SHEET_ID != DEFAULT_SHEET_ID:
        #         save_data_snapshot(CURRENT_SHEET_ID, "S1")
        #         save_data_snapshot(CURRENT_SHEET_ID, "S2")
        #     s1 = trigger_make(MAKE_WEBHOOK_S1_URL, {"reason": "ui_run", "tenant_id": TENANT_ID})
        #     s2 = trigger_make(MAKE_WEBHOOK_S2_URL, {"reason": "ui_run", "tenant_id": TENANT_ID, "use_stock_total": True})
        #     s3 = trigger_make(MAKE_WEBHOOK_S3_URL, {"reason": "ui_run", "tenant_id": TENANT_ID})
        #     if s1.get("ok") and s2.get("ok") and s3.get("ok"):
        #         cooldown_s1_dynamic = calculate_dynamic_cooldown("S1", COOLDOWN_S1)
        #         cooldown_s2_dynamic = calculate_dynamic_cooldown("S2", COOLDOWN_S2)
        #         start_cooldown(max(cooldown_s1_dynamic, cooldown_s2_dynamic, COOLDOWN_S3))
        #         st.session_state["batch_scenarios_disparados"] = True
        #     else:
        #         st.error("Alguno de los escenarios S1/S2/S3 devolvió error.")
        #     st.rerun()

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
            ventas_antes = len(ventas)
            ventas      = ventas[ventas["sku"].str.lower() == f]
            stock_p     = stock_p[stock_p["sku"].str.lower() == f]
            stock_t     = stock_t[stock_t["sku"].str.lower() == f]
            stock_total = stock_total[stock_total["sku"].str.lower() == f]
            config      = config[config["sku"].str.lower() == f]
            inbound     = inbound[inbound["sku"].str.lower() == f]
            
            # DEBUG: Log de filtrado
            if mostrar_debug:
                st.write(f"🔍 DEBUG FILTRADO: SKU buscado='{f}' (original='{sku_q}')")
                st.write(f"   Ventas antes del filtro: {ventas_antes} filas")
                st.write(f"   Ventas después del filtro: {len(ventas)} filas")
                if len(ventas) > 0:
                    st.write(f"   SKUs únicos en ventas filtradas: {ventas['sku'].unique().tolist()}")
                    st.write(f"   Rango de fechas: {ventas['fecha'].min()} a {ventas['fecha'].max()}")
                    st.write(f"   Total cantidad: {ventas['qty'].sum()}")

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
            if mostrar_debug:
                st.write(f"🔍 DEBUG: ventas.empty=True. SKU buscado: '{sku_q if modo == 'Por SKU' and sku_q else 'N/A'}'")
        else:
            with st.spinner("Calculando pronóstico…"):
                freq_code = "M" if freq.startswith("Mensual") else "W"
                
                # DEBUG: Log antes de forecast_all
                if mostrar_debug:
                    st.write(f"🔍 DEBUG PRE-FORECAST:")
                    st.write(f"   Ventas a enviar: {len(ventas)} filas")
                    st.write(f"   SKUs en ventas: {ventas['sku'].unique().tolist()}")
                    st.write(f"   Stock a enviar: {len(stock_total)} filas")
                    st.write(f"   SKUs en stock: {stock_total['sku'].unique().tolist() if not stock_total.empty else '[]'}")
                    st.write(f"   Frecuencia: {freq_code}, Horizonte: {horizon}")
                
                det, res, prop = forecast_all(
                    ventas=ventas,
                    stock=stock_total,
                    config=config,
                    inbound=inbound_core,
                    freq=freq_code,
                    horizon_override=horizon,
                )
                
                # DEBUG: Análisis de estacionalidad (mostrar siempre para SKUs con predicciones constantes)
                if modo == "Por SKU" and sku_q and not det.empty:
                    sku_target = str(sku_q).strip().upper()
                    det_sku = det[det['sku'] == sku_target]
                    if not det_sku.empty:
                        # Verificar si todas las predicciones son iguales (predicción constante)
                        valores_unicos = det_sku['demanda_predicha'].nunique()
                        es_constante = valores_unicos == 1
                        
                        # Verificar si hay períodos con demanda_predicha = 0
                        ceros = det_sku[det_sku['demanda_predicha'] == 0]
                        if len(ceros) > 0:
                            with st.expander("⚠️ DEBUG: Períodos con demanda_predicha = 0", expanded=True):
                                st.write(f"**SKU:** {sku_target}")
                                st.write(f"**Períodos con demanda_predicha = 0:** {len(ceros)} de {len(det_sku)}")
                                st.dataframe(ceros[['period', 'fecha_periodo', 'demanda_predicha']], use_container_width=True)
                                st.write(f"**Períodos con demanda_predicha > 0:**")
                                st.dataframe(det_sku[det_sku['demanda_predicha'] > 0][['period', 'fecha_periodo', 'demanda_predicha']], use_container_width=True)
                                if not res.empty:
                                    res_sku = res[res['sku'] == sku_target]
                                    if not res_sku.empty:
                                        st.write(f"**Modelo usado:** {res_sku.iloc[0]['modelo']}")
                                        st.write(f"**Demanda_H total:** {res_sku.iloc[0]['demanda_H']}")
                                        st.write(f"**Total_qty_hist:** {res_sku.iloc[0].get('total_qty_hist', 'N/A')}")
                                        st.write(f"**nz (períodos con demanda>0 en histórico):** {res_sku.iloc[0].get('nz', 'N/A')}")
                                        st.write(f"**zr (tasa de ceros en histórico):** {res_sku.iloc[0].get('zr', 'N/A')}")
                        
                        # ANÁLISIS DE ESTACIONALIDAD: Mostrar si la predicción es constante
                        if es_constante and not ventas.empty:
                            with st.expander("📊 ANÁLISIS DE ESTACIONALIDAD (Predicción constante detectada)", expanded=True):
                                st.write(f"**SKU:** {sku_target}")
                                st.write(f"**Predicción constante:** {det_sku.iloc[0]['demanda_predicha']:.4f} para todos los períodos")
                                
                                if not res.empty:
                                    res_sku = res[res['sku'] == sku_target]
                                    if not res_sku.empty:
                                        st.write(f"**Modelo usado:** {res_sku.iloc[0]['modelo']}")
                                
                                # Analizar estacionalidad desde los datos históricos
                                import numpy as np
                                ventas_sku = ventas[ventas['sku'].str.upper() == sku_target].copy()
                                if not ventas_sku.empty:
                                    # Agregar por mes (similar a lo que hace predictor_core)
                                    ventas_sku['period'] = ventas_sku['fecha'].dt.to_period('M').dt.to_timestamp()
                                    ventas_mensual = ventas_sku.groupby('period', as_index=False)['qty'].sum().sort_values('period')
                                    
                                    if len(ventas_mensual) > 0:
                                        st.write(f"\n**📈 DATOS HISTÓRICOS:**")
                                        st.write(f"   Períodos únicos: {len(ventas_mensual)}")
                                        st.write(f"   Rango: {ventas_mensual['period'].min().strftime('%Y-%m')} a {ventas_mensual['period'].max().strftime('%Y-%m')}")
                                        st.write(f"   Media mensual: {ventas_mensual['qty'].mean():.2f}")
                                        st.write(f"   Desviación estándar: {ventas_mensual['qty'].std():.2f}")
                                        
                                        # Análisis de tendencia (primera mitad vs segunda mitad)
                                        if len(ventas_mensual) >= 12:
                                            mid_point = len(ventas_mensual) // 2
                                            primera_mitad = ventas_mensual.iloc[:mid_point]['qty'].mean()
                                            segunda_mitad = ventas_mensual.iloc[mid_point:]['qty'].mean()
                                            trend_ratio = ((segunda_mitad - primera_mitad) / primera_mitad * 100) if primera_mitad > 0 else 0
                                            
                                            st.write(f"\n**📉 ANÁLISIS DE TENDENCIA:**")
                                            st.write(f"   Media primera mitad: {primera_mitad:.2f}")
                                            st.write(f"   Media segunda mitad: {segunda_mitad:.2f}")
                                            st.write(f"   Variación: {trend_ratio:+.1f}%")
                                            st.write(f"   {'✅ Tendencia detectada' if abs(trend_ratio) > 20 else '⚠️  No hay tendencia clara (<20%)'}")
                                            st.write(f"   *(El modelo requiere >20% para detectar tendencia)*")
                                        
                                        # Análisis de estacionalidad por mes del año
                                        if len(ventas_mensual) >= 12:
                                            # CORRECCIÓN: Usar ventas_mensual agregadas por mes del año, no registros individuales
                                            ventas_mensual['mes'] = ventas_mensual['period'].dt.month
                                            estacionalidad = ventas_mensual.groupby('mes', as_index=False)['qty'].agg({
                                                'total': 'sum',
                                                'media_mensual': 'mean',  # Media de los meses agregados (no de registros individuales)
                                                'conteo_meses': 'count'  # Cuántos meses de ese tipo hay en el histórico
                                            }).sort_values('mes')
                                            
                                            if len(estacionalidad) > 1:
                                                media_global = ventas_mensual['qty'].mean()
                                                std_global = ventas_mensual['qty'].std()
                                                cv = (std_global / media_global) if media_global > 0 else 0
                                                
                                                st.write(f"\n**🔄 ANÁLISIS DE ESTACIONALIDAD POR MES:**")
                                                st.write(f"   Coeficiente de variación (CV): {cv:.3f}")
                                                st.write(f"   {'Alta variabilidad estacional' if cv > 0.5 else 'Baja variabilidad' if cv < 0.3 else 'Variabilidad moderada'}")
                                                
                                                meses_nombres = {
                                                    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
                                                    5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
                                                    9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
                                                }
                                                
                                                st.write(f"\n**Desglose por mes del año (promedio de meses agregados):**")
                                                for _, row in estacionalidad.iterrows():
                                                    mes_nombre = meses_nombres.get(row['mes'], f"Mes {row['mes']}")
                                                    variacion_pct = ((row['media_mensual'] - media_global) / media_global * 100) if media_global > 0 else 0
                                                    st.write(f"   {mes_nombre:12s}: Total={row['total']:8.2f}, Media mensual={row['media_mensual']:8.2f}, "
                                                             f"Meses en histórico={int(row['conteo_meses']):2d}, Var={variacion_pct:+6.1f}%")
                                                
                                                mes_max = estacionalidad.loc[estacionalidad['media_mensual'].idxmax()]
                                                mes_min = estacionalidad.loc[estacionalidad['media_mensual'].idxmin()]
                                                st.write(f"\n   📌 Mes con MAYOR demanda promedio: {meses_nombres.get(mes_max['mes'])} "
                                                         f"(Media: {mes_max['media_mensual']:.2f})")
                                                st.write(f"   📌 Mes con MENOR demanda promedio: {meses_nombres.get(mes_min['mes'])} "
                                                         f"(Media: {mes_min['media_mensual']:.2f})")
                                                diferencia_pct = ((mes_max['media_mensual'] - mes_min['media_mensual']) / mes_min['media_mensual'] * 100) if mes_min['media_mensual'] > 0 else 0
                                                st.write(f"   📊 Diferencia: {diferencia_pct:.1f}% (mayor vs menor)")
                                                st.write(f"\n   ⚠️  **CONCLUSIÓN:** El modelo robusto NO captura estacionalidad mensual,")
                                                st.write(f"   solo detecta tendencias generales. Si hay estacionalidad pero no tendencia,")
                                                st.write(f"   la predicción será constante (como se ve aquí).")
                                        
                                        # Mostrar últimos períodos
                                        st.write(f"\n**📅 ÚLTIMOS 12 PERÍODOS HISTÓRICOS:**")
                                        ultimos_12 = ventas_mensual.tail(12) if len(ventas_mensual) >= 12 else ventas_mensual
                                        for _, row in ultimos_12.iterrows():
                                            periodo_str = row['period'].strftime('%Y-%m')
                                            st.write(f"   {periodo_str}: {row['qty']:8.2f}")
                
                # DEBUG: Log completo (solo si mostrar_debug está activado)
                if mostrar_debug:
                    st.write(f"🔍 DEBUG POST-FORECAST:")
                    st.write(f"   Detalle generado: {len(det)} filas")
                    st.write(f"   Resumen generado: {len(res)} filas")
                    st.write(f"   Propuesta generada: {len(prop)} filas")
                    if not det.empty:
                        st.write(f"   SKUs en detalle: {det['sku'].unique().tolist()}")
                        if modo == "Por SKU" and sku_q:
                            sku_target = str(sku_q).strip().upper()
                            det_sku = det[det['sku'] == sku_target]
                            if not det_sku.empty:
                                st.write(f"   Detalle para {sku_target}: {len(det_sku)} períodos")
                                if 'demanda_predicha' in det_sku.columns:
                                    st.write(f"   Demanda predicha total: {det_sku['demanda_predicha'].sum()}")
                                    st.write(f"   Valores por período:")
                                    for _, row in det_sku.iterrows():
                                        st.write(f"     {row['period']}: {row['demanda_predicha']}")
                                else:
                                    st.write(f"   Columnas disponibles: {list(det_sku.columns)}")
                            else:
                                st.write(f"   ⚠️ NO se encontró {sku_target} en detalle")
                    if not res.empty:
                        if modo == "Por SKU" and sku_q:
                            sku_target = str(sku_q).strip().upper()
                            res_sku = res[res['sku'] == sku_target]
                            if not res_sku.empty:
                                st.write(f"   Resumen para {sku_target}:")
                                st.write(f"     Modelo: {res_sku.iloc[0]['modelo']}")
                                st.write(f"     Demanda_H: {res_sku.iloc[0]['demanda_H']}")
                                st.write(f"     Total_qty_hist: {res_sku.iloc[0].get('total_qty_hist', 'N/A')}")
                            else:
                                st.write(f"   ⚠️ NO se encontró {sku_target} en resumen")

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

# ======================================
# ======================================
# MÓDULO VENTAS (FUNCIONAL, CON ESTACIONALIDAD MANUAL)
# ======================================
elif modulo_activo == "Ventas":
    st.title("📈 Predictor de Ventas")
    st.caption(f"Fuente de datos: **{modo_datos}** — Tenant: **{TENANT_NAME}**")

    if forecast_sales is None:
        st.error(
            "No se encontró `predictor_sales_core.forecast_sales`. "
            "Verifica que el archivo `predictor_sales_core.py` exista en la misma carpeta."
        )
        st.stop()

    # --------------------------------------
    # BOTÓN ACTUALIZAR VENTAS (S1) - DESHABILITADO - MODO BATCH
    # --------------------------------------
    st.button("📘 Actualizar ventas (S1)", use_container_width=True, disabled=True, help="Modo BATCH: Las actualizaciones se ejecutan automáticamente")
    # CÓDIGO COMENTADO - MODO BATCH
    # if st.button("📘 Actualizar ventas (S1)", use_container_width=True):
    #     if not CURRENT_SHEET_ID or not CURRENT_SHEET_ID.strip():
    #         st.error("❌ **ERROR DE SEGURIDAD**: No se puede actualizar ventas. CURRENT_SHEET_ID no está configurado correctamente.")
    #     else:
    #         save_data_snapshot(CURRENT_SHEET_ID, "S1")
    #         resp = trigger_make(MAKE_WEBHOOK_S1_URL, {"reason": "ui_run", "tenant_id": TENANT_ID})
    #     if resp.get("ok"):
    #         st.success(f"Actualización de ventas enviada a Make. Espera ~{COOLDOWN_S1} segundos antes de ejecutar la predicción.")
    #         start_cooldown(COOLDOWN_S1)
    #         st.rerun()
    #     else:
    #         st.error(
    #             f"No se pudo actualizar ventas (S1). "
    #             f"Código: {resp.get('status')}, detalle: {resp.get('error') or resp.get('text')}"
    #         )
    
    # HENRY: Mostrar última actualización de S1 (módulo Ventas)
    if CURRENT_SHEET_ID and CURRENT_SHEET_ID.strip():
        last_update_s1 = _get_last_update_timestamp(CURRENT_SHEET_ID, "S1")
        # Siempre mostrar la información si hay un sheet ID
        if last_update_s1:
            tiempo_str = _get_elapsed_time_info(last_update_s1)
            st.caption(f"📘 **Última actualización de ventas (S1):** {last_update_s1.strftime('%Y-%m-%d %H:%M:%S')} ({tiempo_str} atrás)")
        else:
            st.caption("📘 **Última actualización de ventas (S1):** No disponible")
        st.markdown("---")

    # MODO BATCH: Verificación de batch (igual que en Compras)
    batch_check_ventas = check_batch_ready(CURRENT_SHEET_ID) if CURRENT_SHEET_ID else {"ready": False, "message": "Sheet ID no configurado", "status": "UNKNOWN"}
    remaining = 0  # Siempre 0 en modo BATCH
    
    # Mostrar estado del batch
    if not batch_check_ventas["ready"]:
        st.info(batch_check_ventas["message"])
    
    # CÓDIGO ELIMINADO - LÓGICA DE COOLDOWN ELIMINADA EN MODO BATCH
    # if remaining > 0:
    #     # Contador regresivo en vivo (máx 60s para S1)
    #     for sec in range(remaining, 0, -1):
    #         cooldown_placeholder.warning(...)
    #         time.sleep(1)
    #     cooldown_placeholder.empty()
    #     st.session_state["cooldown_until"] = 0
    #     remaining = 0
    #     st.rerun()

    st.markdown("")

    col1, col2, col3 = st.columns(3)
    freq_label = col1.selectbox("Frecuencia", ["Mensual (M)", "Semanal (W)"], index=0)
    horizon    = col2.slider("Horizonte (períodos)", 2, 24, 6, 1)
    modo       = col3.selectbox("Modo", ["Global", "Por SKU"], index=0)
    sku_q      = st.text_input("SKU (opcional)") if modo == "Por SKU" else None

    # === NUEVO: configuración de estacionalidad manual por mes ===
    # Persistencia en session_state (Mejora #6)
    if "seasonal_weights" not in st.session_state:
        st.session_state["seasonal_weights"] = None
    if "usar_estacionalidad" not in st.session_state:
        st.session_state["usar_estacionalidad"] = False
    
    seasonal_weights = st.session_state.get("seasonal_weights")
    usar_estacionalidad = st.session_state.get("usar_estacionalidad", False)

    with st.expander("Estacionalidad manual por mes (opcional)", expanded=False):
        usar_estacionalidad = st.checkbox(
            "Activar ajuste manual de meses altos / bajos",
            value=usar_estacionalidad,
            help="Si lo desmarcas, el modelo usa solo la serie histórica sin ajustes manuales."
        )
        st.session_state["usar_estacionalidad"] = usar_estacionalidad

        # Meses en texto para que sea más cómodo
        meses_labels = {
            1: "1 - Enero",
            2: "2 - Febrero",
            3: "3 - Marzo",
            4: "4 - Abril",
            5: "5 - Mayo",
            6: "6 - Junio",
            7: "7 - Julio",
            8: "8 - Agosto",
            9: "9 - Septiembre",
            10: "10 - Octubre",
            11: "11 - Noviembre",
            12: "12 - Diciembre",
        }

        # Defaults pensados para Recotoner (bajos: dic-ene-feb, alto: marzo)
        # Cargar desde session_state si existe
        saved_low = st.session_state.get("low_default_labels", ["12 - Diciembre", "1 - Enero", "2 - Febrero"])
        saved_high = st.session_state.get("high_default_labels", ["3 - Marzo"])
        saved_low_factor = st.session_state.get("low_factor", 0.7)
        saved_high_factor = st.session_state.get("high_factor", 1.1)
        
        low_default_labels  = saved_low
        high_default_labels = saved_high

        low_sel = st.multiselect(
            "Meses con baja estacionalidad (ventas más bajas que el promedio)",
            options=list(meses_labels.values()),
            default=low_default_labels,
        )
        high_sel = st.multiselect(
            "Meses con alta estacionalidad (ventas más altas que el promedio)",
            options=list(meses_labels.values()),
            default=high_default_labels,
        )

        # Mejora #2: Validación de conflictos en meses
        label_to_month = {v: k for k, v in meses_labels.items()}
        low_months = {label_to_month[x] for x in low_sel if x in label_to_month}
        high_months = {label_to_month[x] for x in high_sel if x in label_to_month}
        conflict_months = low_months.intersection(high_months)
        
        if conflict_months and usar_estacionalidad:
            conflict_names = [meses_labels[m] for m in conflict_months]
            st.warning(
                f"⚠️ **Atención**: Los siguientes meses están en ambas categorías (bajos y altos): "
                f"{', '.join(conflict_names)}. "
                f"Se aplicará el factor de meses altos (tiene prioridad)."
            )

        c_low, c_high = st.columns(2)
        low_factor = c_low.slider(
            "Factor para meses bajos",
            min_value=0.3,
            max_value=1.0,
            value=saved_low_factor,
            step=0.05,
            help="Ej: 0.7 significa que esos meses se proyectan al 70% del nivel base.",
        )
        high_factor = c_high.slider(
            "Factor para meses altos",
            min_value=1.0,
            max_value=1.7,
            value=saved_high_factor,
            step=0.05,
            help="Ej: 1.1 significa que esos meses se proyectan al 110% del nivel base.",
        )
        
        # Guardar en session_state
        st.session_state["low_default_labels"] = low_sel
        st.session_state["high_default_labels"] = high_sel
        st.session_state["low_factor"] = low_factor
        st.session_state["high_factor"] = high_factor

        if usar_estacionalidad:
            # Convertimos textos seleccionados a números de mes
            seasonal_weights = {}
            for m in range(1, 13):
                w = 1.0
                if m in low_months:
                    w = low_factor
                if m in high_months:
                    # Mejora #2: Si hay conflicto, prioridad a meses altos
                    w = high_factor
                seasonal_weights[m] = float(w)
            
            # Guardar en session_state
            st.session_state["seasonal_weights"] = seasonal_weights
            
            # Mejora #3: Feedback visual de factores aplicados
            st.markdown("---")
            st.markdown("**📊 Factores de estacionalidad configurados:**")
            factors_df = pd.DataFrame([
                {"Mes": meses_labels[m], "Factor": f"{w:.2f}x", 
                 "Tipo": "Bajo" if w < 1.0 else "Alto" if w > 1.0 else "Normal"}
                for m, w in sorted(seasonal_weights.items())
            ])
            st.dataframe(factors_df, use_container_width=True, hide_index=True)
        else:
            st.session_state["seasonal_weights"] = None
    
    # Actualizar variable local desde session_state después del expander
    seasonal_weights = st.session_state.get("seasonal_weights")
    usar_estacionalidad = st.session_state.get("usar_estacionalidad", False)
    # === FIN estacionalidad manual ===

    # MODO BATCH: Verificación simplificada (módulo Ventas)
    # El botón se habilita solo si el batch está listo
    critical_errors_ventas = []
    critical_warnings_ventas = []
    
    # MODO BATCH: No verificamos snapshots (las actualizaciones son automáticas)
    # CÓDIGO COMENTADO - MODO BATCH
    # if "snapshot_S1" in st.session_state:
    #     check_s1 = check_update_success(CURRENT_SHEET_ID, "S1", COOLDOWN_S1)
    #     ...
    
    # Determinar si el botón debe estar deshabilitado (basado en batch_check)
    button_disabled = not batch_check_ventas["ready"]
    button_label = "Calcular proyección de ventas"
    if not batch_check_ventas["ready"]:
        button_label = f"❌ Calcular proyección (BLOQUEADO: {batch_check_ventas['message']})"
    elif critical_errors_ventas:
        button_label = "❌ Calcular proyección (BLOQUEADO: errores críticos)"
    elif critical_warnings_ventas:
        button_label = "⚠️ Calcular proyección (advertencias críticas)"
    
    if st.button(button_label, type="primary", use_container_width=True, disabled=button_disabled):
        # MODO BATCH: Verificar que el batch esté listo
        if not batch_check_ventas["ready"]:
            st.error(f"❌ **EJECUCIÓN CANCELADA**: {batch_check_ventas['message']}")
            st.stop()
        # BLOQUEO FINAL: No ejecutar si hay errores críticos
        if critical_errors_ventas:
            st.error("❌ **EJECUCIÓN CANCELADA**: No se puede ejecutar la predicción con errores críticos en los datos.")
            st.stop()
        
        # Definir variables al inicio del bloque del botón
        freq_code = "M" if freq_label.startswith("Mensual") else "W"
        estacionalidad_aplicada = False
        
        with st.spinner("Leyendo ventas_raw y calculando proyección…"):
            ventas_raw = read_gsheets(CURRENT_SHEET_ID, TAB_VENTAS)
            ventas_ext = normalize_ventas_for_sales(ventas_raw)

            # Filtro por SKU (opcional)
            if modo == "Por SKU" and sku_q:
                f = str(sku_q).strip().upper()
                ventas_ext = ventas_ext[ventas_ext["sku"] == f]

            if ventas_ext is None or ventas_ext.empty:
                st.warning("No hay ventas para los filtros dados.")
                st.stop()
            
            # HENRY: Verificar si la actualización reciente de ventas fue exitosa
            if "snapshot_S1" in st.session_state:
                check_s1 = check_update_success(CURRENT_SHEET_ID, "S1", COOLDOWN_S1)
                st.markdown("---")
                if check_s1["warnings"] or check_s1["errors"]:
                    st.subheader("🔍 Verificación de actualización de ventas (S1)")
                    for warning in check_s1["warnings"]:
                        st.warning(warning)
                    for error in check_s1["errors"]:
                        st.error(error)
                    if check_s1["success"]:
                        st.info("ℹ️ Los datos se leyeron correctamente, pero se detectaron posibles problemas con la actualización.")
                elif check_s1["success"]:
                    st.subheader("✅ Verificación de actualización de ventas (S1)")
                    st.success("Los datos de ventas se actualizaron correctamente.")
                st.markdown("---")

            # Mejora #1: Advertencia cuando se selecciona frecuencia semanal
            if usar_estacionalidad and seasonal_weights is not None:
                if freq_code == "W":
                    st.warning(
                        "⚠️ **Nota**: Los factores de estacionalidad solo se aplican para frecuencia **Mensual (M)**. "
                        "Con frecuencia Semanal (W), los factores no se aplicarán."
                    )
                else:
                    estacionalidad_aplicada = True

            # Core de ventas, con soporte para seasonality_factors si la firma lo acepta
            extra_kwargs = {}
            if usar_estacionalidad and seasonal_weights is not None and freq_code == "M":
                extra_kwargs["seasonality_factors"] = seasonal_weights

            # Mejora #4: Mejor manejo de errores
            det_v = None
            res_v = None
            error_ocurrido = None
            
            try:
                det_v, res_v = forecast_sales(
                    ventas_ext,
                    freq=freq_code,
                    horizon=horizon,
                    **extra_kwargs,
                )
            except TypeError as e:
                # fallback por si la versión del core no acepta seasonality_factors
                try:
                    det_v, res_v = forecast_sales(
                        ventas_ext,
                        freq=freq_code,
                        horizon=horizon,
                    )
                    if usar_estacionalidad and seasonal_weights is not None:
                        st.info("ℹ️ La versión del core no soporta factores de estacionalidad. Se ejecutó sin ajustes estacionales.")
                except Exception as e2:
                    error_ocurrido = f"Error al ejecutar forecast sin estacionalidad: {str(e2)}"
            except ValueError as e:
                error_ocurrido = f"Error de validación: {str(e)}"
            except Exception as e:
                error_ocurrido = f"Error inesperado al calcular proyección: {str(e)}"
            
            if error_ocurrido:
                st.error(f"❌ {error_ocurrido}")
                st.stop()

        # Métricas rápidas
        total_hist = float(ventas_ext["venta_neta"].sum())
        sku_count  = int(ventas_ext["sku"].nunique())
        
        # Debug: Información sobre la lectura de datos
        with st.expander("🔍 Información de depuración (lectura de datos)", expanded=False):
            st.write("**Columnas del sheet original:**")
            st.write(list(ventas_raw.columns))
            st.write(f"**Filas en sheet original:** {len(ventas_raw)}")
            
            # Verificar qué columna se usó para venta_neta
            cols_lc = {str(c).lower(): c for c in ventas_raw.columns}
            def _col(*candidates):
                for key in candidates:
                    key_lc = key.lower()
                    if key_lc in cols_lc:
                        return cols_lc[key_lc]
                return None
            venta_col = _col("venta_neta", "total_linea", "total línea", "total_line")
            if venta_col:
                st.success(f"✅ Columna de venta encontrada: **{venta_col}**")
                total_raw = pd.to_numeric(ventas_raw[venta_col], errors="coerce").fillna(0.0).sum()
                st.write(f"**Total en columna '{venta_col}' del sheet:** ${total_raw:,.0f}")
            else:
                st.warning("⚠️ **No se encontró columna de venta_neta**. Se buscó: 'venta_neta', 'total_linea', 'total línea', 'total_line'")
                st.write("**Columnas disponibles:**", list(ventas_raw.columns))
            
            st.write(f"**Filas después de normalización:** {len(ventas_ext)}")
            st.write(f"**Filas descartadas (sin fecha o SKU válido):** {len(ventas_raw) - len(ventas_ext)}")
            st.write(f"**Total venta_neta después de normalización:** ${total_hist:,.0f}")
            
            # Verificar si hay valores cero
            zeros = (ventas_ext["venta_neta"] == 0.0).sum()
            st.write(f"**Filas con venta_neta = 0:** {zeros} de {len(ventas_ext)}")
            
            # Mostrar muestra de datos normalizados
            st.write("**Muestra de datos normalizados (primeras 5 filas):**")
            st.dataframe(ventas_ext[["fecha", "sku", "venta_neta"]].head(), use_container_width=True)

        # Mejora #5: Indicador de estacionalidad aplicada
        if estacionalidad_aplicada:
            st.success("✅ **Estacionalidad aplicada**: Los factores de estacionalidad se han aplicado correctamente a las predicciones.")
        elif usar_estacionalidad and seasonal_weights is not None and freq_code == "W":
            st.info("ℹ️ **Estacionalidad no aplicada**: Los factores solo funcionan con frecuencia Mensual (M).")

        m1, m2, m3 = st.columns(3)
        m1.metric("SKUs considerados", sku_count)
        m2.metric("Venta histórica total", f"${total_hist:,.0f}")
        if res_v is not None and not res_v.empty and "venta_forecast_total" in res_v.columns:
            total_forecast = float(res_v["venta_forecast_total"].sum())
            m3.metric("Venta proyectada (horizonte)", f"${total_forecast:,.0f}")
        else:
            m3.metric("Venta proyectada (horizonte)", "$0")

        # Tablas
        if res_v is not None and not res_v.empty:
            st.subheader("Resumen de ventas proyectadas por SKU")
            st.dataframe(res_v, use_container_width=True)
            st.download_button(
                "Descargar resumen de ventas (CSV)",
                res_v.to_csv(index=False).encode("utf-8"),
                "ventas_resumen.csv",
                "text/csv",
            )
            
            # Sección de métricas de calidad del forecast
            # Verificar si existen columnas de métricas
            has_metrics = any(col in res_v.columns for col in ["mape", "rmse", "mae", "demand_class", "adi", "cv2"])
            
            if has_metrics:
                st.markdown("---")
                st.subheader("📊 Métricas de calidad del forecast")
                
                # Debug: mostrar qué columnas están disponibles
                if st.checkbox("🔍 Mostrar información de debug", value=False):
                    st.write("**Columnas disponibles en res_v:**")
                    st.write(list(res_v.columns))
                    st.write("**Primera fila de datos:**")
                    st.write(res_v.iloc[0].to_dict() if len(res_v) > 0 else "No hay datos")
                
                # Filtrar SKUs con métricas disponibles
                metrics_cols = ["sku", "modelo", "demand_class", "mape", "rmse", "mae", "adi", "cv2"]
                available_metrics = [col for col in metrics_cols if col in res_v.columns]
                
                if available_metrics:
                    # Asegurarse de que al menos tenemos SKU y modelo
                    if "sku" not in available_metrics:
                        available_metrics.insert(0, "sku")
                    if "modelo" in res_v.columns and "modelo" not in available_metrics:
                        available_metrics.insert(1, "modelo")
                    
                    metrics_df = res_v[available_metrics].copy()
                    
                    # Mostrar cuántos SKUs tienen métricas calculadas
                    if "mape" in metrics_df.columns:
                        n_with_mape = metrics_df["mape"].notna().sum()
                        if n_with_mape < len(metrics_df):
                            st.info(
                                f"ℹ️ {n_with_mape} de {len(metrics_df)} SKUs tienen métricas calculadas. "
                                f"Los demás pueden tener datos históricos insuficientes (< 8 períodos) para validación."
                            )
                    
                    # Formatear métricas para mejor visualización
                    if "mape" in metrics_df.columns:
                        metrics_df["mape"] = metrics_df["mape"].apply(
                            lambda x: f"{x:.2f}%" if pd.notna(x) and x is not None else "N/A"
                        )
                    if "rmse" in metrics_df.columns:
                        metrics_df["rmse"] = metrics_df["rmse"].apply(
                            lambda x: f"${x:,.0f}" if pd.notna(x) and x is not None else "N/A"
                        )
                    if "mae" in metrics_df.columns:
                        metrics_df["mae"] = metrics_df["mae"].apply(
                            lambda x: f"${x:,.0f}" if pd.notna(x) and x is not None else "N/A"
                        )
                    if "adi" in metrics_df.columns:
                        metrics_df["adi"] = metrics_df["adi"].apply(
                            lambda x: f"{x:.2f}" if pd.notna(x) and x is not None else "N/A"
                        )
                    if "cv2" in metrics_df.columns:
                        metrics_df["cv2"] = metrics_df["cv2"].apply(
                            lambda x: f"{x:.3f}" if pd.notna(x) and x is not None else "N/A"
                        )
                    
                    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                else:
                    st.info("ℹ️ Las métricas no están disponibles. Verifica que el core esté calculando las métricas correctamente.")
                
                # Resumen agregado de métricas (solo si hay métricas disponibles)
                if available_metrics and any(col in res_v.columns for col in ["mape", "rmse", "mae"]):
                    with st.expander("📈 Resumen agregado de métricas", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        
                        # MAPE promedio (solo de los que tienen valor)
                        if "mape" in res_v.columns:
                            mape_values = res_v["mape"].dropna()
                            if len(mape_values) > 0:
                                mape_avg = float(mape_values.mean())
                                col1.metric("MAPE promedio", f"{mape_avg:.2f}%", 
                                          help="Mean Absolute Percentage Error - menor es mejor")
                        
                        # RMSE promedio
                        if "rmse" in res_v.columns:
                            rmse_values = res_v["rmse"].dropna()
                            if len(rmse_values) > 0:
                                rmse_avg = float(rmse_values.mean())
                                col2.metric("RMSE promedio", f"${rmse_avg:,.0f}",
                                          help="Root Mean Squared Error - menor es mejor")
                        
                        # MAE promedio
                        if "mae" in res_v.columns:
                            mae_values = res_v["mae"].dropna()
                            if len(mae_values) > 0:
                                mae_avg = float(mae_values.mean())
                                col3.metric("MAE promedio", f"${mae_avg:,.0f}",
                                          help="Mean Absolute Error - menor es mejor")
                        
                        # Distribución de clasificación de demanda
                        if "demand_class" in res_v.columns:
                            st.markdown("**Distribución de clasificación de demanda:**")
                            demand_dist = res_v["demand_class"].value_counts()
                            st.write(demand_dist.to_dict())
                        
                        # Distribución de modelos
                        if "modelo" in res_v.columns:
                            st.markdown("**Distribución de modelos utilizados:**")
                            model_dist = res_v["modelo"].value_counts()
                            st.write(model_dist.to_dict())

        if det_v is not None and not det_v.empty:
            st.subheader("Detalle histórico + forecast")
            st.dataframe(det_v, use_container_width=True)
            st.download_button(
                "Descargar detalle (CSV)",
                det_v.to_csv(index=False).encode("utf-8"),
                "ventas_detalle.csv",
                "text/csv",
            )





































