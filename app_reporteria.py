# app_reporteria.py
# streamlit run app_reporteria.py

import pandas as pd
import streamlit as st
import altair as alt
from urllib.parse import quote
from datetime import timedelta
from typing import Optional

# ============================
# CONFIG BÁSICA
# ============================
DEFAULT_SHEET_ID   = "1Pbjxy_V-NuTbfnN_SLpexkYx_w62Umsg7eBr2qrQJrI"
TAB_VENTAS         = "ventas_raw"
TAB_STOCK          = "stock_snapshot"
TAB_CLIENTES_CONF  = "clientes_config"

# URL de la otra app (predictor) para navegar
PREDICTOR_APP_URL = "http://localhost:8503"

st.set_page_config(page_title="Reportería de Ventas", layout="wide")

# ============================
# HELPERS
# ============================
def read_gsheets(sheet_id: str, tab: str) -> pd.DataFrame:
    sheet_param = quote(tab, safe="")
    url = (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?"
        f"tqx=out:csv&sheet={sheet_param}"
    )
    return pd.read_csv(url)


@st.cache_data
def load_clientes_config() -> Optional[pd.DataFrame]:
    try:
        df = read_gsheets(DEFAULT_SHEET_ID, TAB_CLIENTES_CONF)
        if "activo" in df.columns:
            df = df[df["activo"].astype(str).str.upper().isin(["TRUE", "1", "SI"])]
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception:
        return None


def normalize_ventas(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    out = pd.DataFrame()
    out["fecha"] = pd.to_datetime(df[cols.get("fecha")], errors="coerce")
    out["sku"]   = df[cols.get("sku")].astype(str).str.strip().str.upper()
    qty_col = cols.get("cantidad") or cols.get("qty")
    out["qty"]   = pd.to_numeric(df[qty_col], errors="coerce").fillna(0)
    out = out.dropna(subset=["fecha", "sku"])
    return out


def normalize_stock(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["sku", "stock"])
    cols = {c.lower(): c for c in df.columns}
    sku_c = cols.get("sku") or cols.get("codigo") or list(df.columns)[0]
    stock_c = None
    for c in df.columns:
        if pd.to_numeric(df[c], errors="coerce").notna().sum() > 0:
            stock_c = c
            break
    out = pd.DataFrame()
    out["sku"]   = df[sku_c].astype(str).str.strip().str.upper()
    out["stock"] = pd.to_numeric(df[stock_c], errors="coerce").fillna(0)
    return out


# ============================
# MULTI-TENANT
# ============================
clientes_df = load_clientes_config()

CURRENT_SHEET_ID = DEFAULT_SHEET_ID
CURRENT_TENANT_ID = "default"

params = st.query_params
tenant_from_url = params.get("tenant", None)

if clientes_df is not None and len(clientes_df):
    tenant_ids = clientes_df["tenant_id"].tolist()
    if tenant_from_url and tenant_from_url in tenant_ids:
        tenant_sel = tenant_from_url
    else:
        tenant_sel = st.sidebar.selectbox("Cliente", tenant_ids, index=0)
    row = clientes_df[clientes_df["tenant_id"] == tenant_sel].iloc[0]
    CURRENT_TENANT_ID = row["tenant_id"]
    CURRENT_SHEET_ID  = row.get("sheet_id", DEFAULT_SHEET_ID)

# --- navegación lateral ---
st.sidebar.markdown("### Navegación")
st.sidebar.markdown(
    f"[🧠 Predictor de compras]({PREDICTOR_APP_URL}?tenant={CURRENT_TENANT_ID})"
)
st.sidebar.markdown("---")

# ============================
# UI PRINCIPAL
# ============================
st.title("📊 Reportería de Ventas y Stock ↩")
st.caption(f"Tenant: **{CURRENT_TENANT_ID}**")

with st.expander("Detalles técnicos (oculto para gerencia)", expanded=False):
    st.write(f"Hoja origen: {CURRENT_SHEET_ID}")

colf1, colf2, colf3 = st.columns(3)
dias = colf1.select_slider("Rango de días para el análisis", options=[7, 30, 60, 90], value=30)
sku_filter = colf2.text_input("Filtrar por SKU (opcional)").strip().upper()

# ============================
# CARGA DE DATOS
# ============================
with st.spinner("Cargando datos desde Google Sheets…"):
    ventas_raw = read_gsheets(CURRENT_SHEET_ID, TAB_VENTAS)
    stock_raw  = read_gsheets(CURRENT_SHEET_ID, TAB_STOCK)

ventas = normalize_ventas(ventas_raw)
stock  = normalize_stock(stock_raw)

if ventas.empty:
    st.warning("No hay ventas en la hoja.")
    st.stop()

# ============================
# FECHA BASE = HOY REAL
# ============================
hoy = pd.Timestamp.today().normalize()
desde = hoy - timedelta(days=dias)
colf3.write(f"Hasta: **{hoy.date()}**")

# filtrar ventas por fecha
ventas_rango = ventas[(ventas["fecha"] >= desde) & (ventas["fecha"] <= hoy)]

# aplicar filtro SKU global
if sku_filter:
    ventas_rango = ventas_rango[ventas_rango["sku"] == sku_filter]
    stock = stock[stock["sku"] == sku_filter]

# ============================
# 1. TOP 10 MÁS VENDIDOS
# ============================
st.subheader(f"🏆 Top 10 más vendidos (últimos {dias} días)")

top10 = (
    ventas_rango.groupby("sku", as_index=False)["qty"]
    .sum()
    .sort_values("qty", ascending=False)
    .head(10)
)
st.dataframe(top10, use_container_width=True, hide_index=True)

bar_top10 = (
    alt.Chart(top10)
    .mark_bar()
    .encode(
        x=alt.X("qty:Q", title="Unidades vendidas"),
        y=alt.Y("sku:N", sort="-x", title="SKU"),
        tooltip=["sku", "qty"]
    )
    .properties(height=300)
)
st.altair_chart(bar_top10, use_container_width=True)

# ============================
# 2. EVOLUCIÓN MENSUAL (12 MESES)
# ============================
st.subheader("📈 Evolución de ventas (mensual, últimos 12 meses)")

ventas_12m = ventas[(ventas["fecha"] <= hoy) & (ventas["fecha"] >= hoy - timedelta(days=365))]
ventas_12m["mes"] = ventas_12m["fecha"].dt.to_period("M").astype(str)
if sku_filter:
    ventas_12m = ventas_12m[ventas_12m["sku"] == sku_filter]

ventas_mensual = (
    ventas_12m.groupby("mes", as_index=False)["qty"].sum().sort_values("mes")
)

if not ventas_mensual.empty:
    line_mes = (
        alt.Chart(ventas_mensual)
        .mark_line(point=True)
        .encode(
            x=alt.X("mes:N", title="Mes"),
            y=alt.Y("qty:Q", title="Unidades"),
            tooltip=["mes", "qty"]
        )
        .properties(height=280)
    )
    st.altair_chart(line_mes, use_container_width=True)
else:
    st.info("No hay ventas en los últimos 12 meses para ese filtro.")

# ============================
# 3. PRODUCTOS EN ALZA / EN BAJA (mes pasado vs antepasado)
# ============================
st.subheader("📊 Productos en alza / en baja ↔")

hoy = pd.Timestamp.today().normalize()
ini_mes_actual = hoy.replace(day=1)
ini_mes_m1 = ini_mes_actual - pd.offsets.MonthBegin(1)
ini_mes_m2 = ini_mes_actual - pd.offsets.MonthBegin(2)

ventas_m1 = ventas[(ventas["fecha"] >= ini_mes_m1) & (ventas["fecha"] < ini_mes_actual)]
ventas_m2 = ventas[(ventas["fecha"] >= ini_mes_m2) & (ventas["fecha"] < ini_mes_m1)]

if sku_filter:
    ventas_m1 = ventas_m1[ventas_m1["sku"] == sku_filter]
    ventas_m2 = ventas_m2[ventas_m2["sku"] == sku_filter]

m1 = ventas_m1.groupby("sku")["qty"].sum().rename("qty_cur").reset_index()
m2 = ventas_m2.groupby("sku")["qty"].sum().rename("qty_prev").reset_index()

alzabaja = pd.merge(m1, m2, on="sku", how="outer").fillna(0)
# quitar filas totalmente en cero
alzabaja = alzabaja[(alzabaja["qty_cur"] != 0) | (alzabaja["qty_prev"] != 0)]

col_1, col_2 = st.columns(2)

with col_1:
    st.markdown("✅ **En alza**")
    en_alza = alzabaja[alzabaja["delta"] if "delta" in alzabaja.columns else (alzabaja["qty_cur"] - alzabaja["qty_prev"]) > 0]
# ups, mejor hacerlo claro:

# recalculamos delta de forma explícita
alzabaja["delta"] = alzabaja["qty_cur"] - alzabaja["qty_prev"]

with col_1:
    en_alza = alzabaja[alzabaja["delta"] > 0].sort_values("delta", ascending=False)
    if en_alza.empty:
        st.info("No hay productos en alza para el período.")
    else:
        st.dataframe(
            en_alza.head(20),
            use_container_width=True,
            hide_index=True,
        )

with col_2:
    st.markdown("📉 **En baja**")
    en_baja = alzabaja[alzabaja["delta"] < 0].sort_values("delta", ascending=True)
    if en_baja.empty:
        st.info("No hay productos en baja para el período.")
    else:
        st.dataframe(
            en_baja.head(20),
            use_container_width=True,
            hide_index=True,
        )

# ============================
# 4. STOCK: SOBRE-STOCK Y BAJO STOCK
# ============================
st.subheader("📦 Productos sobre-stockeados")

ultimos_60 = hoy - timedelta(days=60)
ventas_60 = ventas[(ventas["fecha"] >= ultimos_60) & (ventas["fecha"] <= hoy)]

consumo_60 = (
    ventas_60.groupby("sku")["qty"]
    .sum()
    .rename("consumo_60d")
    .reset_index()
)

over = stock.merge(consumo_60, on="sku", how="left").fillna({"consumo_60d": 0})

if over.empty:
    st.info("No hay datos de stock para este cliente / filtro.")
else:
    over["consumo_dia"] = over["consumo_60d"] / 60.0
    over["dias_cobertura"] = 0.0

    mask_sin_consumo = (over["consumo_60d"] == 0) & (over["stock"] > 0)
    if mask_sin_consumo.any():
        over.loc[mask_sin_consumo, "dias_cobertura"] = 9999

    mask_con_consumo = (over["consumo_60d"] > 0)
    if mask_con_consumo.any():
        over.loc[mask_con_consumo, "dias_cobertura"] = (
            over.loc[mask_con_consumo, "stock"] / over.loc[mask_con_consumo, "consumo_dia"]
        )

    UMBRAL_SOBRE = 20
    UMBRAL_BAJO  = 5

    overstock = over[over["dias_cobertura"] >= UMBRAL_SOBRE].sort_values("dias_cobertura", ascending=False)

    if overstock.empty:
        st.info("No se detectaron productos sobre-stockeados con los criterios actuales.")
        top_cobertura = over.sort_values("dias_cobertura", ascending=False).head(20)
        st.write("Top por cobertura (referencia):")
        st.dataframe(
            top_cobertura[["sku", "stock", "consumo_60d", "dias_cobertura"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.dataframe(
            overstock[["sku", "stock", "consumo_60d", "dias_cobertura"]].head(50),
            use_container_width=True,
            hide_index=True,
        )
        chart_over = (
            alt.Chart(overstock.head(20))
            .mark_bar()
            .encode(
                x=alt.X("dias_cobertura:Q", title="Días de cobertura"),
                y=alt.Y("sku:N", sort="-x", title="SKU"),
                tooltip=["sku", "stock", "dias_cobertura"]
            )
            .properties(height=400)
        )
        st.altair_chart(chart_over, use_container_width=True)

    # 5. BAJO STOCK
    st.subheader("📦 Productos con bajo stock (cobertura ≤ 5 días)")

    bajo_stock = over[(over["dias_cobertura"] > 0) & (over["dias_cobertura"] <= UMBRAL_BAJO)]
    bajo_stock = bajo_stock.sort_values("dias_cobertura", ascending=True)

    if bajo_stock.empty:
        st.info("No se detectaron productos con bajo stock.")
    else:
        st.dataframe(
            bajo_stock[["sku", "stock", "consumo_60d", "dias_cobertura"]],
            use_container_width=True,
            hide_index=True,
        )






