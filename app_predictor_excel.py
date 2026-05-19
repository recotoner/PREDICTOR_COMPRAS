# app_predictor_excel.py — Predictor desde Excel (V2), sin modificar app_predictor.py
# Ejecutar desde la raíz del proyecto:
#   streamlit run app_predictor_excel.py

from __future__ import annotations

from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
import unicodedata as ud

import altair as alt
import pandas as pd
import streamlit as st

from predictor_core import forecast_all

try:
    from predictor_sales_core import forecast_sales
except ImportError:
    forecast_sales = None

from V2.excel_config_defaults import default_config_from_union
from V2.excel_kame_loader import (
    load_inbound_oc_kame,
    load_inbound_oc_kame_bytes,
    load_pack_from_folder,
    load_stock_kame,
    load_stock_kame_bytes,
    load_ventas_kame_df,
    oc_recepcion_preflight,
    pick_excel_in_folder,
    read_first_sheet_bytes,
    read_first_sheet_path,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_EXCEL_DIR = ROOT / "V2"

ALLOWED_UPLOAD_EXT = (".xlsx", ".xls")


def _ext_ok(filename: str) -> bool:
    if not filename:
        return False
    low = filename.lower().strip()
    return any(low.endswith(ext) for ext in ALLOWED_UPLOAD_EXT)


def _validate_upload_extension(uploaded, label: str) -> str | None:
    """Si hay archivo subido, valida extensión. Si no hay archivo, no hay error aquí."""
    if uploaded is None:
        return None
    name = getattr(uploaded, "name", "") or ""
    if not _ext_ok(name):
        return (
            f"**{label}**: el archivo «{name}» no tiene extensión permitida "
            f"({', '.join(ALLOWED_UPLOAD_EXT)})."
        )
    return None


def _read_upload_bytes(uploaded) -> tuple[bytes | None, str | None]:
    try:
        data = uploaded.getvalue()
    except Exception as e:
        return None, f"No se pudo leer el archivo en memoria: {e}"
    if not data:
        return None, "El archivo está vacío."
    return data, None


def _validate_canonical_ventas(ventas: pd.DataFrame) -> list[str]:
    msgs: list[str] = []
    if ventas.empty:
        msgs.append("Ventas: no quedó ninguna fila válida (revisa fechas y SKU).")
        return msgs
    bad_dates = ventas["fecha"].isna().sum()
    if bad_dates > 0:
        msgs.append(f"Ventas: {bad_dates} filas con fecha no parseable fueron excluidas.")
    empty_sku = (ventas["sku"].astype(str).str.strip() == "").sum()
    if empty_sku > 0:
        msgs.append(f"Ventas: {empty_sku} filas con SKU vacío fueron excluidas.")
    if not pd.api.types.is_numeric_dtype(ventas["qty"]):
        msgs.append("Ventas: la columna de cantidad no es numérica tras normalizar.")
    if ventas["qty"].isna().all():
        msgs.append("Ventas: todas las cantidades son inválidas (no numéricas).")
    return msgs


def _validate_canonical_stock(stock: pd.DataFrame) -> list[str]:
    msgs: list[str] = []
    if stock.empty:
        msgs.append("Stock: el archivo no produjo filas (revisa columnas SKU y Saldo).")
    elif stock["stock"].isna().all():
        msgs.append("Stock: todas las cantidades de inventario son inválidas.")
    return msgs


def _prediction_tables_to_excel_bytes(
    det: pd.DataFrame, res: pd.DataFrame, prop: pd.DataFrame
) -> bytes:
    """Un libro .xlsx en memoria: hojas Detalle, Resumen, Propuesta."""
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        det.to_excel(writer, sheet_name="Detalle", index=False)
        res.to_excel(writer, sheet_name="Resumen", index=False)
        prop.to_excel(writer, sheet_name="Propuesta", index=False)
    bio.seek(0)
    return bio.getvalue()


def _det_forecast_en_ventana_calendario(
    det_v: pd.DataFrame,
    ventas_ext: pd.DataFrame | None,
    raw_ventas: pd.DataFrame | None,
    month_trim_meta: dict | None,
) -> pd.DataFrame:
    """Detalle forecast solo en los meses de las barras rojas del gráfico."""
    if det_v is None or det_v.empty:
        return pd.DataFrame()
    fc_m = _forecast_monthly_after_last_complete(
        det_v,
        ventas_ext=ventas_ext,
        raw_ventas=raw_ventas,
        month_trim_meta=month_trim_meta,
    )
    if fc_m.empty:
        return pd.DataFrame()
    allowed = {str(p) for p in fc_m.index}
    mask_fc = det_v["tipo"].astype(str).str.strip().str.lower() == "forecast"
    x = det_v.loc[mask_fc].copy()
    x["fecha"] = _coerce_fecha_cl(x["fecha"])
    x = x.dropna(subset=["fecha"])
    x["periodo_m"] = x["fecha"].dt.to_period("M").astype(str)
    return x.loc[x["periodo_m"].isin(allowed)].drop(columns=["periodo_m"])


def _ventas_forecast_to_excel_bytes(
    det_v: pd.DataFrame,
    res_v: pd.DataFrame | None,
    ventas_ext: pd.DataFrame | None,
    raw_ventas: pd.DataFrame | None,
    month_trim_meta: dict | None,
    sku_count: int,
    total_hist: float,
) -> bytes:
    """
    Excel alineado con la tarjeta «Venta proyectada (horizonte)» y barras rojas del gráfico.
  Hojas: Totales, Proyeccion_mensual, Detalle_proyeccion; Resumen_SKU solo referencia.
    """
    fc_m = _forecast_monthly_after_last_complete(
        det_v,
        ventas_ext=ventas_ext,
        raw_ventas=raw_ventas,
        month_trim_meta=month_trim_meta,
    )
    total_fc = float(fc_m.sum()) if not fc_m.empty else 0.0

    proy_mensual = pd.DataFrame(
        {
            "periodo": [str(p) for p in fc_m.index],
            "mes": [
                _mes_label_mmm_yyyy(
                    (pd.Period(p, freq="M") if not isinstance(p, pd.Period) else p).to_timestamp(
                        how="start"
                    )
                )
                for p in fc_m.index
            ],
            "monto_proyectado_CLP": fc_m.values,
        }
    )
    if not proy_mensual.empty:
        proy_mensual["acumulado_CLP"] = proy_mensual["monto_proyectado_CLP"].cumsum()

    totales = pd.DataFrame(
        [
            {
                "concepto": "Venta proyectada (horizonte)",
                "monto_CLP": total_fc,
            },
            {
                "concepto": "SKUs considerados",
                "monto_CLP": float(sku_count),
            },
            {
                "concepto": "Venta histórica total (entrenamiento)",
                "monto_CLP": float(total_hist),
            },
        ]
    )

    nota = pd.DataFrame(
        [
            {
                "nota": (
                    "El total de la tarjeta y la hoja Proyeccion_mensual coinciden (suma de barras rojas). "
                    "La hoja Resumen_SKU puede sumar más si se totaliza venta_forecast_total: "
                    "cada SKU usa su propio calendario de 6 meses."
                )
            }
        ]
    )

    det_fc = _det_forecast_en_ventana_calendario(
        det_v, ventas_ext, raw_ventas, month_trim_meta
    )
    det_show = _sort_sales_det_for_display(det_fc) if not det_fc.empty else det_fc

    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        nota.to_excel(writer, sheet_name="Leeme", index=False)
        totales.to_excel(writer, sheet_name="Totales", index=False)
        proy_mensual.to_excel(writer, sheet_name="Proyeccion_mensual", index=False)
        if det_show is not None and not det_show.empty:
            det_show.to_excel(writer, sheet_name="Detalle_proyeccion", index=False)
        if res_v is not None and not res_v.empty:
            res_v.to_excel(writer, sheet_name="Resumen_SKU_referencia", index=False)
    bio.seek(0)
    return bio.getvalue()


def _render_ventas_excel_download(
    det_v: pd.DataFrame,
    res_v: pd.DataFrame | None,
    ventas_ext: pd.DataFrame | None,
    raw_ventas: pd.DataFrame | None,
    month_trim_meta: dict | None,
    sku_count: int,
    total_hist: float,
    total_forecast: float,
    *,
    button_key: str,
) -> None:
    """Botón de descarga Excel alineado con la tarjeta y el gráfico."""
    st.markdown("### 📥 Exportar proyección")
    st.caption(
        f"Total a exportar (igual que la tarjeta verde): **${total_forecast:,.0f}**. "
        "Hojas **Totales** y **Proyeccion_mensual** en el archivo."
    )
    try:
        xlsx_ventas = _ventas_forecast_to_excel_bytes(
            det_v,
            res_v,
            ventas_ext=ventas_ext,
            raw_ventas=raw_ventas,
            month_trim_meta=month_trim_meta,
            sku_count=sku_count,
            total_hist=total_hist,
        )
        st.download_button(
            "⬇️ Descargar proyección de ventas (Excel)",
            xlsx_ventas,
            "ventas_proyeccion.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=button_key,
            use_container_width=True,
        )
    except ImportError:
        st.warning("Instale **openpyxl** para exportar: `pip install openpyxl`")
    except Exception as exc:
        st.error(f"No se pudo generar el Excel: {exc}")


SESSION_RUN_KEY = "excel_v2_full_run"
SESSION_VENTAS_FORECAST_KEY = "excel_v2_ventas_forecast_output"
# st.tabs no persiste la pestaña activa tras cada rerun; el radio con key sí.
SESSION_SECTION_NAV_KEY = "excel_v2_section_nav"
_SECTION_LABELS = [
    "📦 Predictor de compras",
    "📊 Reportería de ventas",
    "📈 Predictor de ventas",
]

_PREDICTOR_EXCEL_CSS = """
<style>
    #MainMenu { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
    footer[data-testid="stFooter"] { visibility: hidden; }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    [data-testid="stAppViewContainer"] {
        background-color: #f4f6f9;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #343a40 0%, #2b3035 100%);
        border-right: 1px solid #23272b;
    }
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] small {
        color: #ecf0f1 !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }

    .welcome-banner {
        background: linear-gradient(135deg, #2d5016 0%, #4a7c2a 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.12);
    }
    .welcome-banner h3 {
        color: white !important;
        margin: 0;
        font-size: 1.25rem;
        font-weight: 700;
    }
    .welcome-banner p {
        margin: 0.45rem 0 0 0;
        opacity: 0.92;
        font-size: 0.95rem;
    }

    .dashboard-title {
        color: #2c3e50;
        margin: 0 0 0.25rem 0;
        font-size: 1.75rem;
        font-weight: 700;
    }
    .dashboard-subtitle {
        color: #5a6c7d;
        margin: 0 0 1rem 0;
        font-size: 1.05rem;
        font-weight: 500;
    }
    .company-pill {
        text-align: right;
        padding: 0.65rem 1rem;
        background-color: #eef2f6;
        border-radius: 8px;
        border: 1px solid #d7e3ef;
        color: #2c3e50;
        font-weight: 600;
    }

    h1, h2, h3 {
        color: #2c3e50;
    }
    .main h2 {
        border-bottom: 2px solid #4a7c2a;
        padding-bottom: 0.35rem;
        margin-top: 1.25rem;
    }

    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.25s ease;
        border: none !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2d5016 0%, #4a7c2a 100%) !important;
        color: white !important;
        font-weight: 600 !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1f3a0f 0%, #2d5016 100%) !important;
    }

    [data-testid="stMetricContainer"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8eef5 100%);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
        border: 1px solid #dde5ee;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        font-weight: 600;
        color: #2c3e50;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.75rem;
        font-weight: 700;
        color: #1f3a0f;
    }

    div[data-testid="stRadio"] > div {
        gap: 0.5rem;
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"] {
        background: #eef3f8;
        border: 1px solid #d2dde8;
        border-radius: 10px;
        padding: 0.45rem 0.85rem;
        font-weight: 600;
        color: #26415a;
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
        background: linear-gradient(135deg, #2d5016 0%, #4a7c2a 100%) !important;
        color: #ffffff !important;
        border-color: transparent !important;
        box-shadow: 0 3px 8px rgba(45, 80, 22, 0.25);
    }

    .stAlert {
        border-radius: 8px;
    }

    .kpi-card {
        color: white;
        padding: 1.35rem 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        min-height: 110px;
    }
    .kpi-card h3 {
        color: white !important;
        margin: 0 0 0.5rem 0;
        font-size: 0.9rem;
        opacity: 0.95;
        font-weight: 600;
        border: none !important;
        padding: 0 !important;
    }
    .kpi-card h2 {
        color: white !important;
        margin: 0;
        font-size: 1.85rem;
        font-weight: 700;
        border: none !important;
    }

    .app-footer-kappo {
        text-align: center;
        padding: 1rem 1rem 0.5rem;
        margin-top: 2rem;
        border-top: 1px solid #ced4da;
        color: #6c757d;
        font-size: 0.88rem;
        background: #f8f9fa;
        border-radius: 0 0 8px 8px;
    }

    .upload-card {
        background: rgba(255, 255, 255, 0.07);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 10px;
        padding: 0.65rem 0.85rem 0.25rem;
        margin-bottom: 0.85rem;
    }
    .upload-card-title {
        color: #ffffff !important;
        font-weight: 700;
        font-size: 0.95rem;
        margin: 0 0 0.15rem 0;
    }
    .upload-card-hint {
        color: #b8c5d1 !important;
        font-size: 0.78rem;
        margin: 0 0 0.5rem 0;
        line-height: 1.3;
    }
    .upload-status-ok {
        color: #7dcea0 !important;
        font-size: 0.8rem;
        margin: 0.35rem 0 0 0;
        font-weight: 600;
    }
    .upload-status-pending {
        color: #adb5bd !important;
        font-size: 0.78rem;
        margin: 0.35rem 0 0 0;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background: transparent !important;
        border-radius: 8px;
        padding: 0.15rem 0;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] label,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] small,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] span {
        color: #dee2e6 !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section {
        background: rgba(0, 0, 0, 0.22) !important;
        border: 1px dashed rgba(255, 255, 255, 0.38) !important;
        border-radius: 8px !important;
        padding: 0.75rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section > div {
        background: transparent !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button {
        background: #495057 !important;
        color: #ffffff !important;
        border: 1px solid #6c757d !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover {
        background: #5a6268 !important;
        border-color: #adb5bd !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] {
        background: rgba(52, 58, 64, 0.95) !important;
        border: 1px dashed rgba(255, 255, 255, 0.35) !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] * {
        color: #e9ecef !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section div,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section span,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section p {
        background-color: transparent !important;
        color: #dee2e6 !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] p {
        color: #ced4da !important;
    }
</style>
"""


def _inject_predictor_excel_theme() -> None:
    st.markdown(_PREDICTOR_EXCEL_CSS, unsafe_allow_html=True)


def _render_predictor_excel_header() -> None:
    st.markdown(
        """
        <div class="welcome-banner">
            <h3>👋 Bienvenido al Predictor de Compras (Excel)</h3>
            <p>Gestiona predicciones de compras y ventas desde archivos Kame de forma inteligente.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col_header1, col_header2 = st.columns([3, 1])
    with col_header1:
        st.markdown(
            '<p class="dashboard-title">📦 Dashboard Predictor de Compras</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="dashboard-subtitle">Proyección desde Excel — ventas, stock y órdenes de compra</p>',
            unsafe_allow_html=True,
        )
    with col_header2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="company-pill">
                📊 Excel V2 · Kame
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("---")


def _render_predictor_excel_footer() -> None:
    st.markdown(
        """
        <div class="app-footer-kappo">
            desarrollado por Henry Rubilar - SOLUCIONES TECNOLOGICAS KAPPO LIMITADA
        </div>
        """,
        unsafe_allow_html=True,
    )


def _html_kpi_card(label: str, value: str, gradient: str) -> str:
    return f"""
    <div class="kpi-card" style="background: linear-gradient(135deg, {gradient});">
        <h3>{label}</h3>
        <h2>{value}</h2>
    </div>
    """


def _upload_file_status_html(uploaded) -> str:
    if uploaded is None:
        return (
            '<p class="upload-status-pending">Sin archivo — '
            "si falta, se buscará en carpeta V2/ al ejecutar</p>"
        )
    name = getattr(uploaded, "name", "") or "archivo"
    return f'<p class="upload-status-ok">✅ {name}</p>'


def _render_excel_upload_sidebar() -> tuple:
    """UI de carga en sidebar (modo subir archivos). Devuelve (up_ventas, up_stock, up_oc)."""
    with st.sidebar:
        st.markdown("### 📁 Carga de archivos")
        st.caption("Solo en esta sesión · .xlsx / .xls · no se guarda en disco")

        st.markdown(
            """
            <div class="upload-card">
                <p class="upload-card-title">📊 Ventas (histórico)</p>
                <p class="upload-card-hint">fecha, SKU, cantidad · export Kame</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        up_ventas = st.file_uploader(
            "Seleccionar Excel de ventas",
            type=["xlsx", "xls"],
            label_visibility="collapsed",
            key="up_ventas",
        )
        st.markdown(_upload_file_status_html(up_ventas), unsafe_allow_html=True)

        st.markdown(
            """
            <div class="upload-card">
                <p class="upload-card-title">📦 Stock / inventario</p>
                <p class="upload-card-hint">SKU y saldo (bodega)</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        up_stock = st.file_uploader(
            "Seleccionar Excel de stock",
            type=["xlsx", "xls"],
            label_visibility="collapsed",
            key="up_stock",
        )
        st.markdown(_upload_file_status_html(up_stock), unsafe_allow_html=True)

        st.markdown(
            """
            <div class="upload-card">
                <p class="upload-card-title">🚚 Órdenes de compra</p>
                <p class="upload-card-hint">Recepción Pendiente · Por recibir</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        up_oc = st.file_uploader(
            "Seleccionar Excel de OC",
            type=["xlsx", "xls"],
            label_visibility="collapsed",
            key="up_oc",
        )
        st.markdown(_upload_file_status_html(up_oc), unsafe_allow_html=True)

    return up_ventas, up_stock, up_oc


def _style_reporteria_table(df: pd.DataFrame):
    """Centra columnas numéricas (compatible con Streamlit sin alignment en NumberColumn)."""
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    styled = df.style
    if num_cols:
        styled = styled.set_properties(subset=num_cols, **{"text-align": "center"})
    float_cols = [c for c in num_cols if c in ("dias_cobertura", "consumo_dia")]
    if float_cols:
        styled = styled.format({c: "{:.1f}" for c in float_cols}, subset=float_cols)
    int_cols = [c for c in num_cols if c not in float_cols]
    if int_cols:
        styled = styled.format({c: "{:,.0f}" for c in int_cols}, subset=int_cols)
    return styled


def _show_reporteria_table(df: pd.DataFrame) -> None:
    st.dataframe(
        _style_reporteria_table(df),
        use_container_width=True,
        hide_index=True,
    )


def _ascii_fold(s: str) -> str:
    t = str(s).replace("\u00a0", " ").strip()
    t = ud.normalize("NFKD", t)
    return "".join(ch for ch in t if not ud.combining(ch)).lower()


def _columns_folded(df: pd.DataFrame) -> dict[str, str]:
    return {_ascii_fold(str(c)): str(c) for c in df.columns}


def _resolve_fecha_col(df: pd.DataFrame) -> str | None:
    folded = _columns_folded(df)
    return folded.get("fecha")


def _resolve_monto_linea_col(df: pd.DataFrame) -> str | None:
    """
    Columna de monto por línea. Prioriza **Total Línea** (export Kame) sobre venta_neta.
    """
    folded = _columns_folded(df)
    for exact in ("total linea", "total_linea", "total line"):
        if exact in folded:
            return folded[exact]
    for key, orig in folded.items():
        if "costo" in key or "margen" in key:
            continue
        if "total" in key and "linea" in key:
            return orig
    for target in ("venta neta", "venta_neta"):
        if target in folded:
            return folded[target]
    return None


def _resolve_sku_col(df: pd.DataFrame) -> str | None:
    folded = _columns_folded(df)
    for target in ("sku", "sku2"):
        if target in folded:
            return folded[target]
    return None


def _resolve_qty_col(df: pd.DataFrame) -> str | None:
    folded = _columns_folded(df)
    for target in ("cantidad", "qty", "unidades"):
        if target in folded:
            return folded[target]
    return None


def _coerce_fecha_cl(values) -> pd.Series:
    """Fechas CL: serial Excel numérico o texto DD/MM (dayfirst=True)."""
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_datetime(values, unit="D", origin="1899-12-30", errors="coerce")
    return pd.to_datetime(values, errors="coerce", dayfirst=True)


_SIN_SKU_BUCKET = "SIN_SKU"


def _normalize_sku_ventas_sales(series: pd.Series) -> pd.Series:
    """SKU canónico para forecast; filas sin SKU van a SIN_SKU (no se descartan $)."""
    s = series.astype(str)
    s = s.replace(["nan", "NaN", "None", "<NA>", "NaT"], "")
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.str.strip().str.upper()
    bad = (s == "") | (s == "NAN") | (s == "NONE")
    return s.where(~bad, _SIN_SKU_BUCKET)


def _monthly_totals_from_raw_sheet(
    raw: pd.DataFrame,
) -> tuple[pd.Series, dict[str, str | None]]:
    """Suma por Period('M') directo desde hoja cruda (validación vs Excel)."""
    meta: dict[str, str | None] = {
        "fecha_col": None,
        "monto_col": None,
        "sku_col": _resolve_sku_col(raw) if raw is not None else None,
    }
    if raw is None or raw.empty:
        return pd.Series(dtype=float), meta
    fecha_col = _resolve_fecha_col(raw)
    monto_col = _resolve_monto_linea_col(raw)
    meta["fecha_col"] = fecha_col
    meta["monto_col"] = monto_col
    if not fecha_col or not monto_col:
        return pd.Series(dtype=float), meta
    tmp = pd.DataFrame()
    tmp["fecha"] = _coerce_fecha_cl(raw[fecha_col])
    tmp["monto"] = pd.to_numeric(raw[monto_col], errors="coerce").fillna(0.0)
    tmp = tmp.dropna(subset=["fecha"])
    tmp["periodo_m"] = tmp["fecha"].dt.to_period("M")
    return tmp.groupby("periodo_m", observed=True)["monto"].sum().sort_index(), meta


def _normalize_ventas_for_sales_excel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filas con fecha, sku (o SIN_SKU), qty y monto en `venta_neta` desde **Total Línea**.
    Misma columna y filas con fecha válida que `_monthly_totals_from_raw_sheet`, para que
    la suma mensual coincida con la hoja cruda al agregar por período.
    """
    base_cols = [
        "fecha", "sku", "qty", "venta_neta", "precio_unitario",
        "rut", "razon_social", "tipo_documento", "folio", "glosa",
        "sucursal", "unidad_negocio", "familia", "vendedor",
        "lista_precio", "producto",
    ]
    if df is None or df.empty:
        return pd.DataFrame(columns=base_cols)

    fecha_col = _resolve_fecha_col(df)
    sku_col = _resolve_sku_col(df)
    qty_col = _resolve_qty_col(df)
    venta_col = _resolve_monto_linea_col(df)
    if not fecha_col or not sku_col or not qty_col:
        return pd.DataFrame(columns=base_cols)
    out = pd.DataFrame()
    out["fecha"] = _coerce_fecha_cl(df[fecha_col])
    out["sku"] = _normalize_sku_ventas_sales(df[sku_col])
    out["qty"] = pd.to_numeric(df[qty_col], errors="coerce").fillna(0)
    if venta_col:
        out["venta_neta"] = pd.to_numeric(df[venta_col], errors="coerce").fillna(0.0)
    else:
        out["venta_neta"] = 0.0
    folded = _columns_folded(df)

    def _col_fold(*candidates):
        for key in candidates:
            k = _ascii_fold(key)
            if k in folded:
                return folded[k]
        return None

    precio_col = _col_fold("precio_unitario", "precio un.", "precio_un", "precio unitario")
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
        c = _col_fold(*candidates)
        if c:
            out[logical] = df[c].astype(str).fillna("").str.strip()
        else:
            out[logical] = ""
    out = out.dropna(subset=["fecha"])
    return out


def _filter_raw_ventas_by_sku(raw: pd.DataFrame, sku_upper: str) -> pd.DataFrame:
    if raw is None or raw.empty or not sku_upper:
        return raw
    cols_lc = {str(c).replace("\u00a0", " ").strip().lower(): c for c in raw.columns}
    sku_col = cols_lc.get("sku")
    if not sku_col:
        return raw
    sk = raw[sku_col].astype(str).str.replace(r"\.0$", "", regex=True).str.strip().str.upper()
    return raw.loc[sk == sku_upper].copy()


def _fecha_corte_desde_ventas(ventas: pd.DataFrame) -> pd.Timestamp:
    """Fecha de corte para reportería = última fecha con venta en el archivo (no hoy)."""
    if ventas is None or ventas.empty or "fecha" not in ventas.columns:
        return pd.Timestamp.today().normalize()
    ts = _coerce_fecha_cl(ventas["fecha"]).max()
    if pd.isna(ts):
        return pd.Timestamp.today().normalize()
    return pd.Timestamp(ts).normalize()


def _sales_forecast_month_trim_meta(df: pd.DataFrame) -> dict:
    """
    Detecta si el último mes con ventas en el archivo está incompleto (corte antes del fin
    de mes o mes calendario aún abierto). Solo meses cerrados deben alimentar el ETS.
    """
    empty_meta: dict = {
        "excluded": False,
        "max_fecha": None,
        "incomplete_period": None,
        "last_complete_period": None,
        "forecast_from_period": None,
        "rows_dropped": 0,
    }
    if df is None or df.empty or "fecha" not in df.columns:
        return empty_meta

    fechas = _coerce_fecha_cl(df["fecha"])
    max_f = fechas.max()
    if pd.isna(max_f):
        return empty_meta

    max_f = pd.Timestamp(max_f).normalize()
    p_max = max_f.to_period("M")
    month_end = p_max.to_timestamp(how="end").normalize()
    today = pd.Timestamp.today().normalize()
    calendar_still_open = today.to_period("M") == p_max and today < month_end
    data_before_month_end = max_f < month_end

    if not (calendar_still_open or data_before_month_end):
        return {
            "excluded": False,
            "max_fecha": max_f,
            "incomplete_period": None,
            "last_complete_period": p_max,
            "forecast_from_period": p_max + 1,
            "rows_dropped": 0,
        }

    fechas_norm = fechas.dt.to_period("M")
    last_complete_period = fechas_norm[fechas_norm < p_max].max()
    forecast_from = p_max + 1
    if pd.isna(last_complete_period):
        last_complete_period = None
        forecast_from = p_max + 1

    return {
        "excluded": True,
        "max_fecha": max_f,
        "incomplete_period": p_max,
        "last_complete_period": last_complete_period,
        "forecast_from_period": forecast_from,
        "rows_dropped": int((fechas_norm == p_max).sum()),
    }


def _drop_incomplete_last_month_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Quita filas del mes incompleto; el core proyecta desde el mes siguiente al último cerrado."""
    meta = _sales_forecast_month_trim_meta(df)
    if not meta.get("excluded") or meta.get("incomplete_period") is None:
        return df.copy(), meta

    fechas = _coerce_fecha_cl(df["fecha"])
    p_inc = meta["incomplete_period"]
    mask = fechas.dt.to_period("M") != p_inc
    out = df.loc[mask].copy()
    meta["rows_dropped"] = int((~mask).sum())
    return out, meta


def _drop_incomplete_period_from_monthly_series(
    monthly: pd.Series, incomplete_period: pd.Period | None
) -> pd.Series:
    if monthly is None or monthly.empty or incomplete_period is None:
        return monthly
    idx = monthly.index
    if isinstance(idx, pd.PeriodIndex):
        keep = idx != incomplete_period
    else:
        keep = pd.Index(
            [
                (pd.Period(p, freq="M") if not isinstance(p, pd.Period) else p)
                != incomplete_period
                for p in idx
            ]
        )
    return monthly.loc[keep]


def _filter_invalid_skus_ventas_sales(ventas_ext: pd.DataFrame) -> pd.DataFrame:
    """Re-mapea SKU inválidos a SIN_SKU (no elimina filas ni montos)."""
    if ventas_ext is None or ventas_ext.empty:
        return ventas_ext
    out = ventas_ext.copy()
    out["sku"] = _normalize_sku_ventas_sales(out["sku"])
    return out


def _postprocess_forecast_cuts_and_alerts(
    det_v: pd.DataFrame | None,
    res_v: pd.DataFrame | None,
    horizon: int,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    - SKUs con <4 períodos históricos: tope de forecast total = promedio por período × horizonte
      y se escala el detalle forecast acorde.
    - Alerta si el forecast total (tras tope) supera 3× promedio mensual hist. × horizonte.
    """
    if res_v is None or res_v.empty:
        return det_v, res_v
    res = res_v.copy()
    res["alerta_inflacion"] = ""
    det = det_v.copy() if det_v is not None and not det_v.empty else None
    h = max(int(horizon), 1)

    for i in res.index:
        sku = str(res.at[i, "sku"])
        ph_raw = res.at[i, "periodos_hist"]
        ph = int(ph_raw) if pd.notna(ph_raw) else 0
        vht = float(res.at[i, "venta_hist_total"])
        mean_p = (vht / ph) if ph > 0 else 0.0
        fc_tot = float(res.at[i, "venta_forecast_total"])
        cap_total = mean_p * h

        if det is not None and ph > 0 and ph < 4 and fc_tot > cap_total + 1e-8:
            scale = cap_total / fc_tot if fc_tot > 1e-12 else 1.0
            mask = (det["sku"].astype(str) == sku) & (det["tipo"].astype(str).str.lower() == "forecast")
            if mask.any():
                vn = pd.to_numeric(det.loc[mask, "venta_neta"], errors="coerce").fillna(0.0) * scale
                det.loc[mask, "venta_neta"] = vn
            res.at[i, "venta_forecast_total"] = cap_total
            fc_tot = cap_total

        thr = 3.0 * mean_p * h
        if ph > 0 and mean_p > 1e-12 and fc_tot > thr + 1e-8:
            res.at[i, "alerta_inflacion"] = (
                f"Forecast > 3× prom. mensual hist.×h ({fc_tot:,.0f} > {thr:,.0f})"
            )

    return det, res


def _sort_sales_det_for_display(det_v: pd.DataFrame) -> pd.DataFrame:
    """Histórico antes que forecast, orden cronológico por SKU (evita lecturas confusas)."""
    if det_v is None or det_v.empty:
        return det_v
    out = det_v.copy()
    tipo_ord = out["tipo"].astype(str).str.lower().map({"hist": 0, "forecast": 1}).fillna(2)
    out = out.assign(_tipo_ord=tipo_ord)
    out = out.sort_values(["sku", "_tipo_ord", "fecha"], kind="mergesort").drop(columns=["_tipo_ord"])
    return out


_MMM_EN = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)


def _mes_label_mmm_yyyy(ts: pd.Timestamp) -> str:
    t = pd.Timestamp(ts)
    return f"{_MMM_EN[t.month - 1]}-{t.year}"


def _line_level_monthly_totals(
    df: pd.DataFrame, monto_col: str = "venta_neta"
) -> pd.Series:
    """Suma por Period('M') desde filas (misma lógica que validar contra el Excel fuente)."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    x = df.copy()
    x["fecha"] = _coerce_fecha_cl(x["fecha"])
    x = x.dropna(subset=["fecha"])
    x["monto"] = pd.to_numeric(x[monto_col], errors="coerce").fillna(0.0)
    x["periodo_m"] = x["fecha"].dt.to_period("M")
    return x.groupby("periodo_m", observed=True)["monto"].sum().sort_index()


def _det_monthly_totals(det_v: pd.DataFrame, tipo: str) -> pd.Series:
    """Suma detalle del core por Period('M') y tipo (hist o forecast)."""
    if det_v is None or det_v.empty:
        return pd.Series(dtype=float)
    mask = det_v["tipo"].astype(str).str.strip().str.lower() == tipo
    if not mask.any():
        return pd.Series(dtype=float)
    x = det_v.loc[mask].copy()
    x["fecha"] = _coerce_fecha_cl(x["fecha"])
    x = x.dropna(subset=["fecha"])
    x["monto"] = pd.to_numeric(x["venta_neta"], errors="coerce").fillna(0.0)
    x["periodo_m"] = x["fecha"].dt.to_period("M")
    return x.groupby("periodo_m", observed=True)["monto"].sum().sort_index()


def _debug_monthly_validation_table(
    ventas_ext: pd.DataFrame | None,
    det_v: pd.DataFrame | None,
    monthly_chart: pd.DataFrame | None,
    raw_ventas: pd.DataFrame | None = None,
    month_trim_meta: dict | None = None,
) -> pd.DataFrame:
    """Comparación temporal: hoja cruda, normalizado, det del core y gráfico."""
    inc_period = (
        month_trim_meta.get("incomplete_period") if month_trim_meta else None
    )
    raw_m, raw_meta = _monthly_totals_from_raw_sheet(raw_ventas)
    raw_m = _drop_incomplete_period_from_monthly_series(raw_m, inc_period)
    excel_m = (
        _line_level_monthly_totals(ventas_ext)
        if ventas_ext is not None and not ventas_ext.empty
        else pd.Series(dtype=float)
    )
    det_hist_m = _det_monthly_totals(det_v, "hist") if det_v is not None else pd.Series(dtype=float)
    det_fc_m = _det_monthly_totals(det_v, "forecast") if det_v is not None else pd.Series(dtype=float)

    hist_ref = raw_m if len(raw_m) >= 1 else excel_m
    prom_hist_6 = float(hist_ref.tail(6).mean()) if len(hist_ref) >= 1 else float("nan")

    chart_map: dict = {}
    if monthly_chart is not None and not monthly_chart.empty and "periodo_m" in monthly_chart.columns:
        for _, row in monthly_chart.iterrows():
            p = row["periodo_m"]
            per = pd.Period(p, freq="M") if not isinstance(p, pd.Period) else p
            chart_map[per] = (float(row["monto"]), str(row.get("tipo_lc", "")))

    periods = sorted(
        set(raw_m.index)
        | set(excel_m.index)
        | set(det_hist_m.index)
        | set(det_fc_m.index)
        | set(chart_map.keys()),
        key=lambda x: pd.Period(x) if not isinstance(x, pd.Period) else x,
    )
    rows: list[dict] = []
    for p in periods:
        per = pd.Period(p, freq="M") if not isinstance(p, pd.Period) else p
        ch_monto, ch_tipo = chart_map.get(per, (float("nan"), ""))
        fc_val = det_fc_m.get(per, float("nan"))
        ratio_fc = float("nan")
        if pd.notna(fc_val) and prom_hist_6 > 0:
            ratio_fc = fc_val / prom_hist_6
        rows.append(
            {
                "periodo": str(per),
                "mes": _mes_label_mmm_yyyy(per.to_timestamp()),
                "excel_hoja_CLP": raw_m.get(per, float("nan")),
                "excel_ventas_ext_CLP": excel_m.get(per, float("nan")),
                "det_hist_CLP": det_hist_m.get(per, float("nan")),
                "det_forecast_CLP": fc_val,
                "grafico_CLP": ch_monto,
                "grafico_tipo": ch_tipo,
                "fc_vs_prom_hist_6x": ratio_fc,
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["raw_meta"] = raw_meta
    out.attrs["prom_hist_6"] = prom_hist_6
    return out


def _hist_monthly_for_sales_chart(
    det_v: pd.DataFrame | None,
    ventas_ext: pd.DataFrame | None = None,
    raw_ventas: pd.DataFrame | None = None,
    month_trim_meta: dict | None = None,
) -> pd.Series:
    """Serie mensual histórica (meses completos) alineada al gráfico de ventas."""
    inc_period = (
        month_trim_meta.get("incomplete_period") if month_trim_meta else None
    )
    if raw_ventas is not None and not raw_ventas.empty:
        raw_use, _ = _drop_incomplete_last_month_rows(raw_ventas)
        hist_by_m, _ = _monthly_totals_from_raw_sheet(raw_use)
    elif ventas_ext is not None and not ventas_ext.empty:
        hist_by_m = _line_level_monthly_totals(ventas_ext)
    elif det_v is not None and not det_v.empty:
        hist_by_m = _det_monthly_totals(det_v, "hist")
    else:
        hist_by_m = pd.Series(dtype=float)
    return _drop_incomplete_period_from_monthly_series(hist_by_m, inc_period)


def _forecast_monthly_after_last_complete(
    det_v: pd.DataFrame | None,
    ventas_ext: pd.DataFrame | None = None,
    raw_ventas: pd.DataFrame | None = None,
    month_trim_meta: dict | None = None,
) -> pd.Series:
    """
    Forecast agregado por mes calendario, solo períodos posteriores al último mes
    completo global (misma regla que las barras rojas del gráfico).
    """
    if det_v is None or det_v.empty:
        return pd.Series(dtype=float)
    hist_by_m = _hist_monthly_for_sales_chart(
        det_v, ventas_ext=ventas_ext, raw_ventas=raw_ventas, month_trim_meta=month_trim_meta
    )
    fc_by_m = _det_monthly_totals(det_v, "forecast")
    if fc_by_m.empty:
        return pd.Series(dtype=float)
    last_hist_period = hist_by_m.index.max() if not hist_by_m.empty else None
    if last_hist_period is not None:
        return fc_by_m.loc[fc_by_m.index > last_hist_period]
    return fc_by_m


def _forecast_total_global_calendar_window(
    det_v: pd.DataFrame | None,
    ventas_ext: pd.DataFrame | None = None,
    raw_ventas: pd.DataFrame | None = None,
    month_trim_meta: dict | None = None,
) -> float:
    fc_only = _forecast_monthly_after_last_complete(
        det_v,
        ventas_ext=ventas_ext,
        raw_ventas=raw_ventas,
        month_trim_meta=month_trim_meta,
    )
    return float(fc_only.sum()) if not fc_only.empty else 0.0


def _scale_det_forecast_calendar_month(
    det_v: pd.DataFrame, period: pd.Period, target_total: float
) -> pd.DataFrame:
    """Escala filas forecast de un mes calendario para que la suma global sea target_total."""
    det = det_v.copy()
    mask_fc = det["tipo"].astype(str).str.strip().str.lower() == "forecast"
    if not mask_fc.any():
        return det
    fechas = _coerce_fecha_cl(det.loc[mask_fc, "fecha"])
    per = pd.Period(period, freq="M") if not isinstance(period, pd.Period) else period
    month_mask = mask_fc & (fechas.dt.to_period("M") == per)
    if not month_mask.any():
        return det
    vn = pd.to_numeric(det.loc[month_mask, "venta_neta"], errors="coerce").fillna(0.0)
    current = float(vn.sum())
    if current <= 1e-8:
        return det
    det.loc[month_mask, "venta_neta"] = vn * (float(target_total) / current)
    return det


def _recompute_res_v_forecast_totals_from_det(
    det_v: pd.DataFrame | None, res_v: pd.DataFrame | None
) -> pd.DataFrame | None:
    if res_v is None or res_v.empty or det_v is None or det_v.empty:
        return res_v
    res = res_v.copy()
    mask_fc = det_v["tipo"].astype(str).str.strip().str.lower() == "forecast"
    if not mask_fc.any():
        return res
    sub = det_v.loc[mask_fc].copy()
    sub["venta_neta"] = pd.to_numeric(sub["venta_neta"], errors="coerce").fillna(0.0)
    fc_by_sku = sub.groupby(sub["sku"].astype(str), observed=True)["venta_neta"].sum()
    for i in res.index:
        sku = str(res.at[i, "sku"])
        if sku in fc_by_sku.index:
            res.at[i, "venta_forecast_total"] = float(fc_by_sku.loc[sku])
    return res


def _partial_month_run_rate_target(
    raw_ventas: pd.DataFrame | None,
    month_trim_meta: dict | None,
) -> tuple[float, pd.Period | None, dict]:
    """
    Mes en curso con datos parciales: ritmo diario = parcial ÷ días transcurridos;
    proyección del mes = parcial + ritmo × días restantes (no ETS).
    """
    empty: dict = {"applied": False}
    if not month_trim_meta or not month_trim_meta.get("excluded"):
        return float("nan"), None, empty
    if raw_ventas is None or raw_ventas.empty:
        return float("nan"), None, empty

    inc = month_trim_meta.get("incomplete_period")
    max_f = month_trim_meta.get("max_fecha")
    if inc is None or max_f is None:
        return float("nan"), None, empty

    inc_p = inc if isinstance(inc, pd.Period) else pd.Period(str(inc), freq="M")
    max_f = pd.Timestamp(max_f).normalize()
    if max_f.to_period("M") != inc_p:
        return float("nan"), None, empty

    raw_m, _ = _monthly_totals_from_raw_sheet(raw_ventas)
    if inc_p not in raw_m.index:
        return float("nan"), None, empty

    partial = float(raw_m.loc[inc_p])
    days_elapsed = int(max_f.day)
    days_in_month = int(inc_p.days_in_month)
    if days_elapsed < 1 or days_elapsed >= days_in_month:
        return float("nan"), None, empty

    days_remaining = days_in_month - days_elapsed
    daily_rate = partial / days_elapsed
    target = partial + daily_rate * days_remaining

    return (
        float(target),
        inc_p,
        {
            "applied": True,
            "method": "ritmo_diario",
            "period": str(inc_p),
            "mes_label": _mes_label_mmm_yyyy(inc_p.to_timestamp(how="start")),
            "partial_clp": partial,
            "days_elapsed": days_elapsed,
            "days_remaining": days_remaining,
            "days_in_month": days_in_month,
            "daily_rate_clp": daily_rate,
            "forecast_month_total": float(target),
        },
    )


def _apply_partial_month_run_rate_to_forecast(
    det_v: pd.DataFrame | None,
    res_v: pd.DataFrame | None,
    raw_ventas: pd.DataFrame | None,
    month_trim_meta: dict | None,
    *,
    ventas_ext: pd.DataFrame | None = None,
    freq_code: str = "M",
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict]:
    """
    Sustituye el 1.er mes proyectado (mes en curso parcial) por ritmo diario real.
    Los meses siguientes del horizonte conservan el ETS por SKU.
    """
    empty_meta: dict = {"applied": False}
    if det_v is None or det_v.empty or not str(freq_code).upper().startswith("M"):
        return det_v, res_v, empty_meta

    target, period, run_meta = _partial_month_run_rate_target(raw_ventas, month_trim_meta)
    if period is None or not pd.notna(target):
        return det_v, res_v, empty_meta

    fc_by_m = _forecast_monthly_after_last_complete(
        det_v,
        ventas_ext=ventas_ext,
        raw_ventas=raw_ventas,
        month_trim_meta=month_trim_meta,
    )
    if fc_by_m.empty or period not in fc_by_m.index:
        return det_v, res_v, {**run_meta, "applied": False, "reason": "mes_parcial_fuera_de_fc"}

    ets_before = float(fc_by_m.loc[period])
    det_new = _scale_det_forecast_calendar_month(det_v, period, target)
    res_new = _recompute_res_v_forecast_totals_from_det(det_new, res_v)

    meta = {
        **run_meta,
        "applied": True,
        "forecast_ets_before": ets_before,
        "forecast_after": float(target),
    }
    return det_new, res_new, meta


def _build_monthly_sales_clp_chart(
    det_v: pd.DataFrame,
    ventas_ext: pd.DataFrame | None = None,
    raw_ventas: pd.DataFrame | None = None,
    month_trim_meta: dict | None = None,
) -> tuple[alt.Chart | None, pd.DataFrame]:
    """
    Barras mensuales agregadas (todos los SKUs): últimos 6 meses históricos + proyección futura.

    Histórico (azul): suma mensual desde la hoja Excel cruda (Total Línea), últimos 6 meses.
    Proyección (rojo): suma mensual desde det_v tipo forecast, solo meses **después** del
    último mes completo usado en entrenamiento (mes parcial excluido si aplica).
    """
    empty_monthly = pd.DataFrame(columns=["periodo_m", "monto", "tipo_lc", "mes", "mes_fmt"])

    hist_by_m = _hist_monthly_for_sales_chart(
        det_v, ventas_ext=ventas_ext, raw_ventas=raw_ventas, month_trim_meta=month_trim_meta
    )
    fc_only = _forecast_monthly_after_last_complete(
        det_v, ventas_ext=ventas_ext, raw_ventas=raw_ventas, month_trim_meta=month_trim_meta
    )

    if hist_by_m.empty and fc_only.empty:
        return None, empty_monthly

    hist_last6 = hist_by_m.tail(6)

    parts: list[pd.DataFrame] = []
    if not hist_last6.empty:
        parts.append(
            pd.DataFrame(
                {
                    "periodo_m": hist_last6.index,
                    "monto": hist_last6.values,
                    "tipo_lc": "hist",
                }
            )
        )
    if not fc_only.empty:
        parts.append(
            pd.DataFrame(
                {
                    "periodo_m": fc_only.index,
                    "monto": fc_only.values,
                    "tipo_lc": "forecast",
                }
            )
        )
    monthly = pd.concat(parts, ignore_index=True)
    if monthly.empty:
        return None, empty_monthly

    monthly["mes"] = monthly["periodo_m"].apply(
        lambda p: pd.Period(p, freq="M").to_timestamp(how="start")
        if not isinstance(p, pd.Period)
        else p.to_timestamp(how="start")
    )
    monthly = monthly.sort_values("mes", ascending=True).reset_index(drop=True)
    monthly["mes_fmt"] = monthly["mes"].map(_mes_label_mmm_yyyy)
    mes_sort_dt = monthly["mes"].tolist()

    bars = (
        alt.Chart(monthly)
        .mark_bar(width=18)
        .encode(
            x=alt.X(
                "mes:T",
                title="Mes",
                sort=mes_sort_dt,
                axis=alt.Axis(
                    format="%b-%Y",
                    labelAngle=0,
                    titlePadding=8,
                ),
            ),
            y=alt.Y(
                "monto:Q",
                title="CLP",
                scale=alt.Scale(zero=True),
                axis=alt.Axis(
                    labelExpr="'$' + replace(format(datum.value, ',.0f'), ',', '.')",
                    titlePadding=8,
                ),
            ),
            color=alt.Color(
                "tipo_lc:N",
                title="",
                scale=alt.Scale(
                    domain=["hist", "forecast"],
                    range=["#1d4ed8", "#dc2626"],
                ),
                legend=alt.Legend(
                    title=None,
                    orient="top",
                    labelExpr="datum.value == 'hist' ? 'Histórico' : 'Proyección (forecast)'",
                ),
            ),
            tooltip=[
                alt.Tooltip("mes_fmt:N", title="Mes"),
                alt.Tooltip(
                    "tipo_lc:N",
                    title="Serie",
                ),
                alt.Tooltip(
                    "monto:Q",
                    title="Monto (CLP)",
                    format=",.0f",
                ),
            ],
        )
        .properties(title="Proyección de ventas mensual (CLP)", height=440)
    )

    hist_rows = monthly[monthly["tipo_lc"] == "hist"]
    fc_rows = monthly[monthly["tipo_lc"] == "forecast"]
    if hist_rows.empty or fc_rows.empty:
        return bars, monthly

    last_hist = pd.Timestamp(hist_rows["mes"].max())
    first_fc = pd.Timestamp(fc_rows["mes"].min())
    rule_mes = last_hist + (first_fc - last_hist) / 2
    ymax = float(monthly["monto"].max()) * 1.05 if monthly["monto"].notna().any() else 0.0

    rule = (
        alt.Chart(pd.DataFrame({"mes": [rule_mes]}))
        .mark_rule(color="#475569", strokeWidth=2, strokeDash=[6, 4], opacity=0.85)
        .encode(x=alt.X("mes:T", sort=mes_sort_dt, axis=None))
    )
    ann = (
        alt.Chart(pd.DataFrame({"mes": [rule_mes], "monto": [ymax]}))
        .mark_text(
            text="Inicio proyección",
            align="center",
            dy=-8,
            fontSize=11,
            color="#475569",
        )
        .encode(
            x=alt.X("mes:T", sort=mes_sort_dt, axis=None),
            y=alt.Y("monto:Q", axis=None),
        )
    )
    return bars + rule + ann, monthly


def _render_reporteria_tab(ventas: pd.DataFrame, stock: pd.DataFrame) -> None:
    """Reportería equivalente a app_reporteria.py usando ventas/stock ya cargados."""
    st.caption(
        "Mismos datos de ventas y stock que usaste para compras (sin Google Sheets ni API)."
    )
    if ventas.empty:
        st.warning("No hay ventas para reportería.")
        return
    colf1, colf2, colf3 = st.columns(3)
    dias = colf1.select_slider("Rango de días para el análisis", options=[7, 30, 60, 90], value=30)
    sku_filter = colf2.text_input("Filtrar por SKU (opcional)", "").strip().upper()
    fecha_corte = _fecha_corte_desde_ventas(ventas)
    desde = fecha_corte - timedelta(days=dias)
    colf3.write(f"Hasta: **{fecha_corte.date()}** (última fecha con venta en el archivo)")

    ventas_rango = ventas[(ventas["fecha"] >= desde) & (ventas["fecha"] <= fecha_corte)].copy()
    stock_r = stock.copy()
    if sku_filter:
        ventas_rango = ventas_rango[ventas_rango["sku"] == sku_filter]
        stock_r = stock_r[stock_r["sku"] == sku_filter]

    st.subheader(f"🏆 Top 10 más vendidos (últimos {dias} días)")
    top10 = (
        ventas_rango.groupby("sku", as_index=False)["qty"]
        .sum()
        .sort_values("qty", ascending=False)
        .head(10)
    )
    if top10.empty:
        st.info("No hay ventas en el rango de días seleccionado para el ranking.")
    else:
        _show_reporteria_table(top10)
        bar_top10 = (
            alt.Chart(top10)
            .mark_bar()
            .encode(
                x=alt.X("qty:Q", title="Unidades vendidas"),
                y=alt.Y("sku:N", sort="-x", title="SKU"),
                tooltip=["sku", "qty"],
            )
            .properties(height=300)
        )
        st.altair_chart(bar_top10, use_container_width=True)

    st.subheader("📈 Evolución de ventas (mensual, últimos 12 meses)")
    ventas_12m = ventas[
        (ventas["fecha"] <= fecha_corte) & (ventas["fecha"] >= fecha_corte - timedelta(days=365))
    ].copy()
    ventas_12m["mes"] = ventas_12m["fecha"].dt.to_period("M").astype(str)
    if sku_filter:
        ventas_12m = ventas_12m[ventas_12m["sku"] == sku_filter]
    ventas_mensual = ventas_12m.groupby("mes", as_index=False)["qty"].sum().sort_values("mes")
    if not ventas_mensual.empty:
        line_mes = (
            alt.Chart(ventas_mensual)
            .mark_line(point=True)
            .encode(
                x=alt.X("mes:N", title="Mes"),
                y=alt.Y("qty:Q", title="Unidades"),
                tooltip=["mes", "qty"],
            )
            .properties(height=280)
        )
        st.altair_chart(line_mes, use_container_width=True)
    else:
        st.info("No hay ventas en los últimos 12 meses para ese filtro.")

    st.subheader("📊 Productos en alza / en baja ↔")
    ini_mes_actual = fecha_corte.replace(day=1)
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
    alzabaja["delta"] = alzabaja["qty_cur"] - alzabaja["qty_prev"]
    col_1, col_2 = st.columns(2)
    with col_1:
        st.markdown("✅ **En alza**")
        en_alza = alzabaja[alzabaja["delta"] > 0].sort_values("delta", ascending=False)
        if en_alza.empty:
            st.info("No hay productos en alza para el período.")
        else:
            _show_reporteria_table(en_alza.head(20))
    with col_2:
        st.markdown("📉 **En baja**")
        en_baja = alzabaja[alzabaja["delta"] < 0].sort_values("delta", ascending=True)
        if en_baja.empty:
            st.info("No hay productos en baja para el período.")
        else:
            _show_reporteria_table(en_baja.head(20))

    st.subheader("📦 Productos sobre-stockeados")
    ultimos_60 = fecha_corte - timedelta(days=60)
    ventas_60 = ventas[(ventas["fecha"] >= ultimos_60) & (ventas["fecha"] <= fecha_corte)]
    consumo_60 = ventas_60.groupby("sku")["qty"].sum().rename("consumo_60d").reset_index()
    over = stock_r.merge(consumo_60, on="sku", how="left").fillna({"consumo_60d": 0})
    if over.empty:
        st.info("No hay datos de stock para este filtro.")
        return
    over["consumo_dia"] = over["consumo_60d"] / 60.0
    over["dias_cobertura"] = 0.0
    mask_sin_consumo = (over["consumo_60d"] == 0) & (over["stock"] > 0)
    if mask_sin_consumo.any():
        over.loc[mask_sin_consumo, "dias_cobertura"] = 9999
    mask_con_consumo = over["consumo_60d"] > 0
    if mask_con_consumo.any():
        over.loc[mask_con_consumo, "dias_cobertura"] = (
            over.loc[mask_con_consumo, "stock"] / over.loc[mask_con_consumo, "consumo_dia"]
        )
    UMBRAL_SOBRE = 20
    UMBRAL_BAJO = 5
    overstock = over[over["dias_cobertura"] >= UMBRAL_SOBRE].sort_values("dias_cobertura", ascending=False)
    if overstock.empty:
        st.info("No se detectaron productos sobre-stockeados con los criterios actuales.")
        top_cobertura = over.sort_values("dias_cobertura", ascending=False).head(20)
        st.write("Top por cobertura (referencia):")
        _show_reporteria_table(top_cobertura[["sku", "stock", "consumo_60d", "dias_cobertura"]])
    else:
        _show_reporteria_table(
            overstock[["sku", "stock", "consumo_60d", "dias_cobertura"]].head(50)
        )
        chart_over = (
            alt.Chart(overstock.head(20))
            .mark_bar()
            .encode(
                x=alt.X("dias_cobertura:Q", title="Días de cobertura"),
                y=alt.Y("sku:N", sort="-x", title="SKU"),
                tooltip=["sku", "stock", "dias_cobertura"],
            )
            .properties(height=400)
        )
        st.altair_chart(chart_over, use_container_width=True)

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


def _render_ventas_predictor_tab() -> None:
    """
    Equivalente funcional al módulo Ventas de app_predictor.py: parámetros, estacionalidad
    manual, forecast en $ (Total Línea → venta_neta), sin API Kame ni batch.
    """
    st.subheader("📈 Predictor de ventas")
    st.caption("Fuente de datos: **desde upload** — Variable proyectada: monto ($) por **Total Línea** (agregación mensual/semanal por SKU, mismo core que la app principal).")

    if forecast_sales is None:
        st.error(
            "No se encontró `predictor_sales_core.forecast_sales`. "
            "Verifica que el archivo `predictor_sales_core.py` exista en la misma carpeta."
        )
        return

    run = st.session_state.get(SESSION_RUN_KEY)
    if not run:
        return

    raw = run.get("ventas_raw_full")
    if raw is None or (isinstance(raw, pd.DataFrame) and raw.empty):
        raw = run.get("ventas_raw")
    if raw is None or (isinstance(raw, pd.DataFrame) and raw.empty):
        st.warning(
            "No hay hoja cruda de ventas en memoria para este run. "
            "Vuelve a ejecutar **Cargar Excel y ejecutar predicción** con un archivo de ventas válido."
        )
        return

    col1, col2, col3 = st.columns(3)
    freq_label = col1.selectbox(
        "Frecuencia",
        ["Mensual (M)", "Semanal (W)"],
        index=0,
        key="excel_v2_ventas_freq",
    )
    horizon_v = col2.slider(
        "Horizonte (períodos)",
        2,
        24,
        6,
        1,
        key="excel_v2_ventas_horizon",
    )
    modo_v = col3.selectbox("Modo", ["Global", "Por SKU"], index=0, key="excel_v2_ventas_modo")
    sku_q_v = (
        st.text_input("SKU (opcional)", key="excel_v2_ventas_sku")
        if modo_v == "Por SKU"
        else None
    )

    if "excel_v2_seasonal_weights" not in st.session_state:
        st.session_state["excel_v2_seasonal_weights"] = None
    if "excel_v2_usar_estacionalidad" not in st.session_state:
        st.session_state["excel_v2_usar_estacionalidad"] = False
    if "excel_v2_low_sel" not in st.session_state:
        st.session_state["excel_v2_low_sel"] = [
            "12 - Diciembre",
            "1 - Enero",
            "2 - Febrero",
        ]
    if "excel_v2_high_sel" not in st.session_state:
        st.session_state["excel_v2_high_sel"] = ["3 - Marzo"]
    if "excel_v2_low_factor" not in st.session_state:
        st.session_state["excel_v2_low_factor"] = 0.7
    if "excel_v2_high_factor" not in st.session_state:
        st.session_state["excel_v2_high_factor"] = 1.1

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

    with st.expander("Estacionalidad manual por mes (opcional)", expanded=False):
        usar_estacionalidad = st.checkbox(
            "Activar ajuste manual de meses altos / bajos",
            help="Si lo desmarcas, el modelo usa solo la serie histórica sin ajustes manuales.",
            key="excel_v2_usar_estacionalidad",
        )

        low_sel = st.multiselect(
            "Meses con baja estacionalidad (ventas más bajas que el promedio)",
            options=list(meses_labels.values()),
            key="excel_v2_low_sel",
        )
        high_sel = st.multiselect(
            "Meses con alta estacionalidad (ventas más altas que el promedio)",
            options=list(meses_labels.values()),
            key="excel_v2_high_sel",
        )

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
            step=0.05,
            help="Ej: 0.7 significa que esos meses se proyectan al 70% del nivel base.",
            key="excel_v2_low_factor",
        )
        high_factor = c_high.slider(
            "Factor para meses altos",
            min_value=1.0,
            max_value=1.7,
            step=0.05,
            help="Ej: 1.1 significa que esos meses se proyectan al 110% del nivel base.",
            key="excel_v2_high_factor",
        )

        if usar_estacionalidad:
            seasonal_weights = {}
            for m in range(1, 13):
                w = 1.0
                if m in low_months:
                    w = low_factor
                if m in high_months:
                    w = high_factor
                seasonal_weights[m] = float(w)
            st.session_state["excel_v2_seasonal_weights"] = seasonal_weights
            st.markdown("---")
            st.markdown("**📊 Factores de estacionalidad configurados:**")
            factors_df = pd.DataFrame(
                [
                    {
                        "Mes": meses_labels[m],
                        "Factor": f"{w:.2f}x",
                        "Tipo": "Bajo" if w < 1.0 else "Alto" if w > 1.0 else "Normal",
                    }
                    for m, w in sorted(seasonal_weights.items())
                ]
            )
            st.dataframe(factors_df, use_container_width=True, hide_index=True)
        else:
            st.session_state["excel_v2_seasonal_weights"] = None

    seasonal_weights = st.session_state.get("excel_v2_seasonal_weights")
    usar_estacionalidad = st.session_state.get("excel_v2_usar_estacionalidad", False)

    if st.button(
        "Calcular proyección de ventas",
        type="primary",
        use_container_width=True,
        key="excel_v2_btn_calcular_ventas",
    ):
        run_now = st.session_state.get(SESSION_RUN_KEY) or {}
        raw_now = run_now.get("ventas_raw_full")
        if raw_now is None or (isinstance(raw_now, pd.DataFrame) and raw_now.empty):
            raw_now = run_now.get("ventas_raw")
        if raw_now is None or (isinstance(raw_now, pd.DataFrame) and raw_now.empty):
            st.session_state[SESSION_VENTAS_FORECAST_KEY] = {
                "error": "No hay datos crudos de ventas en esta sesión.",
                "det_v": None,
                "res_v": None,
                "info_flags": [],
            }
            st.rerun()

        freq_code = "M" if str(freq_label).startswith("Mensual") else "W"
        estacionalidad_aplicada = False
        info_flags: list[str] = []

        if usar_estacionalidad and seasonal_weights is not None and freq_code == "W":
            info_flags.append("season_w_ignored")
        elif usar_estacionalidad and seasonal_weights is not None and freq_code == "M":
            estacionalidad_aplicada = True

        with st.spinner("Leyendo ventas y calculando proyección…"):
            monto_col_used = _resolve_monto_linea_col(raw_now)
            ventas_ext = _normalize_ventas_for_sales_excel(raw_now.copy())
            if modo_v == "Por SKU" and sku_q_v:
                fsku = str(sku_q_v).strip().upper()
                ventas_ext = ventas_ext[ventas_ext["sku"] == fsku]

            ventas_ext = _filter_invalid_skus_ventas_sales(ventas_ext)
            ventas_ext["fecha"] = _coerce_fecha_cl(ventas_ext["fecha"])

            if ventas_ext is None or ventas_ext.empty:
                st.session_state[SESSION_VENTAS_FORECAST_KEY] = {
                    "error": "No hay ventas para los filtros dados (revisa fecha, SKU y cantidad en el Excel).",
                    "det_v": None,
                    "res_v": None,
                    "ventas_ext": pd.DataFrame(),
                    "freq_code": freq_code,
                    "estacionalidad_aplicada": False,
                    "info_flags": [],
                }
                st.rerun()

            ventas_ext_train, month_trim_meta = _drop_incomplete_last_month_rows(ventas_ext)
            if month_trim_meta.get("excluded"):
                info_flags.append("month_incomplete_excluded")

            if ventas_ext_train is None or ventas_ext_train.empty:
                st.session_state[SESSION_VENTAS_FORECAST_KEY] = {
                    "error": (
                        "No quedan meses completos para entrenar el modelo "
                        "(el archivo solo tiene datos del mes en curso incompleto)."
                    ),
                    "det_v": None,
                    "res_v": None,
                    "ventas_ext": pd.DataFrame(),
                    "month_trim_meta": month_trim_meta,
                    "freq_code": freq_code,
                    "estacionalidad_aplicada": False,
                    "info_flags": info_flags,
                }
                st.rerun()

            extra_kwargs = {}
            if usar_estacionalidad and seasonal_weights is not None and freq_code == "M":
                extra_kwargs["seasonality_factors"] = seasonal_weights

            det_v = None
            res_v = None
            error_ocurrido = None
            took_season_fallback = False
            try:
                det_v, res_v = forecast_sales(
                    ventas_ext_train,
                    freq=freq_code,
                    horizon=int(horizon_v),
                    **extra_kwargs,
                )
            except TypeError:
                try:
                    det_v, res_v = forecast_sales(
                        ventas_ext_train,
                        freq=freq_code,
                        horizon=int(horizon_v),
                    )
                    took_season_fallback = True
                except Exception as e2:
                    error_ocurrido = f"Error al ejecutar forecast sin estacionalidad: {e2}"
            except ValueError as e:
                error_ocurrido = f"Error de validación: {e}"
            except Exception as e:
                error_ocurrido = f"Error inesperado al calcular proyección: {e}"

            if took_season_fallback and usar_estacionalidad and seasonal_weights is not None:
                info_flags.append("core_no_seasonality")

            partial_month_meta: dict = {"applied": False}
            if error_ocurrido is None and det_v is not None and res_v is not None:
                det_v, res_v = _postprocess_forecast_cuts_and_alerts(
                    det_v, res_v, int(horizon_v)
                )
                det_v, res_v, partial_month_meta = _apply_partial_month_run_rate_to_forecast(
                    det_v,
                    res_v,
                    raw_now,
                    month_trim_meta,
                    ventas_ext=ventas_ext_train,
                    freq_code=freq_code,
                )
                if partial_month_meta.get("applied"):
                    info_flags.append("partial_month_run_rate")

            if error_ocurrido:
                st.session_state[SESSION_VENTAS_FORECAST_KEY] = {
                    "error": error_ocurrido,
                    "det_v": None,
                    "res_v": None,
                    "ventas_ext": ventas_ext,
                    "freq_code": freq_code,
                    "estacionalidad_aplicada": False,
                    "info_flags": info_flags,
                }
                st.rerun()

            st.session_state[SESSION_VENTAS_FORECAST_KEY] = {
                "error": "",
                "det_v": det_v,
                "res_v": res_v,
                "ventas_ext": ventas_ext_train,
                "month_trim_meta": month_trim_meta,
                "monto_col_used": monto_col_used,
                "modo": modo_v,
                "sku_filter": str(sku_q_v).strip().upper() if modo_v == "Por SKU" and sku_q_v else "",
                "freq_code": freq_code,
                "estacionalidad_aplicada": estacionalidad_aplicada,
                "info_flags": info_flags,
                "partial_month_run_rate_meta": partial_month_meta,
            }
        st.rerun()

    out = st.session_state.get(SESSION_VENTAS_FORECAST_KEY)
    if not out:
        st.info("Configura frecuencia, horizonte y opciones, luego pulsa **Calcular proyección de ventas**.")
        return

    if out.get("error"):
        st.error(f"❌ {out['error']}")
        return

    for flag in out.get("info_flags") or []:
        if flag == "season_w_ignored":
            st.info(
                "ℹ️ **Estacionalidad no aplicada**: Los factores de estacionalidad solo se aplican "
                "para frecuencia **Mensual (M)**. Con frecuencia Semanal (W), los factores no se aplicaron."
            )
        elif flag == "core_no_seasonality":
            st.info(
                "ℹ️ La versión del core no soporta factores de estacionalidad. "
                "Se ejecutó sin ajustes estacionales."
            )
        elif flag == "partial_month_run_rate":
            pm = out.get("partial_month_run_rate_meta") or {}
            mes_lbl = pm.get("mes_label", "mes en curso")
            parcial = pm.get("partial_clp")
            ritmo = pm.get("daily_rate_clp")
            rest = pm.get("days_remaining")
            total = pm.get("forecast_month_total")
            ets_antes = pm.get("forecast_ets_before")
            if total is not None and parcial is not None:
                st.info(
                    f"ℹ️ **{mes_lbl} (mes en curso):** proyección por **ritmo diario real** = "
                    f"**${total:,.0f}** "
                    f"(${parcial:,.0f} en {pm.get('days_elapsed')} días + "
                    f"${ritmo:,.0f}/día × {rest} días restantes). "
                    f"Junio en adelante sigue con ETS"
                    + (
                        f" (ETS había proyectado ${ets_antes:,.0f} para {mes_lbl})."
                        if ets_antes is not None
                        else "."
                    )
                )
        elif flag == "month_incomplete_excluded":
            mt = out.get("month_trim_meta") or {}
            inc = mt.get("incomplete_period")
            last_c = mt.get("last_complete_period")
            fc_from = mt.get("forecast_from_period")
            max_f = mt.get("max_fecha")
            if inc is not None:
                inc_lbl = _mes_label_mmm_yyyy(inc.to_timestamp(how="start"))
                last_lbl = (
                    _mes_label_mmm_yyyy(last_c.to_timestamp(how="start"))
                    if last_c is not None
                    else "—"
                )
                fc_lbl = (
                    _mes_label_mmm_yyyy(fc_from.to_timestamp(how="start"))
                    if fc_from is not None
                    else "—"
                )
                st.info(
                    f"ℹ️ **Mes incompleto detectado:** {inc_lbl} "
                    f"(última venta en archivo: {max_f.date() if max_f is not None else '—'}). "
                    f"ETS entrena solo hasta **{last_lbl}**; **{inc_lbl}** se proyecta por ritmo diario "
                    f"(ventas parciales ÷ días transcurridos × días restantes). "
                    f"Desde **{fc_lbl}** el horizonte usa ETS."
                )

    det_v = out.get("det_v")
    res_v = out.get("res_v")
    if res_v is not None and not res_v.empty and "meses_winsorizados" in res_v.columns:
        win_rows = res_v.loc[pd.to_numeric(res_v["meses_winsorizados"], errors="coerce").fillna(0) > 0]
        if not win_rows.empty:
            n_skus = len(win_rows)
            n_meses = int(pd.to_numeric(win_rows["meses_winsorizados"], errors="coerce").sum())
            st.warning(
                f"⚠️ **Winsorización ETS** (solo entrenamiento, histórico en pantalla = Excel): "
                f"**{n_skus}** SKU(s) con **{n_meses}** mes(es) con pico recortado (tope Q3 + 2,5×IQR). "
                "Croston y series cortas no se modifican."
            )
    ventas_ext = out.get("ventas_ext")
    freq_code = out.get("freq_code", "M")
    estacionalidad_aplicada = bool(out.get("estacionalidad_aplicada"))

    if ventas_ext is not None and isinstance(ventas_ext, pd.DataFrame) and not ventas_ext.empty:
        total_hist = float(ventas_ext["venta_neta"].sum())
        sku_count = int(ventas_ext["sku"].nunique())
    else:
        total_hist = 0.0
        sku_count = 0

    raw_df = run.get("ventas_raw_full")
    if raw_df is None or (isinstance(raw_df, pd.DataFrame) and raw_df.empty):
        raw_df = run.get("ventas_raw")
    with st.expander("🔍 Información de depuración (lectura de datos)", expanded=False):
        if raw_df is not None and not raw_df.empty:
            st.write("**Columnas del Excel de ventas (run actual):**")
            st.write(list(raw_df.columns))
            st.write(f"**Filas en hoja:** {len(raw_df)}")
            venta_col = _resolve_monto_linea_col(raw_df)
            folded = _columns_folded(raw_df)
            alt_vn = folded.get("venta neta") or folded.get("venta_neta")
            if venta_col:
                st.success(
                    f"✅ Columna de monto usada en proyección: **{venta_col}** "
                    "(prioridad Total Línea sobre venta_neta)."
                )
                total_raw = pd.to_numeric(raw_df[venta_col], errors="coerce").fillna(0.0).sum()
                st.write(f"**Total en esa columna (hoja):** ${total_raw:,.0f}")
                if alt_vn and alt_vn != venta_col:
                    total_vn = pd.to_numeric(raw_df[alt_vn], errors="coerce").fillna(0.0).sum()
                    st.caption(
                        f"Columna alternativa **{alt_vn}** (no usada): ${total_vn:,.0f} en toda la hoja. "
                        "Si difiere mucho, el gráfico histórico sigue Total Línea."
                    )
            else:
                st.warning(
                    "⚠️ No se encontró columna de monto (Total Línea / venta_neta). "
                    "`venta_neta` quedará en 0 si no hay coincidencia."
                )
        if ventas_ext is not None and isinstance(ventas_ext, pd.DataFrame):
            st.write(f"**Filas después de normalización:** {len(ventas_ext)}")
            if not ventas_ext.empty and "venta_neta" in ventas_ext.columns:
                st.write(f"**Total venta_neta (monto $) tras normalizar:** ${float(ventas_ext['venta_neta'].sum()):,.0f}")
                st.dataframe(
                    ventas_ext[["fecha", "sku", "venta_neta"]].head(),
                    use_container_width=True,
                )

    if estacionalidad_aplicada:
        st.success(
            "✅ **Estacionalidad aplicada**: Los factores de estacionalidad se han aplicado correctamente a las predicciones."
        )

    raw_for_fc = run.get("ventas_raw_full")
    if raw_for_fc is None or (isinstance(raw_for_fc, pd.DataFrame) and raw_for_fc.empty):
        raw_for_fc = run.get("ventas_raw")
    raw_fc = raw_for_fc
    sku_fc_metric = out.get("sku_filter") or ""
    if out.get("modo") == "Por SKU" and sku_fc_metric and raw_fc is not None:
        raw_fc = _filter_raw_ventas_by_sku(raw_fc, sku_fc_metric)
    month_trim_meta_out = out.get("month_trim_meta")
    total_forecast = _forecast_total_global_calendar_window(
        det_v,
        ventas_ext=ventas_ext,
        raw_ventas=raw_fc,
        month_trim_meta=month_trim_meta_out,
    )

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            _html_kpi_card("📦 SKUs considerados", f"{sku_count:,}", "#1f77b4 0%, #2a9d8f 100%"),
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            _html_kpi_card(
                "📊 Venta histórica total",
                f"${total_hist:,.0f}",
                "#28a745 0%, #20c997 100%",
            ),
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            _html_kpi_card(
                "📈 Venta proyectada (horizonte)",
                f"${total_forecast:,.0f}",
                "#2d5016 0%, #4a7c2a 100%",
            ),
            unsafe_allow_html=True,
        )
        st.caption(
            "Suma de las barras rojas del gráfico (meses calendario tras el último mes completo). "
            "Exportación Excel al final de la página, debajo del gráfico."
        )

    run_rate_review = out.get("partial_month_run_rate_meta") or {}
    if not run_rate_review.get("forecast_month_total") and raw_fc is not None:
        _, _, run_rate_review = _partial_month_run_rate_target(raw_fc, month_trim_meta_out)
        if det_v is not None and not det_v.empty:
            fc_by_m = _forecast_monthly_after_last_complete(
                det_v,
                ventas_ext=ventas_ext,
                raw_ventas=raw_fc,
                month_trim_meta=month_trim_meta_out,
            )
            if not fc_by_m.empty:
                run_rate_review["forecast_ets_before"] = float(fc_by_m.iloc[0])
    if run_rate_review.get("period") or run_rate_review.get("mes_label"):
        with st.expander(
            f"📋 Mes en curso: {run_rate_review.get('mes_label', 'proyección por ritmo')}",
            expanded=bool(run_rate_review.get("applied")),
        ):
            parcial = run_rate_review.get("partial_clp")
            ritmo = run_rate_review.get("daily_rate_clp")
            de = run_rate_review.get("days_elapsed")
            rest = run_rate_review.get("days_remaining")
            dim = run_rate_review.get("days_in_month")
            ets_antes = run_rate_review.get("forecast_ets_before")
            total = run_rate_review.get("forecast_month_total") or run_rate_review.get(
                "forecast_after"
            )
            restante_clp = (
                float(ritmo) * int(rest)
                if pd.notna(ritmo) and rest is not None
                else float("nan")
            )
            rows_rev = [
                {
                    "Concepto": "Ventas parciales (Excel, mes en curso)",
                    "Valor CLP": parcial,
                    "Detalle": f"{de} días transcurridos de {dim}",
                },
                {
                    "Concepto": "Ritmo diario (parcial ÷ días)",
                    "Valor CLP": ritmo,
                    "Detalle": "Promedio $/día observado",
                },
                {
                    "Concepto": "Proyección días restantes",
                    "Valor CLP": restante_clp,
                    "Detalle": f"{rest} días × ritmo diario",
                },
                {
                    "Concepto": "Total mes (parcial + restante)",
                    "Valor CLP": total,
                    "Detalle": "Usado en gráfico y Excel (no ETS)",
                },
                {
                    "Concepto": "ETS (solo referencia, no usado en este mes)",
                    "Valor CLP": ets_antes,
                    "Detalle": "Junio+ sí usa ETS",
                },
            ]
            st.dataframe(pd.DataFrame(rows_rev), use_container_width=True, hide_index=True)

    if res_v is not None and not res_v.empty:
        st.subheader("Resumen de ventas proyectadas por SKU")
        if "alerta_inflacion" in res_v.columns:
            n_alert = (res_v["alerta_inflacion"].astype(str).str.strip() != "").sum()
            if n_alert > 0:
                st.warning(
                    f"⚠️ **{int(n_alert)}** SKU(s) con advertencia en **alerta_inflacion** "
                    "(forecast muy alto vs. promedio histórico × horizonte)."
                )
        st.dataframe(res_v, use_container_width=True, hide_index=True)

        has_metrics = any(
            col in res_v.columns for col in ["mape", "rmse", "mae", "demand_class", "adi", "cv2"]
        )
        if has_metrics:
            st.markdown("---")
            st.subheader("📊 Métricas de calidad del forecast")
            metrics_cols = ["sku", "modelo", "demand_class", "mape", "rmse", "mae", "adi", "cv2"]
            available_metrics = [col for col in metrics_cols if col in res_v.columns]
            if available_metrics:
                if "sku" not in available_metrics:
                    available_metrics.insert(0, "sku")
                if "modelo" in res_v.columns and "modelo" not in available_metrics:
                    available_metrics.insert(1, "modelo")
                metrics_df = res_v[available_metrics].copy()
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    if det_v is not None and not det_v.empty:
        det_show = _sort_sales_det_for_display(det_v)
        st.caption(
            "Vista agregada por mes (todos los SKUs). "
            "**Azul:** últimos 6 **meses completos** (Total Línea; el mes en curso parcial no se muestra). "
            "**Rojo:** proyección desde el mes siguiente al último mes cerrado. "
            "Descargue el Excel debajo del gráfico para totales alineados a las barras rojas."
        )
        raw_for_dbg = raw_for_fc
        raw_chart = raw_fc
        month_trim_meta = month_trim_meta_out
        ch, monthly_plot = _build_monthly_sales_clp_chart(
            det_v,
            ventas_ext=ventas_ext,
            raw_ventas=raw_chart,
            month_trim_meta=month_trim_meta,
        )
        monto_col_used = out.get("monto_col_used")
        with st.expander("🔍 Debug: suma por mes (validación vs Excel)", expanded=False):
            if monto_col_used:
                st.caption(f"Columna de monto usada en normalización: **{monto_col_used}** (prioridad Total Línea).")
            dbg = _debug_monthly_validation_table(
                ventas_ext,
                det_v,
                monthly_plot,
                raw_ventas=raw_for_dbg,
                month_trim_meta=month_trim_meta,
            )
            prom6 = dbg.attrs.get("prom_hist_6", float("nan"))
            if pd.notna(prom6) and prom6 > 0:
                st.caption(
                    f"Promedio mensual histórico (últimos 6 meses, hoja cruda): **${prom6:,.0f}**. "
                    "`excel_ventas_ext_CLP` debe coincidir con `excel_hoja_CLP` (misma Total Línea; "
                    "filas sin SKU van a **SIN_SKU**). `fc_vs_prom_hist_6x` = forecast ÷ ese promedio "
                    "(no es factor de estacionalidad manual)."
                )
            if dbg.empty:
                st.info("Sin datos para la tabla de validación.")
            else:
                st.dataframe(dbg, use_container_width=True, hide_index=True)
                may26 = dbg[dbg["periodo"] == "2026-05"]
                if not may26.empty:
                    row = may26.iloc[0]
                    mt = month_trim_meta or {}
                    may_note = (
                        " (mes parcial: excluido del ETS y del gráfico azul)"
                        if mt.get("excluded")
                        and mt.get("incomplete_period") is not None
                        and str(mt["incomplete_period"]) == "2026-05"
                        else ""
                    )
                    st.caption(
                        f"**May-2026:** Hoja cruda = ${row['excel_hoja_CLP']:,.0f} · "
                        f"ventas_ext (entrenamiento) = ${row['excel_ventas_ext_CLP']:,.0f} · "
                        f"Gráfico = ${row['grafico_CLP']:,.0f} ({row['grafico_tipo']}){may_note}"
                    )
                nov26 = dbg[dbg["periodo"] == "2026-11"]
                if not nov26.empty and pd.notna(prom6):
                    r = nov26.iloc[0]
                    fc = r.get("det_forecast_CLP", float("nan"))
                    if pd.notna(fc):
                        ratio = fc / prom6
                        st.caption(
                            f"**Nov-2026 forecast:** ${fc:,.0f} "
                            f"({ratio:.2f}× el promedio de los últimos 6 meses en Excel, ${prom6:,.0f}). "
                            "Eso refleja el modelo ETS, no un descuento de estacionalidad "
                            "(noviembre usa factor 1,0 salvo que lo marques como mes bajo/alto)."
                        )
        if ch is not None:
            st.altair_chart(ch, use_container_width=True)
        else:
            st.info("No hay datos suficientes para armar el gráfico mensual.")
        st.markdown("---")
        _render_ventas_excel_download(
            det_v,
            res_v,
            ventas_ext=ventas_ext,
            raw_ventas=raw_fc,
            month_trim_meta=month_trim_meta_out,
            sku_count=sku_count,
            total_hist=total_hist,
            total_forecast=total_forecast,
            button_key="excel_v2_dl_ventas_proyeccion",
        )


st.set_page_config(
    page_title="Predictor de Compras (Excel)",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)
_inject_predictor_excel_theme()
_render_predictor_excel_header()
st.caption("Sus archivos se procesan solo en esta sesión; no se almacenan en el servidor.")

fuente = st.radio(
    "Fuente de datos",
    ["Subir archivos (.xlsx / .xls)", "Carpeta local (pruebas)"],
    horizontal=True,
)

up_ventas = up_stock = up_oc = None
excel_dir = str(DEFAULT_EXCEL_DIR)

if fuente.startswith("Subir"):
    up_ventas, up_stock, up_oc = _render_excel_upload_sidebar()
else:
    with st.sidebar:
        st.markdown("### 📁 Carpeta local (pruebas)")
        st.caption("Los Excel se leen desde la ruta indicada abajo.")
    excel_dir = st.text_input("Carpeta con los 3 Excel", value=str(DEFAULT_EXCEL_DIR))

st.markdown("### ⚙️ Parámetros de predicción")
col_a, col_b, col_c = st.columns(3)
freq = col_a.selectbox("Frecuencia", ["Mensual (M)", "Semanal (W)"], index=0)
horizon = col_b.slider("Horizonte (períodos)", 2, 24, 6, 1)
modo = col_c.selectbox("Modo", ["Global", "Por SKU"], index=0)
sku_q = st.text_input("SKU (opcional)") if modo == "Por SKU" else None

mostrar_previo = st.checkbox(
    "Mostrar vista previa de datos canónicos (ventas / stock / OC)",
    value=False,
    help="Muestra las primeras filas normalizadas tras ejecutar la predicción.",
)

if st.button("Cargar Excel y ejecutar predicción", type="primary", use_container_width=True):
    if SESSION_VENTAS_FORECAST_KEY in st.session_state:
        del st.session_state[SESSION_VENTAS_FORECAST_KEY]
    st.session_state[SESSION_SECTION_NAV_KEY] = _SECTION_LABELS[0]
    upload_run_log: tuple[str, bool] | None = None
    raw_ventas_df = pd.DataFrame()

    if fuente.startswith("Subir"):
        fallback_folder = DEFAULT_EXCEL_DIR
        source_log: list[str] = []

        for up, label in (
            (up_ventas, "ventas"),
            (up_stock, "stock / inventario"),
            (up_oc, "órdenes de compra"),
        ):
            err = _validate_upload_extension(up, label)
            if err:
                st.error(err)
                st.stop()

        # --- Ventas ---
        if up_ventas is not None:
            b_v, err_v = _read_upload_bytes(up_ventas)
            if err_v:
                st.error(err_v)
                st.stop()
            name_v = up_ventas.name
            try:
                raw_ventas_df = read_first_sheet_bytes(b_v, ("Reporte", "Report"))
                ventas = load_ventas_kame_df(raw_ventas_df, source_label=name_v)
            except Exception as e:
                st.error(f"No se pudo leer el Excel de ventas: {e}")
                st.stop()
            source_log.append(f"✅ Ventas: {name_v} (desde upload)")
        else:
            pv = pick_excel_in_folder(fallback_folder, "informeventas")
            if pv is None:
                st.error(
                    f"No subiste archivo de ventas y en **{fallback_folder}** no hay ningún "
                    "archivo cuyo nombre contenga «informeventas» (.xlsx / .xls)."
                )
                st.stop()
            try:
                raw_ventas_df = read_first_sheet_path(pv, ("Reporte", "Report"))
                ventas = load_ventas_kame_df(raw_ventas_df, source_label=pv.name)
            except Exception as e:
                st.error(f"No se pudo leer ventas desde carpeta: {e}")
                st.stop()
            source_log.append(
                f"⚠️ Ventas: {pv.name} — usando carpeta local (no se subió archivo)"
            )

        # --- Stock ---
        if up_stock is not None:
            b_s, err_s = _read_upload_bytes(up_stock)
            if err_s:
                st.error(err_s)
                st.stop()
            name_s = up_stock.name
            try:
                stock = load_stock_kame_bytes(b_s, source_label=name_s)
            except Exception as e:
                st.error(f"No se pudo leer el Excel de stock: {e}")
                st.stop()
            source_log.append(f"✅ Stock: {name_s} (desde upload)")
        else:
            ps = pick_excel_in_folder(fallback_folder, "inventariobodega")
            if ps is None:
                st.error(
                    f"No subiste archivo de stock y en **{fallback_folder}** no hay ningún "
                    "archivo cuyo nombre contenga «inventariobodega» (.xlsx / .xls)."
                )
                st.stop()
            try:
                stock = load_stock_kame(ps)
            except Exception as e:
                st.error(f"No se pudo leer stock desde carpeta: {e}")
                st.stop()
            source_log.append(
                f"⚠️ Stock: {ps.name} — usando carpeta local (no se subió archivo)"
            )

        # --- OC: preflight + inbound ---
        raw_oc: pd.DataFrame | None = None
        b_o: bytes | None = None
        name_o = ""
        po: Path | None = None
        if up_oc is not None:
            b_o, err_o = _read_upload_bytes(up_oc)
            if err_o:
                st.error(err_o)
                st.stop()
            name_o = up_oc.name
            try:
                raw_oc = read_first_sheet_bytes(b_o, ("Report", "Reporte"))
            except Exception as e:
                st.error(f"No se pudo leer la hoja de órdenes de compra: {e}")
                st.stop()
            source_log.append(f"✅ OC: {name_o} (desde upload)")
        else:
            po = pick_excel_in_folder(fallback_folder, "ordenescompra")
            if po is None:
                st.error(
                    f"No subiste archivo de OC y en **{fallback_folder}** no hay ningún "
                    "archivo cuyo nombre contenga «ordenescompra» (.xlsx / .xls)."
                )
                st.stop()
            try:
                raw_oc = read_first_sheet_path(po, ("Report", "Reporte"))
            except Exception as e:
                st.error(f"No se pudo leer OC desde carpeta: {e}")
                st.stop()
            source_log.append(
                f"⚠️ OC: {po.name} — usando carpeta local (no se subió archivo)"
            )

        assert raw_oc is not None
        pf = oc_recepcion_preflight(raw_oc)
        if pf["recepcion_col"] is None:
            st.error(
                "Órdenes de compra: no se encontró una columna **Recepción** "
                "(nombre reconocido como «Recepción»). "
                f"Columnas: {list(raw_oc.columns)}"
            )
            st.stop()
        if pf["n_pendiente"] == 0:
            st.warning(
                "Órdenes de compra: no hay filas con estado de recepción **Pendiente**. "
                f"Valores de ejemplo en «{pf['recepcion_col']}»: {pf['recepcion_values_sample']}. "
                "El inbound quedará vacío (la predicción sigue sin compras en tránsito)."
            )
        if pf["por_recibir_col"] is None:
            st.error(
                "Órdenes de compra: no se encontró la columna **Por recibir** "
                "(necesaria para la cantidad en tránsito)."
            )
            st.stop()

        try:
            if up_oc is not None and b_o is not None:
                inbound = load_inbound_oc_kame_bytes(b_o, source_label=name_o)
            elif po is not None:
                inbound = load_inbound_oc_kame(po)
            else:
                st.error("No se pudo resolver el archivo de órdenes de compra.")
                st.stop()
        except ValueError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(
                "No se pudieron normalizar las órdenes de compra. "
                "Si usas **.xls**, puede faltar **xlrd**. "
                f"Detalle: `{type(e).__name__}: {e}`"
            )
            st.stop()

        for msg in _validate_canonical_ventas(ventas):
            st.error(msg)
            st.stop()
        for msg in _validate_canonical_stock(stock):
            st.warning(msg)

        upload_run_log = (
            "Origen de los datos\n\n" + "\n".join(source_log),
            all(line.startswith("✅") for line in source_log),
        )

    else:
        folder = Path(excel_dir)
        if not folder.is_dir():
            st.error(f"No existe la carpeta: {folder}")
            st.stop()

        try:
            with st.spinner("Leyendo archivos desde carpeta…"):
                ventas, stock, inbound = load_pack_from_folder(folder)
                pv_v = pick_excel_in_folder(folder, "informeventas")
                if pv_v is not None:
                    raw_ventas_df = read_first_sheet_path(pv_v, ("Reporte", "Report"))
                else:
                    raw_ventas_df = pd.DataFrame()

            oc_path = None
            for p in sorted(folder.glob("*.xlsx")):
                if p.name.startswith("~$"):
                    continue
                if "ordenescompra" in p.name.lower():
                    oc_path = p
                    break
            if oc_path is not None:
                raw_oc = read_first_sheet_path(oc_path, ("Report", "Reporte"))
                pf = oc_recepcion_preflight(raw_oc)
                if pf["recepcion_col"] is None:
                    st.warning(
                        "OC: no se detectó columna Recepción en el archivo de órdenes; "
                        "revisa el export."
                    )
                elif pf["n_pendiente"] == 0:
                    st.info(
                        "Órdenes de compra: no hay filas **Pendiente** en recepción; "
                        "no se sumará inbound en tránsito."
                    )
        except Exception as e:
            st.error(f"No se pudieron leer los archivos de la carpeta: {e}")
            st.stop()

        for msg in _validate_canonical_ventas(ventas):
            st.error(msg)
            st.stop()
        for msg in _validate_canonical_stock(stock):
            st.warning(msg)

    config = default_config_from_union(ventas, stock)

    ventas_raw_full = raw_ventas_df.copy()

    if modo == "Por SKU" and sku_q:
        f = str(sku_q).strip().upper()
        ventas = ventas[ventas["sku"] == f]
        stock = stock[stock["sku"] == f]
        config = config[config["sku"] == f]
        inbound = inbound[inbound["sku"] == f]
        raw_ventas_df = _filter_raw_ventas_by_sku(raw_ventas_df, f)

    if ventas.empty:
        st.warning("No hay filas de ventas para los filtros dados.")
        st.stop()

    fcode = "M" if str(freq).upper().startswith("M") else "W"
    with st.spinner("Calculando predicción…"):
        try:
            det, res, prop = forecast_all(
                ventas,
                stock,
                config,
                freq=fcode,
                horizon_override=int(horizon),
                inbound=inbound if not inbound.empty else None,
            )
        except Exception as e:
            st.error(f"Error al calcular la predicción: {type(e).__name__}: {e}")
            st.stop()

    st.session_state[SESSION_RUN_KEY] = {
        "ventas_raw": raw_ventas_df.copy(),
        "ventas_raw_full": ventas_raw_full.copy(),
        "ventas": ventas.copy(),
        "stock": stock.copy(),
        "inbound": inbound.copy(),
        "det": det.copy(),
        "res": res.copy(),
        "prop": prop.copy(),
        "upload_run_log": upload_run_log,
        "mostrar_previo": mostrar_previo,
        "freq": str(freq),
        "horizon": int(horizon),
        "modo": modo,
        "sku_filtro": str(sku_q).strip() if (modo == "Por SKU" and sku_q) else "",
    }
    st.success("Listo. Usa el selector inferior para compras, reportería y ventas.")

run = st.session_state.get(SESSION_RUN_KEY)
if run is not None:
    st.divider()
    section = st.radio(
        "Sección",
        options=_SECTION_LABELS,
        horizontal=True,
        key=SESSION_SECTION_NAV_KEY,
        label_visibility="collapsed",
    )
    st.markdown("---")

    if section == _SECTION_LABELS[0]:
        st.subheader("Predictor de compras")
        if run.get("mostrar_previo"):
            st.subheader("Vista previa canónica")
            st.dataframe(run["ventas"].head(20), use_container_width=True)
            st.dataframe(run["stock"].head(20), use_container_width=True)
            st.dataframe(run["inbound"].head(50), use_container_width=True)
        ul = run.get("upload_run_log")
        if ul is not None:
            log_body, all_from_upload = ul
            if all_from_upload:
                st.success(log_body)
            else:
                st.info(log_body)
        st.subheader("Detalle")
        st.dataframe(run["det"].head(100), use_container_width=True)
        st.subheader("Resumen")
        st.dataframe(run["res"].head(100), use_container_width=True)
        st.subheader("Propuesta")
        st.dataframe(run["prop"].head(100), use_container_width=True)
        st.subheader("Exportar predicción (compras)")
        st.caption(
            "Excel en memoria: hojas Detalle, Resumen, Propuesta. No se guarda en el servidor."
        )
        try:
            xlsx_bytes = _prediction_tables_to_excel_bytes(
                run["det"], run["res"], run["prop"]
            )
        except ImportError:
            st.warning(
                "Instala **openpyxl** para exportar: `pip install openpyxl` "
                "(y en Render si aplica)."
            )
        else:
            ts = datetime.now().strftime("%Y-%m-%d_%H%M")
            fname = f"prediccion_compras_{ts}.xlsx"
            st.download_button(
                label="Descargar predicción (.xlsx)",
                data=xlsx_bytes,
                file_name=fname,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_prediccion_excel_v2",
            )
    elif section == _SECTION_LABELS[1]:
        st.subheader("Reportería de ventas")
        _render_reporteria_tab(run["ventas"], run["stock"])
    else:
        _render_ventas_predictor_tab()

_render_predictor_excel_footer()
