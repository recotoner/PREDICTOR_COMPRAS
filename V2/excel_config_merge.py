# V2/excel_config_merge.py — Config de compras desde Excel opcional (GLOBAL + por SKU)
from __future__ import annotations

import unicodedata as ud

import pandas as pd

from .excel_config_defaults import default_config_from_union

CONFIG_GLOBAL_SKU = "GLOBAL"

CONFIG_COLUMNS = (
    "sku",
    "lead_time_dias",
    "seguridad_dias",
    "minimo_compra",
    "multiplo",
    "activo",
    "proveedor",
)

OVERRIDE_COLUMNS = (
    "proveedor",
    "lead_time_dias",
    "seguridad_dias",
    "minimo_compra",
    "multiplo",
    "activo",
)

_COLUMN_ALIASES = {
    "min_lote": "minimo_compra",
    "minimo_lote": "minimo_compra",
    "multiplo_lote": "multiplo",
    "lead_time": "lead_time_dias",
    "seguridad": "seguridad_dias",
}


def _ascii_fold(s: str) -> str:
    t = str(s).replace("\u00a0", " ").strip()
    t = ud.normalize("NFKD", t)
    return "".join(ch for ch in t if not ud.combining(ch)).lower()


def _parse_activo(value) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return True
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("0", "false", "no", "n", "inactive", "inactivo", "falso"):
        return False
    return True


def _coerce_config_column(col: str, value):
    if col == "proveedor":
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        return str(value).strip()
    if col == "activo":
        return _parse_activo(value)
    if col in ("lead_time_dias", "seguridad_dias", "minimo_compra", "multiplo"):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return float(pd.to_numeric(value, errors="coerce"))
    return value


def normalize_config_upload(raw: pd.DataFrame) -> pd.DataFrame:
    """Normaliza hoja de config subida; filas con sku (incl. GLOBAL). Sin fila GLOBAL en salida."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=list(CONFIG_COLUMNS))

    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    rename = {k: v for k, v in _COLUMN_ALIASES.items() if k in df.columns}
    if rename:
        df = df.rename(columns=rename)

    if "sku" not in df.columns:
        return pd.DataFrame(columns=list(CONFIG_COLUMNS))

    df["sku"] = (
        df["sku"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .str.upper()
    )
    df = df[df["sku"].notna() & (df["sku"] != "")]
    if df.empty:
        return pd.DataFrame(columns=list(CONFIG_COLUMNS))

    out = df[["sku"]].copy()
    for col in OVERRIDE_COLUMNS:
        out[col] = df[col] if col in df.columns else None

    for col in OVERRIDE_COLUMNS:
        out[col] = out[col].map(lambda v, c=col: _coerce_config_column(c, v))

    out = out.drop_duplicates(subset=["sku"], keep="last")
    return out.reset_index(drop=True)


def build_config_final(
    ventas: pd.DataFrame,
    stock: pd.DataFrame,
    config_raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    1. base = default_config_from_union(ventas, stock)
    2. Sin archivo config → base (idéntico al comportamiento histórico)
    3–7. GLOBAL → todos los SKU; filas SKU pisan GLOBAL; sin fila GLOBAL en salida
    """
    base = default_config_from_union(ventas, stock)
    if config_raw is None or config_raw.empty:
        return base

    parsed = normalize_config_upload(config_raw)
    if parsed.empty:
        return base

    out = base.copy()
    global_rows = parsed[parsed["sku"] == CONFIG_GLOBAL_SKU]
    sku_rows = parsed[parsed["sku"] != CONFIG_GLOBAL_SKU]

    if not global_rows.empty:
        grow = global_rows.iloc[-1]
        for col in OVERRIDE_COLUMNS:
            val = grow.get(col)
            coerced = _coerce_config_column(col, val)
            if coerced is None:
                continue
            out[col] = coerced

    if not sku_rows.empty:
        by_sku = sku_rows.set_index("sku", drop=False)
        for idx in out.index:
            sku = str(out.at[idx, "sku"])
            if sku not in by_sku.index:
                continue
            srow = by_sku.loc[sku]
            if isinstance(srow, pd.DataFrame):
                srow = srow.iloc[-1]
            for col in OVERRIDE_COLUMNS:
                val = srow.get(col)
                coerced = _coerce_config_column(col, val)
                if coerced is None:
                    continue
                out.at[idx, col] = coerced

    for col, default in [
        ("lead_time_dias", 0),
        ("seguridad_dias", 0),
        ("minimo_compra", 1),
        ("multiplo", 1),
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(default).astype(int)

    out["proveedor"] = out["proveedor"].fillna("").astype(str)
    out["activo"] = out["activo"].apply(_parse_activo)

    if "alias" not in out.columns:
        out["alias"] = None
    if "predictor_url" not in out.columns:
        out["predictor_url"] = None
    if "reporteria_url" not in out.columns:
        out["reporteria_url"] = None

    assert CONFIG_GLOBAL_SKU not in set(out["sku"].astype(str))
    return out
