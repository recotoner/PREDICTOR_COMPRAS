# V2/excel_kame_loader.py — Ingesta de exportes Excel tipo Kame → DataFrames canónicos
from __future__ import annotations

import unicodedata as ud
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .excel_data_contract import INBOUND_COLS, OC_RECEPCION_PENDIENTE, STOCK_COLS, VENTAS_COLS


def _ascii_fold(s: str) -> str:
    t = str(s).replace("\u00a0", " ").strip()
    t = ud.normalize("NFKD", t)
    return "".join(ch for ch in t if not ud.combining(ch)).lower()


def _col_by_fold(columns: Iterable, target_fold: str) -> str | None:
    """Encuentra la primera columna cuyo nombre plegado coincide con target_fold."""
    for c in columns:
        if _ascii_fold(c) == target_fold:
            return str(c)
    return None


def _col_contains(columns: Iterable, *parts: str) -> str | None:
    folded = {str(c): _ascii_fold(c) for c in columns}
    for name, f in folded.items():
        if all(p in f for p in parts):
            return name
    return None


def read_first_sheet_path(path: Path, preferred_names: tuple[str, ...]) -> pd.DataFrame:
    xl = pd.ExcelFile(path)
    names = [str(x) for x in xl.sheet_names]
    for pref in preferred_names:
        for i, n in enumerate(names):
            if _ascii_fold(n) == _ascii_fold(pref):
                return pd.read_excel(path, sheet_name=xl.sheet_names[i])
    return pd.read_excel(path, sheet_name=0)


def read_first_sheet_bytes(data: bytes, preferred_names: tuple[str, ...]) -> pd.DataFrame:
    """Lee la primera hoja lógica desde bytes (sin escribir disco)."""
    bio = BytesIO(data)
    xl = pd.ExcelFile(bio)
    names = [str(x) for x in xl.sheet_names]
    for pref in preferred_names:
        for i, n in enumerate(names):
            if _ascii_fold(n) == _ascii_fold(pref):
                return pd.read_excel(BytesIO(data), sheet_name=xl.sheet_names[i])
    return pd.read_excel(BytesIO(data), sheet_name=0)


def _normalize_sku_series(s: pd.Series) -> pd.Series:
    sku = s.astype(str)
    sku = sku.replace(["nan", "NaN", "None", "<NA>", "NaT"], "")
    sku = sku.str.replace(r"\.0$", "", regex=True)
    return sku.str.strip().str.upper()


def build_sku_producto_map(raw: pd.DataFrame) -> dict[str, str]:
    """
    Diccionario sku (normalizado) → descripción desde columna Producto del export de ventas.
    Si un SKU aparece varias veces, se conserva la última descripción no vacía del archivo.
    """
    if raw is None or raw.empty:
        return {}
    c_sku = _col_by_fold(raw.columns, "sku")
    c_prod = _col_by_fold(raw.columns, "producto")
    if not c_prod:
        c_prod = _col_contains(raw.columns, "producto")
    if not c_sku or not c_prod:
        return {}
    tmp = pd.DataFrame()
    tmp["sku"] = _normalize_sku_series(raw[c_sku])
    tmp["producto"] = (
        raw[c_prod]
        .astype(str)
        .fillna("")
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )
    tmp = tmp[(tmp["sku"] != "") & (tmp["producto"] != "")]
    bad = tmp["producto"].str.upper().isin(["NAN", "NONE", "NAT"])
    tmp = tmp.loc[~bad]
    if tmp.empty:
        return {}
    tmp = tmp.drop_duplicates(subset=["sku"], keep="last")
    return dict(zip(tmp["sku"], tmp["producto"]))


def load_ventas_kame_df(raw: pd.DataFrame, source_label: str = "ventas") -> pd.DataFrame:
    """S1: DataFrame crudo de export ventas → fecha, sku, qty."""
    c_fecha = _col_by_fold(raw.columns, "fecha")
    c_sku = _col_by_fold(raw.columns, "sku")
    c_qty = _col_by_fold(raw.columns, "cantidad") or _col_by_fold(raw.columns, "qty")
    if not c_fecha or not c_sku or not c_qty:
        raise ValueError(
            f"Ventas ({source_label}): faltan columnas fecha, sku y cantidad (o qty). "
            f"Columnas encontradas: {list(raw.columns)}"
        )
    out = pd.DataFrame()
    out["fecha"] = pd.to_datetime(raw[c_fecha], errors="coerce", dayfirst=True)
    out["sku"] = _normalize_sku_series(raw[c_sku])
    out["qty"] = pd.to_numeric(raw[c_qty], errors="coerce").fillna(0.0)
    out = out[out["fecha"].notna()]
    out = out[out["sku"].notna() & (out["sku"].astype(str).str.strip() != "")]
    out = out[out["sku"].astype(str).str.upper() != "NAN"]
    return out[list(VENTAS_COLS)]


def load_stock_kame_df(raw: pd.DataFrame, source_label: str = "stock") -> pd.DataFrame:
    """S2: inventario bodega → sku, stock (suma Saldo por SKU si hay varias bodegas)."""
    c_sku = _col_by_fold(raw.columns, "sku")
    c_saldo = _col_contains(raw.columns, "saldo") or _col_by_fold(raw.columns, "stock")
    if not c_sku or not c_saldo:
        raise ValueError(
            f"Stock ({source_label}): faltan columnas sku y saldo (o stock). "
            f"Columnas encontradas: {list(raw.columns)}"
        )
    tmp = pd.DataFrame()
    tmp["sku"] = _normalize_sku_series(raw[c_sku])
    tmp["stock"] = pd.to_numeric(raw[c_saldo], errors="coerce").fillna(0.0).clip(lower=0)
    tmp = tmp[tmp["sku"] != ""]
    out = tmp.groupby("sku", as_index=False)["stock"].sum()
    return out[list(STOCK_COLS)]


def load_inbound_oc_kame_df(raw: pd.DataFrame, source_label: str = "órdenes de compra") -> pd.DataFrame:
    """S3: órdenes de compra — solo recepción Pendiente; cantidad = Por Recibir."""
    c_recepcion = None
    for c in raw.columns:
        if _ascii_fold(c) == "recepcion":
            c_recepcion = str(c)
            break
    c_sku = _col_by_fold(raw.columns, "sku")
    c_por_recibir = None
    for c in raw.columns:
        if "por" in _ascii_fold(c) and "recibir" in _ascii_fold(c):
            c_por_recibir = str(c)
            break
    c_fecha = _col_by_fold(raw.columns, "fecha")
    c_fecha_recep = None
    for c in raw.columns:
        f = _ascii_fold(c)
        if "fecha" in f and "recepcion" in f:
            c_fecha_recep = str(c)
            break

    if not c_recepcion or not c_sku or not c_por_recibir:
        raise ValueError(
            f"Órdenes de compra ({source_label}): faltan columnas recepción, sku y/o por recibir. "
            f"Columnas encontradas: {list(raw.columns)}"
        )

    df = raw.copy()
    rec_norm = df[c_recepcion].map(lambda x: _ascii_fold(str(x)) if pd.notna(x) else "")
    df = df[rec_norm == OC_RECEPCION_PENDIENTE]

    out = pd.DataFrame()
    out["sku"] = _normalize_sku_series(df[c_sku])
    out["qty"] = pd.to_numeric(df[c_por_recibir], errors="coerce").fillna(0.0).clip(lower=0)
    today = pd.Timestamp.today().normalize()
    if c_fecha_recep:
        eta1 = pd.to_datetime(df[c_fecha_recep], errors="coerce", dayfirst=True)
    else:
        eta1 = pd.Series(pd.NaT, index=df.index)
    if c_fecha:
        eta2 = pd.to_datetime(df[c_fecha], errors="coerce", dayfirst=True)
    else:
        eta2 = pd.Series(pd.NaT, index=df.index)
    out["eta"] = eta1.where(eta1.notna(), eta2)
    out["eta"] = out["eta"].where(out["eta"].notna(), today)
    out["estado"] = "PENDIENTE"
    out = out[(out["sku"] != "") & (out["qty"] > 0)]
    return out[list(INBOUND_COLS)]


def oc_recepcion_preflight(raw: pd.DataFrame) -> dict[str, Any]:
    """
    Información para validación UI (sin filtrar todavía a canónico).
    """
    c_recepcion = None
    for c in raw.columns:
        if _ascii_fold(c) == "recepcion":
            c_recepcion = str(c)
            break
    c_por_recibir = None
    for c in raw.columns:
        if "por" in _ascii_fold(c) and "recibir" in _ascii_fold(c):
            c_por_recibir = str(c)
            break
    out: dict[str, Any] = {
        "recepcion_col": c_recepcion,
        "por_recibir_col": c_por_recibir,
        "n_rows": len(raw),
        "n_pendiente": 0,
        "recepcion_values_sample": [],
    }
    if c_recepcion is None or raw.empty:
        return out
    vals = raw[c_recepcion].dropna().astype(str).unique().tolist()
    out["recepcion_values_sample"] = sorted(vals, key=str)[:15]
    rec_norm = raw[c_recepcion].map(lambda x: _ascii_fold(str(x)) if pd.notna(x) else "")
    out["n_pendiente"] = int((rec_norm == OC_RECEPCION_PENDIENTE).sum())
    return out


def load_ventas_kame(path: Path) -> pd.DataFrame:
    raw = read_first_sheet_path(path, ("Reporte", "Report"))
    return load_ventas_kame_df(raw, source_label=path.name)


def load_stock_kame(path: Path) -> pd.DataFrame:
    raw = read_first_sheet_path(path, ("Report", "Reporte"))
    return load_stock_kame_df(raw, source_label=path.name)


def load_inbound_oc_kame(path: Path) -> pd.DataFrame:
    raw = read_first_sheet_path(path, ("Report", "Reporte"))
    return load_inbound_oc_kame_df(raw, source_label=path.name)


def load_ventas_kame_bytes(data: bytes, source_label: str = "ventas.xlsx") -> pd.DataFrame:
    raw = read_first_sheet_bytes(data, ("Reporte", "Report"))
    return load_ventas_kame_df(raw, source_label=source_label)


def load_stock_kame_bytes(data: bytes, source_label: str = "stock.xlsx") -> pd.DataFrame:
    raw = read_first_sheet_bytes(data, ("Report", "Reporte"))
    return load_stock_kame_df(raw, source_label=source_label)


def load_inbound_oc_kame_bytes(data: bytes, source_label: str = "oc.xlsx") -> pd.DataFrame:
    raw = read_first_sheet_bytes(data, ("Report", "Reporte"))
    return load_inbound_oc_kame_df(raw, source_label=source_label)


def pick_excel_in_folder(folder: Path, name_substring: str) -> Path | None:
    """
    Primer archivo .xlsx o .xls en `folder` cuyo nombre contiene `name_substring` (sin ~$).
    """
    if not folder.is_dir():
        return None
    low = name_substring.lower()
    candidates = sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and not p.name.startswith("~$") and p.suffix.lower() in (".xlsx", ".xls")
    )
    for p in candidates:
        if low in p.name.lower():
            return p
    return None


def load_pack_from_folder(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Busca los tres Excel por nombre de archivo y devuelve (ventas, stock, inbound).
    Nombres esperados (substring, case-insensitive):
      informeventas, inventariobodega, ordenescompra
    """
    pv = pick_excel_in_folder(folder, "informeventas")
    ps = pick_excel_in_folder(folder, "inventariobodega")
    po = pick_excel_in_folder(folder, "ordenescompra")
    missing = []
    if pv is None:
        missing.append("informeventas")
    if ps is None:
        missing.append("inventariobodega")
    if po is None:
        missing.append("ordenescompra")
    if missing:
        raise FileNotFoundError(
            f"No se encontró en {folder} archivo(s) que contengan: {', '.join(missing)}"
        )
    v = load_ventas_kame(pv)
    s = load_stock_kame(ps)
    o = load_inbound_oc_kame(po)
    return v, s, o
