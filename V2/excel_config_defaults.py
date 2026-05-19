# Config por defecto cuando no hay hoja Google (misma idea que normalize_config_sheet vacío).
from __future__ import annotations

import pandas as pd


def default_config_from_union(ventas: pd.DataFrame, stock: pd.DataFrame) -> pd.DataFrame:
    vs = ventas.loc[ventas["sku"].notna(), "sku"].astype(str).str.strip().str.upper()
    ss = stock.loc[stock["sku"].notna(), "sku"].astype(str).str.strip().str.upper()
    vs = vs[vs != ""]
    ss = ss[ss != ""]
    skus = sorted(set(vs) | set(ss))
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
