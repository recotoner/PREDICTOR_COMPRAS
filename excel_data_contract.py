# V2/excel_data_contract.py
"""
Contrato de datos entre ingesta Excel (ERP / exportes tipo Kame) y el núcleo
`predictor_core.forecast_all`.

IMPORTANTE
----------
- `app_predictor.py` (producción Sheets + Make) NO se modifica.
- Esta capa solo define nombres canónicos y reglas; la app Excel vive en
  `app_predictor_excel.py`.

Salidas canónicas (DataFrames)
------------------------------
1) Ventas (equivalente S1 / pestaña `ventas_raw` en Sheets)
   Columnas obligatorias: fecha, sku, qty
   - fecha: datetime
   - sku: str, mayúsculas, sin espacios laterales
   - qty: numérico (unidades); el core usa solo qty > 0 para demanda

2) Stock (equivalente S2 / `stock_snapshot`)
   Columnas obligatorias: sku, stock
   - stock: numérico >= 0 (existencia disponible en bodega)

3) Inbound / OC en tránsito (equivalente S3 / `inbound_po`)
   Columnas obligatorias: sku, qty, eta, estado
   - qty: unidades aún no recepcionadas (pendientes de bodega)
   - eta: fecha estimada de recepción; si falta, usar hoy para que el core
     las incluya en la ventana (ver filtro por fecha en predictor_core).
   - estado: texto libre; el core excluye estados \"cerrados\" (COMPLETA, etc.).
     Para OC pendientes se recomienda \"PENDIENTE\" o \"ABIERTA\".

4) Config (misma semántica que `normalize_config_sheet` en apps legacy)
   Mínimo por SKU: sku, lead_time_dias, minimo_compra, multiplo, seguridad_dias,
   proveedor, activo, alias, etc. La app Excel puede arrancar con config vacía
   y expandir a todos los SKU de ventas ∪ stock.

Regla de negocio OC (export Kame)
---------------------------------
- Incluir solo filas cuya columna de recepción (p. ej. \"Recepción\") sea
  exactamente **Pendiente** (comparación insensible a mayúsculas / acentos).
- Cantidad en tránsito: columna **Por Recibir** (fallback documentado en loader).

Mapeo por defecto de columnas fuente (exportes actuales en /V2)
----------------------------------------------------------------
- Ventas (hoja \"Reporte\"): Fecha → fecha, SKU → sku, Cantidad → qty
- Inventario (hoja \"Report\"): SKU → sku, Saldo → stock (sumar por SKU si hay varias bodegas)
- OC (hoja \"Report\"): filtrar Recepción == Pendiente; SKU → sku;
  Por Recibir → qty; eta = Fecha Recepción si válida, si no Fecha del documento, si no hoy.
"""

from __future__ import annotations

# Nombres de columnas EXACTOS que debe recibir predictor_core (post-normalización ligera interna).
VENTAS_COLS = ("fecha", "sku", "qty")
STOCK_COLS = ("sku", "stock")
INBOUND_COLS = ("sku", "qty", "eta", "estado")

# Valores de recepción en fuente OC que definen \"aún no en bodega\".
OC_RECEPCION_PENDIENTE = "pendiente"

# Estados que predictor_core trata como líneas cerradas (no aportan inbound).
# Copiado de la lógica en predictor_core._normalize_inbound (referencia).
INBOUND_CLOSED_STATES_REF = frozenset(
    {
        "CERRADA",
        "CERRADO",
        "CANCELADA",
        "CANCELADO",
        "ANULADA",
        "ANULADO",
        "RECEPCIONADA",
        "RECEPCIONADO",
        "INGRESADA",
        "INGRESADO",
        "CLOSED",
        "COMPLETE",
        "COMPLETA",
    }
)
