#!/usr/bin/env python3
"""
batch_update.py - Script standalone para ejecución BATCH de actualizaciones

Este script:
- NO toca Streamlit
- NO modifica Make
- Ejecuta webhooks S1, S2, S3 en orden fijo para cada tenant
- Lee tenants desde clientes_config
- URLs de webhooks desde variables de entorno
- Maneja errores por tenant (log + continuar)
- Loguea resultados de cada ejecución

Nota: Make.com es el único responsable de escribir timestamps y STATUS_FINAL en las hojas config.
Este script solo actúa como gatillo para ejecutar los webhooks.

Uso:
    python batch_update.py

Variables de entorno requeridas:
    DEFAULT_SHEET_ID: Sheet ID del sheet principal (donde está clientes_config)
    DEFAULT_WEBHOOK_S1_URL: URL del webhook S1 (opcional, usa default si no está)
    DEFAULT_WEBHOOK_S2_URL: URL del webhook S2 (opcional, usa default si no está)
    DEFAULT_WEBHOOK_S3_URL: URL del webhook S3 (opcional, usa default si no está)

Nota: Este script NO escribe en Google Sheets. Make.com es el único responsable de escribir
timestamps y STATUS_FINAL en las hojas config.
"""

import os
import sys
import time
import logging
import requests
import argparse
import pandas as pd
from typing import Optional
from urllib.parse import quote

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_update.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ======================================
# CONFIGURACIÓN
# ======================================

# Modos de ejecución
MODE_FULL = "full"
MODE_INBOUND = "inbound"

# Sheet ID por defecto (donde está clientes_config)
DEFAULT_SHEET_ID = os.getenv("DEFAULT_SHEET_ID", "1Pbjxy_V-NuTbfnN_SLpexkYx_w62Umsg7eBr2qrQJrI")
TAB_CLIENTES_CONF = "clientes_config"
TAB_CONFIG = "config"

# Webhooks por defecto (pueden ser sobrescritos por variables de entorno)
DEFAULT_WEBHOOK_S1_URL = os.getenv(
    "DEFAULT_WEBHOOK_S1_URL",
    "https://hook.us1.make.com/qfr459tm0yth3xjbsjl3ef7vq3m44hwv"
)
DEFAULT_WEBHOOK_S2_URL = os.getenv(
    "DEFAULT_WEBHOOK_S2_URL",
    "https://hook.us1.make.com/vdj87rfcjpmeuccds9vieu45410tnsug"
)
DEFAULT_WEBHOOK_S3_URL = os.getenv(
    "DEFAULT_WEBHOOK_S3_URL",
    "https://hook.us1.make.com/k50t6u1rtrswqd6vl4s8mqf2ndu6noa3"
)

# Timeout para webhooks (segundos)
WEBHOOK_TIMEOUT = 30

# ======================================
# FUNCIONES DE LECTURA DE GOOGLE SHEETS
# ======================================

def read_gsheets(sheet_id: str, tab: str) -> pd.DataFrame:
    """Lee una pestaña de Google Sheets como CSV."""
    sheet_param = quote(tab, safe="")
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_param}"
    
    try:
        response = requests.get(url, timeout=30)
        response.encoding = 'utf-8'
        csv_content = response.text
        
        from io import StringIO
        df = pd.read_csv(
            StringIO(csv_content),
            dtype=str,
            na_filter=False,
            keep_default_na=False,
            on_bad_lines='skip',
            engine='python'
        )
        
        # Limpiar representaciones de NaN/None
        df = df.replace(['nan', 'NaN', 'None', '<NA>', 'NaT', 'null', 'NULL'], '')
        return df
    except Exception as e:
        logger.error(f"Error leyendo sheet {sheet_id}, tab {tab}: {e}")
        return pd.DataFrame()

def ensure_clientes_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Asegura que el DataFrame de clientes_config tenga las columnas necesarias."""
    if df is None or df.empty:
        return pd.DataFrame()
    
    d = df.copy()
    d.columns = [str(c).replace("\u00a0", " ").strip() for c in d.columns]
    
    # Normalizar nombres de columnas
    if "tenant_name" not in d.columns and "tenant" in d.columns:
        d = d.rename(columns={"tenant": "tenant_name"})
    if "is_active" not in d.columns and "activo" in d.columns:
        d["is_active"] = d["activo"]
    
    # Columnas requeridas
    required_cols = [
        "tenant_id", "tenant_name", "sheet_id",
        "webhook_s1", "webhook_s2", "webhook_s3",
        "is_active"
    ]
    
    for col in required_cols:
        if col not in d.columns:
            if col == "is_active":
                d[col] = True
            else:
                d[col] = ""
    
    # Normalizar columnas string
    for col in ["tenant_id", "tenant_name", "sheet_id", "webhook_s1", "webhook_s2", "webhook_s3"]:
        if col in d.columns:
            d[col] = d[col].astype(str).str.strip()
    
    return d

def load_clientes_config() -> Optional[pd.DataFrame]:
    """Lee la pestaña clientes_config del sheet por defecto."""
    try:
        df = read_gsheets(DEFAULT_SHEET_ID, TAB_CLIENTES_CONF)
        if df.empty:
            logger.warning(f"No se pudo leer {TAB_CLIENTES_CONF} del sheet {DEFAULT_SHEET_ID}")
            return None
        
        df = ensure_clientes_columns(df)
        
        # Filtrar solo tenants activos
        if "is_active" in df.columns:
            # Convertir is_active a boolean
            df["is_active"] = df["is_active"].astype(str).str.lower().isin(["true", "1", "yes", "sí", "si"])
            df = df[df["is_active"] == True].reset_index(drop=True)
        
        return df
    except Exception as e:
        logger.error(f"Error cargando clientes_config: {e}")
        return None

# ======================================
# FUNCIONES DE WEBHOOKS
# ======================================

def trigger_webhook(url: str, payload: dict) -> dict:
    """Ejecuta un webhook y retorna el resultado."""
    if not url or not url.strip():
        return {"ok": False, "error": "webhook no configurado"}
    
    try:
        r = requests.post(url, json=payload, timeout=WEBHOOK_TIMEOUT)
        return {
            "ok": r.ok,
            "status": r.status_code,
            "text": r.text[:500] if r.text else ""
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ======================================
# NOTA: ESCRITURA EN GOOGLE SHEETS
# ======================================
# Este script NO escribe en Google Sheets.
# Make.com es el único responsable de escribir timestamps y STATUS_FINAL
# en las hojas config después de completar cada escenario.

# ======================================
# FUNCIÓN PRINCIPAL DE ACTUALIZACIÓN
# ======================================

def update_tenant(tenant_row: pd.Series, mode: str = MODE_FULL) -> bool:
    """
    Actualiza un tenant según el modo especificado.
    
    Modo FULL: Ejecuta S1, S2, S3 en orden.
    Modo INBOUND: Solo ejecuta S3.
    
    Args:
        tenant_row: Fila del DataFrame de clientes_config con la info del tenant
        mode: Modo de ejecución (full o inbound)
    
    Returns:
        True si todas las actualizaciones del modo fueron exitosas, False en caso contrario
    """
    tenant_id = str(tenant_row.get("tenant_id", "")).strip()
    tenant_name = str(tenant_row.get("tenant_name", "")).strip()
    sheet_id = str(tenant_row.get("sheet_id", "")).strip()
    
    if not tenant_id or not sheet_id:
        logger.error(f"Tenant sin tenant_id o sheet_id: {tenant_name}")
        return False
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Procesando tenant: {tenant_name} (ID: {tenant_id})")
    logger.info(f"Modo: {mode.upper()}")
    logger.info(f"Sheet ID: {sheet_id}")
    logger.info(f"{'='*60}")
    
    # Obtener URLs de webhooks (del tenant o usar defaults)
    webhook_s1 = str(tenant_row.get("webhook_s1", "")).strip() or DEFAULT_WEBHOOK_S1_URL
    webhook_s2 = str(tenant_row.get("webhook_s2", "")).strip() or DEFAULT_WEBHOOK_S2_URL
    webhook_s3 = str(tenant_row.get("webhook_s3", "")).strip() or DEFAULT_WEBHOOK_S3_URL
    
    # --------------------------------------
    # MODO FULL (S1, S2, S3)
    # --------------------------------------
    if mode == MODE_FULL:
        # Ejecutar S1 (ventas)
        logger.info(f"\n[1/3] Ejecutando S1 (ventas)...")
        payload_s1 = {"reason": "batch_update_full", "tenant_id": tenant_id}
        result_s1 = trigger_webhook(webhook_s1, payload_s1)
        
        if not result_s1.get("ok"):
            logger.error(f"  ❌ Error en S1: {result_s1.get('error', result_s1.get('text', 'Unknown error'))}")
            return False
        
        logger.info(f"  ✓ S1 ejecutado correctamente (status: {result_s1.get('status')})")
        time.sleep(2)  # Pequeña pausa entre webhooks
        
        # Ejecutar S2 (stock total)
        logger.info(f"\n[2/3] Ejecutando S2 (stock total)...")
        payload_s2 = {"reason": "batch_update_full", "tenant_id": tenant_id, "use_stock_total": True}
        result_s2 = trigger_webhook(webhook_s2, payload_s2)
        
        if not result_s2.get("ok"):
            logger.error(f"  ❌ Error en S2: {result_s2.get('error', result_s2.get('text', 'Unknown error'))}")
            return False
        
        logger.info(f"  ✓ S2 ejecutado correctamente (status: {result_s2.get('status')})")
        time.sleep(2)  # Pequeña pausa entre webhooks
        
        # Ejecutar S3 (inbound)
        logger.info(f"\n[3/3] Ejecutando S3 (inbound)...")
        payload_s3 = {"reason": "batch_update_full", "tenant_id": tenant_id}
        result_s3 = trigger_webhook(webhook_s3, payload_s3)
        
        if not result_s3.get("ok"):
            logger.error(f"  ❌ Error en S3: {result_s3.get('error', result_s3.get('text', 'Unknown error'))}")
            return False
        
        logger.info(f"  ✓ S3 ejecutado correctamente (status: {result_s3.get('status')})")

    # --------------------------------------
    # MODO INBOUND (Solo S3)
    # --------------------------------------
    elif mode == MODE_INBOUND:
        logger.info(f"\n[1/1] Ejecutando S3 (inbound) - MODO INBOUND...")
        payload_s3 = {"reason": "batch_update_inbound", "tenant_id": tenant_id}
        result_s3 = trigger_webhook(webhook_s3, payload_s3)
        
        if not result_s3.get("ok"):
            logger.error(f"  ❌ Error en S3: {result_s3.get('error', result_s3.get('text', 'Unknown error'))}")
            return False
        
        logger.info(f"  ✓ S3 ejecutado correctamente (status: {result_s3.get('status')})")
    
    # Nota: Make.com es responsable de escribir timestamps y STATUS_FINAL
    logger.info(f"\n✅ Tenant {tenant_name} procesado correctamente")
    logger.info(f"   ℹ️  Make.com escribirá timestamps y STATUS_FINAL='DONE' en la hoja config")
    return True

# ======================================
# FUNCIÓN PRINCIPAL
# ======================================

def main():
    """Función principal del script."""
    # Parsear argumentos
    parser = argparse.ArgumentParser(description="Script standalone para ejecución BATCH de actualizaciones")
    parser.add_argument(
        "--mode", 
        choices=[MODE_FULL, MODE_INBOUND], 
        default=MODE_FULL,
        help="Modo de ejecución: full (S1+S2+S3) o inbound (solo S3). Default: full."
    )
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info(f"Iniciando batch_update.py - MODO: {args.mode.upper()}")
    logger.info("="*60)
    
    # Cargar configuración de tenants
    logger.info("\nCargando configuración de tenants...")
    clientes_df = load_clientes_config()
    
    if clientes_df is None or clientes_df.empty:
        logger.error("No se pudo cargar clientes_config o no hay tenants activos")
        return 1
    
    logger.info(f"✓ {len(clientes_df)} tenant(s) activo(s) encontrado(s)")
    
    # Procesar cada tenant
    success_count = 0
    error_count = 0
    
    for idx, row in clientes_df.iterrows():
        try:
            if update_tenant(row, mode=args.mode):
                success_count += 1
            else:
                error_count += 1
                logger.error(f"❌ Error procesando tenant: {row.get('tenant_name', 'Unknown')}")
        except Exception as e:
            error_count += 1
            logger.error(f"❌ Excepción procesando tenant {row.get('tenant_name', 'Unknown')}: {e}", exc_info=True)
        
        # Pausa entre tenants
        if idx < len(clientes_df) - 1:
            time.sleep(5)
    
    # Resumen final
    logger.info("\n" + "="*60)
    logger.info("RESUMEN FINAL")
    logger.info("="*60)
    logger.info(f"Modo ejecución: {args.mode.upper()}")
    logger.info(f"Total tenants procesados: {len(clientes_df)}")
    logger.info(f"✓ Exitosos: {success_count}")
    logger.info(f"❌ Errores: {error_count}")
    logger.info("="*60)
    
    return 0 if error_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())

