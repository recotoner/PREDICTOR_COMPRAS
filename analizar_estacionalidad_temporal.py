"""
Script temporal para analizar estacionalidad de SKUs específicos.
NO modifica el código principal, solo es para análisis.
"""
import pandas as pd
import numpy as np
from urllib.parse import quote
import time
import requests

# Configuración - usar el mismo método que la app
CLIENTES_SHEET_ID = "1Pbjxy_V-NuTbfnN_SLpexkYx_w62Umsg7eBr2qrQJrI"
TAB_CLIENTES = "clientes_config"
TAB_VENTAS = "ventas_raw"

def read_gsheets(sheet_id: str, tab: str) -> pd.DataFrame:
    """Lee una pestaña de Google Sheets como CSV."""
    sheet_param = quote(tab, safe="")
    url = (f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?"
           f"tqx=out:csv&sheet={sheet_param}&t={int(time.time())}")
    
    try:
        response = requests.get(url, timeout=30)
        response.encoding = 'utf-8'
        csv_content = response.text
        from io import StringIO
        df = pd.read_csv(StringIO(csv_content))
        return df
    except Exception as e:
        print(f"Error leyendo {tab}: {e}")
        return pd.DataFrame()

def normalize_ventas(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza datos de ventas (similar a app_predictor.py)."""
    if df.empty:
        return pd.DataFrame(columns=["fecha", "sku", "qty"])
    
    # Buscar columnas relevantes
    cols_lower = {str(c).lower(): c for c in df.columns}
    
    fecha_col = None
    for k in ["fecha", "date", "fecha_venta"]:
        if k in cols_lower:
            fecha_col = cols_lower[k]
            break
    
    sku_col = None
    for k in ["sku", "codigo", "cod", "producto"]:
        if k in cols_lower:
            sku_col = cols_lower[k]
            break
    
    qty_col = None
    for k in ["cantidad", "qty", "qty.", "unidades"]:
        if k in cols_lower:
            qty_col = cols_lower[k]
            break
    
    if not fecha_col or not sku_col or not qty_col:
        print(f"Columnas encontradas: {df.columns.tolist()}")
        return pd.DataFrame(columns=["fecha", "sku", "qty"])
    
    out = pd.DataFrame()
    out["fecha"] = pd.to_datetime(df[fecha_col], errors="coerce")
    out["sku"] = df[sku_col].astype(str).str.strip().str.upper()
    
    # Normalizar cantidad (manejar formatos con puntos/comas)
    qty_str = df[qty_col].astype(str).str.replace('\xa0', '', regex=False)
    qty_str = qty_str.str.replace('.', '', regex=False)
    qty_str = qty_str.str.replace(',', '.', regex=False)
    out["qty"] = pd.to_numeric(qty_str, errors="coerce").fillna(0)
    
    # Solo ventas positivas (como hace _prep_series)
    out = out[out["qty"] > 0].copy()
    out = out.dropna(subset=["fecha"])
    
    return out[["fecha", "sku", "qty"]]

def analizar_estacionalidad(sku: str, ventas: pd.DataFrame):
    """Analiza estacionalidad mensual de un SKU."""
    print(f"\n{'='*70}")
    print(f"ANÁLISIS DE ESTACIONALIDAD: {sku}")
    print(f"{'='*70}")
    
    # Filtrar por SKU
    ventas_sku = ventas[ventas["sku"] == sku].copy()
    
    if ventas_sku.empty:
        print(f"❌ No se encontraron ventas para {sku}")
        return
    
    # Agregar por mes
    ventas_sku["period"] = ventas_sku["fecha"].dt.to_period("M").dt.to_timestamp()
    ventas_mensual = ventas_sku.groupby("period", as_index=False)["qty"].sum()
    ventas_mensual = ventas_mensual.sort_values("period")
    
    print(f"\n📊 RESUMEN GENERAL:")
    print(f"   Total de registros de venta: {len(ventas_sku)}")
    print(f"   Períodos únicos (meses): {len(ventas_mensual)}")
    print(f"   Rango de fechas: {ventas_mensual['period'].min()} a {ventas_mensual['period'].max()}")
    print(f"   Total unidades vendidas: {ventas_mensual['qty'].sum():.2f}")
    print(f"   Media mensual: {ventas_mensual['qty'].mean():.2f}")
    print(f"   Mediana mensual: {ventas_mensual['qty'].median():.2f}")
    print(f"   Desviación estándar: {ventas_mensual['qty'].std():.2f}")
    
    if len(ventas_mensual) < 12:
        print(f"\n⚠️  Solo hay {len(ventas_mensual)} meses de datos (menos de 1 año)")
        print(f"   Esto puede limitar la detección de estacionalidad anual.")
    
    # Análisis de estacionalidad por mes del año
    ventas_sku["mes"] = ventas_sku["fecha"].dt.month
    estacionalidad = ventas_sku.groupby("mes", as_index=False)["qty"].agg({
        'total': 'sum',
        'media': 'mean',
        'conteo': 'count'
    })
    
    if len(estacionalidad) > 1:
        # Calcular coeficiente de variación por mes
        media_global = ventas_mensual['qty'].mean()
        std_global = ventas_mensual['qty'].std()
        cv = (std_global / media_global) if media_global > 0 else 0
        
        print(f"\n📈 ANÁLISIS DE ESTACIONALIDAD POR MES DEL AÑO:")
        print(f"   Coeficiente de variación (CV): {cv:.3f}")
        print(f"   {'Alta variabilidad' if cv > 0.5 else 'Baja variabilidad' if cv < 0.3 else 'Variabilidad moderada'}")
        
        meses_nombres = {
            1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
            5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
            9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
        }
        
        print(f"\n   Desglose por mes del año:")
        estacionalidad = estacionalidad.sort_values("mes")
        for _, row in estacionalidad.iterrows():
            mes_nombre = meses_nombres.get(row['mes'], f"Mes {row['mes']}")
            variacion_pct = ((row['media'] - media_global) / media_global * 100) if media_global > 0 else 0
            print(f"     {mes_nombre:12s}: Total={row['total']:8.2f}, Media={row['media']:8.2f}, "
                  f"Registros={int(row['conteo']):3d}, Var={variacion_pct:+6.1f}%")
        
        # Detectar meses con mayor/menor demanda
        mes_max = estacionalidad.loc[estacionalidad['media'].idxmax()]
        mes_min = estacionalidad.loc[estacionalidad['media'].idxmin()]
        print(f"\n   📌 Mes con MAYOR demanda promedio: {meses_nombres.get(mes_max['mes'])} "
              f"(Media: {mes_max['media']:.2f})")
        print(f"   📌 Mes con MENOR demanda promedio: {meses_nombres.get(mes_min['mes'])} "
              f"(Media: {mes_min['media']:.2f})")
        diferencia_pct = ((mes_max['media'] - mes_min['media']) / mes_min['media'] * 100) if mes_min['media'] > 0 else 0
        print(f"   📊 Diferencia: {diferencia_pct:.1f}% (mayor vs menor)")
    
    # Análisis de tendencia (primera mitad vs segunda mitad)
    if len(ventas_mensual) >= 12:
        mid_point = len(ventas_mensual) // 2
        primera_mitad = ventas_mensual.iloc[:mid_point]['qty'].mean()
        segunda_mitad = ventas_mensual.iloc[mid_point:]['qty'].mean()
        trend_ratio = ((segunda_mitad - primera_mitad) / primera_mitad * 100) if primera_mitad > 0 else 0
        
        print(f"\n📉 ANÁLISIS DE TENDENCIA:")
        print(f"   Media primera mitad: {primera_mitad:.2f}")
        print(f"   Media segunda mitad: {segunda_mitad:.2f}")
        print(f"   Variación de tendencia: {trend_ratio:+.1f}%")
        print(f"   {'✅ Tendencia detectada' if abs(trend_ratio) > 20 else '⚠️  No hay tendencia clara (variación < 20%)'}")
        print(f"   (El modelo robusto requiere >20% para detectar tendencia)")
    
    # Mostrar últimos 12 meses
    print(f"\n📅 ÚLTIMOS 12 PERÍODOS HISTÓRICOS:")
    ultimos_12 = ventas_mensual.tail(12) if len(ventas_mensual) >= 12 else ventas_mensual
    for _, row in ultimos_12.iterrows():
        periodo_str = row['period'].strftime('%Y-%m')
        print(f"   {periodo_str}: {row['qty']:8.2f}")
    
    print(f"\n{'='*70}\n")

def main():
    """Análisis principal."""
    print("🔍 ANALIZADOR DE ESTACIONALIDAD TEMPORAL")
    print("=" * 70)
    
    # 1. Cargar configuración de tenants
    print("\n1️⃣ Cargando configuración de tenants...")
    tenants_df = read_gsheets(CLIENTES_SHEET_ID, TAB_CLIENTES)
    if tenants_df.empty:
        print("❌ No se pudo cargar la configuración de tenants")
        return
    
    tenants_df.columns = [c.strip().lower() for c in tenants_df.columns]
    if "activo" in tenants_df.columns:
        tenants_df = tenants_df[tenants_df["activo"] == True]
    
    print(f"   ✅ Encontrados {len(tenants_df)} tenants activos")
    
    # 2. SKUs a analizar
    skus_a_analizar = ["NC-HPCE285A", "NC-BRTN1060"]
    
    # 3. Analizar cada tenant para encontrar los SKUs
    print(f"\n2️⃣ Buscando SKUs {skus_a_analizar} en todos los tenants...")
    
    for idx, tenant in tenants_df.iterrows():
        tenant_id = tenant.get("tenant_id", "")
        sheet_id = tenant.get("sheet_id", "")
        tenant_name = tenant.get("tenant_name", tenant_id)
        
        if not sheet_id:
            continue
        
        print(f"\n   🔎 Revisando tenant: {tenant_name} (ID: {tenant_id})")
        
        # Cargar ventas
        ventas_raw = read_gsheets(sheet_id, TAB_VENTAS)
        if ventas_raw.empty:
            print(f"      ⚠️  No hay datos de ventas en este tenant")
            continue
        
        ventas = normalize_ventas(ventas_raw)
        
        # Verificar si alguno de los SKUs está en este tenant
        skus_encontrados = [sku for sku in skus_a_analizar if sku in ventas["sku"].values]
        
        if skus_encontrados:
            print(f"      ✅ SKUs encontrados: {skus_encontrados}")
            for sku in skus_encontrados:
                analizar_estacionalidad(sku, ventas)
        else:
            print(f"      ℹ️  No se encontraron los SKUs en este tenant")

if __name__ == "__main__":
    main()



