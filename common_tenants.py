# common_tenants.py
import pandas as pd
from urllib.parse import quote
import streamlit as st

CLIENTES_SHEET_ID = "1Pbjxy_V-NuTbfnN_SLpexkYx_w62Umsg7eBr2qrQJrI"
TAB_CLIENTES = "clientes_config"

@st.cache_data(ttl=300)
def load_tenants():
    url = (
        f"https://docs.google.com/spreadsheets/d/{CLIENTES_SHEET_ID}/gviz/tq?"
        f"tqx=out:csv&sheet={quote(TAB_CLIENTES, safe='')}"
    )
    df = pd.read_csv(url)
    df.columns = [c.strip().lower() for c in df.columns]
    if "activo" in df.columns:
        df = df[df["activo"] == True]
    return df

def get_current_tenant():
    tenants_df = load_tenants()

    # 1) query param
    params = st.query_params
    tenant_from_url = params.get("tenant", None)
    if isinstance(tenant_from_url, list):
        tenant_from_url = tenant_from_url[0]

    if "tenant_id" not in st.session_state:
        st.session_state["tenant_id"] = None

    if tenant_from_url:
        st.session_state["tenant_id"] = tenant_from_url

    if not st.session_state["tenant_id"]:
        if not tenants_df.empty:
            st.session_state["tenant_id"] = tenants_df.iloc[0]["tenant_id"]

    # sidebar selector (para ti)
    tenant_ids = tenants_df["tenant_id"].tolist()
    tenant_names = tenants_df.set_index("tenant_id")["tenant_name"].to_dict()

    sel = st.sidebar.selectbox(
        "Cliente",
        tenant_ids,
        format_func=lambda t: tenant_names.get(t, t),
        index=tenant_ids.index(st.session_state["tenant_id"]) if st.session_state["tenant_id"] in tenant_ids else 0
    )

    if sel != st.session_state["tenant_id"]:
        st.session_state["tenant_id"] = sel
        st.query_params["tenant"] = sel  # actualiza URL

    current = tenants_df[tenants_df["tenant_id"] == st.session_state["tenant_id"]].iloc[0]
    return current
