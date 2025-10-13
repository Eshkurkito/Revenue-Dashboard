import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import date

from utils import load_groups, parse_dates

# NOTA: no llamamos st.set_page_config aquí (hazlo solo en streamlit_app.py)

def _ensure_parsed(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Alojamiento","Fecha alta","Fecha entrada","Fecha salida","Alquiler con IVA (€)"])
    dfx = parse_dates(df.copy())
    dfx["Alquiler con IVA (€)"] = pd.to_numeric(dfx.get("Alquiler con IVA (€)"), errors="coerce").fillna(0.0)
    dfx = dfx.dropna(subset=["Fecha entrada","Fecha salida"])
    dfx = dfx[dfx["Fecha salida"] > dfx["Fecha entrada"]]
    dfx["los"] = (dfx["Fecha salida"] - dfx["Fecha entrada"]).dt.days.clip(lower=1)
    dfx["adr_reserva"] = dfx["Alquiler con IVA (€)"] / dfx["los"]
    return dfx

def _expand_daily(dfx: pd.DataFrame, p_start: pd.Timestamp, p_end: pd.Timestamp, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Expande a diario por propiedad, filtrando por cutoff y solape con el periodo."""
    if dfx.empty:
        return pd.DataFrame(columns=["date","Alojamiento","nights","revenue","adr"])
    dfx = dfx[(dfx["Fecha alta"].notna()) & (dfx["Fecha alta"] <= pd.to_datetime(cutoff).normalize())]
    if dfx.empty:
        return pd.DataFrame(columns=["date","Alojamiento","nights","revenue","adr"])

    p_start = pd.to_datetime(p_start).normalize()
    p_end   = pd.to_datetime(p_end).normalize()

    # Rango de días (end inclusivo)
    dfx["in_start"] = dfx["Fecha entrada"].clip(lower=p_start)
    dfx["in_end"]   = (dfx["Fecha salida"]).clip(upper=p_end + pd.Timedelta(days=1))
    dfx["n_in"] = (dfx["in_end"] - dfx["in_start"]).dt.days
    dfx = dfx[dfx["n_in"] > 0].copy()

    # Fechas de estancia en el rango
    dfx["dates"] = dfx.apply(lambda r: pd.date_range(r["in_start"], r["in_end"] - pd.Timedelta(days=1), freq="D"), axis=1)
    exploded = dfx.loc[dfx["dates"].str.len() > 0, ["Alojamiento","dates","adr_reserva"]].explode("dates")
    exploded = exploded.rename(columns={"dates":"date"})
    exploded["nights"] = 1.0
    exploded["revenue"] = exploded["adr_reserva"]
    # Agregar por día-propiedad
    out = exploded.groupby(["date","Alojamiento"], as_index=False).agg(
        nights=("nights","sum"),
        revenue=("revenue","sum"),
    )
    out["adr"] = np.where(out["nights"] > 0, out["revenue"]/out["nights"], 0.0)
    return out.sort_values(["date","Alojamiento"])

def _map_groups(df_daily: pd.DataFrame, groups: dict) -> pd.DataFrame:
    if df_daily.empty:
        df_daily["Grupo"] = []
        return df_daily
    prop_to_group = {}
    for g, props in (groups or {}).items():
        for p in props:
            prop_to_group[str(p)] = str(g)
    df_daily["Grupo"] = df_daily["Alojamiento"].astype(str).map(prop_to_group).fillna("Sin grupo")
    return df_daily

def _inv_by_group(df_scope: pd.DataFrame, groups: dict, props_scope: list[str] | None) -> dict:
    """Inventario por grupo = número de alojamientos únicos dentro del ámbito."""
    inv = {}
    if props_scope:
        props_in = set(map(str, props_scope))
    else:
        props_in = set(df_scope["Alojamiento"].astype(str).unique())
    # contar por grupo
    for g, props in (groups or {}).items():
        inv[g] = len(set(map(str, props)).intersection(props_in))
    # Sin grupo
    known = set(p for lst in (groups or {}).values() for p in map(str, lst))
    other = len(props_in - known)
    if other > 0:
        inv["Sin grupo"] = other
    # Limpieza
    for g in list(inv.keys()):
        if inv[g] <= 0:
            inv.pop(g, None)
    return inv

def _dow_map():
    # 0=L,1=M,2=X,3=J,4=V,5=S,6=D
    return ["L","M","X","J","V","S","D"]

def render_what_if(raw: pd.DataFrame | None = None):
    """Módulo What-if avanzado (precio y pickup) con desglose por grupos."""
    st.header("🧪 What‑if avanzado (precio y pickup)")

    # Dataset activo (si no se pasa, intentar session_state)
    if raw is None:
        raw = st.session_state.get("df_active") or st.session_state.get("raw", pd.DataFrame())
    if raw is None or raw.empty:
        st.info("No hay datos cargados. Vuelve a la portada y sube un CSV/Excel.")
        return

    groups = load_groups()
    active_group_name = st.session_state.get("active_group_name", None)
    active_props = st.session_state.get("active_props", [])

    with st.sidebar:
        st.markdown("—")
        scope = st.selectbox("Ámbito de análisis", ["Grupo activo", "Todos los grupos", "Seleccionar grupos"], index=0 if active_group_name else 1)
        if scope == "Seleccionar grupos":
            sel_groups = st.multiselect("Elige grupos", options=sorted(groups.keys()), default=sorted(groups.keys()))
            sel_props = sorted({p for g in sel_groups for p in groups.get(g, [])})
        elif scope == "Grupo activo" and active_props:
            sel_groups = [active_group_name] if active_group_name else []
            sel_props = active_props
        else:
            sel_groups = []
            sel_props = []  # todos

        # Parámetros de periodo y corte
        today = date.today()
        cut = st.date_input("Fecha de corte (OTB)", value=today)
        colp1, colp2 = st.columns(2)
        p_start = colp1.date_input("Inicio del periodo", value=today.replace(day=1))
        p_end   = colp2.date_input("Fin del periodo", value=today)

        st.divider()
        st.caption("Parámetros del escenario")
        price_delta = st.slider("Cambio de precio (ADR) global %", -40, 40, 0, 1)
        adv = st.checkbox("Ajuste de precio por día de semana", value=False)
        dow_delta = {i: price_delta for i in range(7)}
        if adv:
            dcols = st.columns(7)
            for i, lab in enumerate(_dow_map()):
                dow_delta[i] = dcols[i].number_input(f"{lab} %", value=int(price_delta), step=1, key=f"dow_{i}")

        delta_pickup = st.number_input("Delta de pickup (noches)", value=0, step=10)
        adr_tail = st.number_input("ADR del pickup (tail) €", value=0.0, step=5.0, help="Si 0, se usará ADR base del grupo.")
        st.caption("Distribución del pickup")
        dist_group = st.selectbox("Entre grupos", ["Proporcional a noches base", "Uniforme"])
        dist_days  = st.selectbox("Dentro de cada grupo", ["Uniforme", "Priorizar valles (días con menos noches)", "Solo fines de semana"])

        st.divider()
        inv_override_global = st.number_input("Inventario global (opcional, 0=auto)", min_value=0, value=0, step=1)

        # → NUEVO: selección de alojamientos sueltos
        st.caption("Alojamientos sueltos (opcional)")
        props_all = sorted(raw["Alojamiento"].dropna().astype(str).unique()) if "Alojamiento" in raw.columns else []
        props_pool = sorted(set(map(str, sel_props))) if sel_props else props_all
        selected_props = st.multiselect(
            "Alojamientos", options=props_pool, default=[],
            help="Deja vacío para incluir todos los alojamientos del ámbito seleccionado."
        )

    # Preparación
    dfx = _ensure_parsed(raw)
    # Aplicar filtro de ámbito y alojamientos sueltos (si hay)
    filter_props = selected_props if selected_props else sel_props
    if filter_props:
        dfx = dfx[dfx["Alojamiento"].astype(str).isin(list(map(str, filter_props)))].copy()

    daily = _expand_daily(dfx, p_start, p_end, cut)
    if daily.empty:
        st.warning("No hay noches en el periodo a este corte con los filtros actuales.")
        return

    # Mapear grupos e inventario por grupo
    daily = _map_groups(daily, groups)
    inv_map = _inv_by_group(dfx if not sel_props else dfx[dfx["Alojamiento"].astype(str).isin(sel_props)], groups, sel_props)

    # KPIs base por grupo
    base_g = (
        daily.groupby("Grupo", as_index=False)
        .agg(nights=("nights","sum"), revenue=("revenue","sum"), props=("Alojamiento","nunique"))
        .sort_values("revenue", ascending=False)
    )
    base_g["adr"] = np.where(base_g["nights"] > 0, base_g["revenue"]/base_g["nights"], 0.0)

    # Inventario del grupo (override si global > 0)
    nights_period = max((pd.to_datetime(p_end) - pd.to_datetime(p_start)).days + 1, 1)
    base_g["inv"] = base_g["Grupo"].map(inv_map).fillna(0).astype(int)
    if inv_override_global and int(inv_override_global) > 0:
        total_inv = int(inv_override_global)
        weights = (base_g["inv"].replace(0, np.nan)).fillna(0)
        if weights.sum() > 0:
            base_g["inv"] = np.floor(weights / weights.sum() * total_inv).astype(int).clip(lower=1)
    base_g["occ_pct"] = np.where(nights_period > 0, base_g["nights"] / (base_g["inv"].clip(lower=1) * nights_period) * 100.0, 0.0)

    # Escenario: precio por DOW
    daily["dow"] = pd.to_datetime(daily["date"]).dt.dayofweek
    daily["adr_scn"] = daily.apply(lambda r: r["adr"] * (1.0 + (dow_delta.get(int(r["dow"]), price_delta))/100.0), axis=1)

    # Revenue base y escenario (solo precio aplicado a noches base)
    daily["rev_base"] = daily["nights"] * daily["adr"]
    daily["rev_price_scn"] = daily["nights"] * daily["adr_scn"]

    # Distribución del pickup entre grupos
    total_delta = int(delta_pickup)
    by_group_nights = daily.groupby("Grupo")["nights"].sum()
    groups_list = base_g["Grupo"].tolist()

    if total_delta != 0 and groups_list:
        if dist_group.startswith("Uniforme"):
            shares = {g: 1/len(groups_list) for g in groups_list}
        else:
            total_n = by_group_nights.sum() if by_group_nights.sum() > 0 else 1.0
            shares = {g: float(by_group_nights.get(g,0.0))/total_n for g in groups_list}
        extra_g = {g: 0 for g in groups_list}
        remain = abs(total_delta)
        order = sorted(groups_list, key=lambda x: -shares[x])
        i = 0
        while remain > 0:
            extra_g[order[i % len(order)]] += 1
            remain -= 1
            i += 1
        if total_delta < 0:
            extra_g = {g: -v for g, v in extra_g.items()}
    else:
        extra_g = {g: 0 for g in groups_list}

    # Distribución dentro de cada grupo por días
    daily["extra"] = 0.0
    for g, delta_g in extra_g.items():
        if delta_g == 0:
            continue
        sub = daily[daily["Grupo"] == g].copy()
        if sub.empty:
            continue
        if dist_days.startswith("Solo fines"):
            sub = sub[pd.to_datetime(sub["date"]).dt.dayofweek.isin([4,5,6])]  # V,S,D
            if sub.empty:
                continue
        if dist_days.startswith("Priorizar"):
            order_idx = sub.sort_values("nights").index.tolist()
        else:
            order_idx = sub.sort_values("date").index.tolist()
        if delta_g > 0:
            i = 0
            remain = delta_g
            while remain > 0 and order_idx:
                daily.loc[order_idx[i % len(order_idx)], "extra"] += 1.0
                i += 1
                remain -= 1
        else:
            i = 0
            remain = -delta_g
            order_idx = sub.sort_values("nights", ascending=False).index.tolist()
            while remain > 0 and order_idx:
                idx = order_idx[i % len(order_idx)]
                if (daily.loc[idx, "nights"] + daily.loc[idx, "extra"]) > 0:
                    daily.loc[idx, "extra"] -= 1.0
                    remain -= 1
                i += 1

    # Ingresos del pickup extra a ADR tail (si 0, usar ADR medio del grupo base)
    if adr_tail and adr_tail > 0:
        daily["adr_tail_use"] = float(adr_tail)
    else:
        adr_grp = daily.groupby("Grupo")["adr"].mean()
        daily["adr_tail_use"] = daily["Grupo"].map(adr_grp).fillna(daily["adr"])
    daily["rev_extra"] = daily["extra"].clip(lower=0) * daily["adr_tail_use"]

    # Totales por grupo (base vs escenario)
    agg_base = daily.groupby("Grupo", as_index=False).agg(
        nights=("nights","sum"),
        rev_base=("rev_base","sum"),
    )
    agg_price = daily.groupby("Grupo", as_index=False).agg(
        rev_price=("rev_price_scn","sum"),
        extra=("extra","sum"),
        rev_extra=("rev_extra","sum"),
    )
    res = agg_base.merge(agg_price, on="Grupo", how="outer").fillna(0.0)
    res["n_scn"] = (res["nights"] + res["extra"]).clip(lower=0)
    res["rev_scn"] = res["rev_price"] + res["rev_extra"]

    # KPIs calculados
    res["adr_base"] = np.where(res["nights"] > 0, res["rev_base"]/res["nights"], 0.0)
    res["adr_scn"] = np.where(res["n_scn"] > 0, res["rev_scn"]/res["n_scn"], res["adr_base"])
    res["inv"] = res["Grupo"].map(base_g.set_index("Grupo")["inv"]).fillna(1).astype(int)
    res["occ_base"] = np.where(nights_period > 0, res["nights"] / (res["inv"].clip(lower=1)*nights_period) * 100.0, 0.0)
    res["occ_scn"]  = np.where(nights_period > 0, res["n_scn"]  / (res["inv"].clip(lower=1)*nights_period) * 100.0, 0.0)

    # Totales generales
    total_row = pd.DataFrame({
        "Grupo": ["TOTAL"],
        "nights": [res["nights"].sum()],
        "rev_base": [res["rev_base"].sum()],
        "rev_price": [res["rev_price"].sum()],
        "extra": [res["extra"].sum()],
        "rev_extra": [res["rev_extra"].sum()],
        "n_scn": [res["n_scn"].sum()],
        "rev_scn": [res["rev_scn"].sum()],
        "adr_base": [ (res["rev_base"].sum() / res["nights"].sum()) if res["nights"].sum() > 0 else 0.0 ],
        "adr_scn":  [ (res["rev_scn"].sum()  / res["n_scn"].sum())  if res["n_scn"].sum()  > 0 else 0.0 ],
        "inv": [max(int(base_g["inv"].sum()), 1)],
        "occ_base": [ (res["nights"].sum() / (max(int(base_g["inv"].sum()),1) * nights_period) * 100.0) if nights_period>0 else 0.0 ],
        "occ_scn":  [ (res["n_scn"].sum()  / (max(int(base_g["inv"].sum()),1) * nights_period) * 100.0) if nights_period>0 else 0.0 ],
    })
    res_full = pd.concat([res, total_row], ignore_index=True)

    # KPIs generales
    total = res_full[res_full["Grupo"]=="TOTAL"].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Noches (base)", f"{total['nights']:,.0f}".replace(",", "."))
    c2.metric("ADR (base) €", f"{total['adr_base']:,.2f}")
    c3.metric("Ingresos (base) €", f"{total['rev_base']:,.2f}")
    c4.metric("Ocupación (base) %", f"{total['occ_base']:,.1f}%")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Noches (esc.)", f"{total['n_scn']:,.0f}".replace(",", "."), f"{(total['n_scn']-total['nights']):+.0f}".replace(",", "."))
    s2.metric("ADR (esc.) €", f"{total['adr_scn']:,.2f}", f"{(total['adr_scn']-total['adr_base']):+.2f}")
    s3.metric("Ingresos (esc.) €", f"{total['rev_scn']:,.2f}", f"{(total['rev_scn']-total['rev_base']):+.2f}")
    s4.metric("Ocupación (esc.) %", f"{total['occ_scn']:,.1f}%", f"{(total['occ_scn']-total['occ_base']):+.1f} p.p.")

    st.subheader("Desglose por grupo")
    show_cols = ["Grupo","inv","nights","n_scn","extra","adr_base","adr_scn","rev_base","rev_scn","occ_base","occ_scn"]
    grid = res_full[show_cols].rename(columns={
        "inv":"Inventario",
        "nights":"Noches base",
        "n_scn":"Noches esc.",
        "extra":"Δ pickup",
        "adr_base":"ADR base",
        "adr_scn":"ADR esc.",
        "rev_base":"Ingresos base (€)",
        "rev_scn":"Ingresos esc. (€)",
        "occ_base":"Occ base %",
        "occ_scn":"Occ esc. %",
    })
    st.dataframe(grid, use_container_width=True)

    # Gráfico ingresos por grupo (base vs escenario)
    plot = res[["Grupo","rev_base","rev_scn"]].melt(id_vars=["Grupo"], var_name="Serie", value_name="Ingresos")
    plot["Serie"] = plot["Serie"].map({"rev_base":"Base","rev_scn":"Escenario"})
    chart = (
        alt.Chart(plot)
        .mark_bar()
        .encode(
            x=alt.X("Grupo:N", sort="-y", title=None),
            y=alt.Y("Ingresos:Q", title="Ingresos (€)"),
            color=alt.Color("Serie:N", scale=alt.Scale(domain=["Base","Escenario"], range=["#9e9e9e","#1f77b4"]), title=None),
            tooltip=[alt.Tooltip("Grupo:N"), alt.Tooltip("Serie:N"), alt.Tooltip("Ingresos:Q", format=",.2f")],
        )
        .properties(height=340)
    )
    st.altair_chart(chart, use_container_width=True)

    # Exportaciones
    st.download_button(
        "📥 Descargar tabla por grupo (CSV)",
        data=res_full.to_csv(index=False).encode("utf-8-sig"),
        file_name="what_if_grupos.csv",
        mime="text/csv",
    )

    # Detalle diario opcional
    with st.expander("Detalle diario (por grupo)"):
        day_grp = daily.groupby(["date","Grupo"], as_index=False).agg(
            nights=("nights","sum"), extra=("extra","sum"),
            rev_base=("rev_base","sum"), rev_price=("rev_price_scn","sum"), rev_extra=("rev_extra","sum")
        )
        day_grp["n_scn"] = (day_grp["nights"] + day_grp["extra"]).clip(lower=0)
        day_grp["rev_scn"] = day_grp["rev_price"] + day_grp["rev_extra"]
        st.dataframe(day_grp.tail(60), use_container_width=True)
        st.download_button(
            "📥 Descargar detalle diario (CSV)",
            data=day_grp.to_csv(index=False).encode("utf-8-sig"),
            file_name="what_if_diario_grupo.csv",
            mime="text/csv",
        )