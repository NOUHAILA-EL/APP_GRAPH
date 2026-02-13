# app.py
# Describe → Chart (mini Power BI) with EN/FR parsing and robust charting

import re
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

# ---------------------------------
# Streamlit page setup
# ---------------------------------
st.set_page_config(page_title="Describe2Chart", page_icon="📊", layout="wide")
st.title("Cap Visual Intelligence powered by GenAI")

st.markdown("""
**Upload** your dataset, **describe** the visual (EN/FR), then click **Generate chart**.

"""**Examples (EN)**
- "Bar chart of **count of Description** grouped by **Region**; **sort descending by revenue and profit**"
- "Line chart of **Sales over Date**; **rolling average 7**; **color by Region**"
- "Pie (donut) of **sum of Sales by Region**; title: Sales share"
- "Heatmap of **sum of Profit by Region and Description**"
- "Table of **sum of Sales by Region**"
- "Matrix of **sum of Sales by Region and Description**"
- "KPI of **sum of Sales**; title: Total Sales"

**Exemples (FR)**
- "Un **camembert** de la **somme des ventes par région**"
- "Barres du **nombre de Description** par **Région**; **tri descendant par bénéfice**"
- "Courbe des **ventes sur Date**; **moyenne mobile 7**; **couleur par** Région"
- "Tableau de la **somme des ventes par région**"
- "Matrice de la **somme des ventes par région et description**"
- "KPI de la **somme des ventes**; **titre: CA total**" 
            """
""")

# ---------------------------------
# Upload & load
# ---------------------------------
uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

@st.cache_data
def load_df(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    elif name.endswith(".xlsx"):
        return pd.read_excel(file, engine="openpyxl")
    elif name.endswith(".xls"):
        return pd.read_excel(file, engine="xlrd")
    else:
        raise ValueError("Unsupported file type")

# ---------------------------------
# Dictionaries (tune to your schema)
# ---------------------------------
COL_SYNONYMS = {
    # EN
    "revenue": "Sales", "revenues": "Sales", "amount": "Sales", "sale": "Sales",
    "profit": "Profit", "margin": "Profit",
    "region": "Region", "date": "Date", "description": "Description",

    # FR
    "chiffre d'affaires": "Sales", "ca": "Sales", "ventes": "Sales",
    "bénéfice": "Profit", "benefice": "Profit", "marge": "Profit",
    "région": "Region",
}

AGG_SYNONYMS = {
    # EN
    "total": "sum", "sum": "sum", "average": "mean", "avg": "mean", "mean": "mean",
    "median": "median", "max": "max", "min": "min", "count": "count",
    # FR
    "somme": "sum", "moyenne": "mean", "mediane": "median",
    "nombre": "count", "compte": "count", "nb": "count"
}

CHART_ALIASES = {
    # EN
    "bar": "bar", "column": "bar", "stacked bar": "bar_stacked",
    "line": "line", "area": "area", "scatter": "scatter", "bubble": "scatter",
    "hist": "hist", "histogram": "hist", "box": "box", "boxplot": "box",
    "pie": "pie", "donut": "donut",
    "heatmap": "heatmap", "table": "table", "matrix": "matrix", "kpi": "kpi",

    # FR → EN
    "barres": "bar", "colonne": "bar", "empilé": "bar_stacked",
    "courbe": "line", "aire": "area", "nuage de points": "scatter",
    "camembert": "pie", "anneau": "donut",
    "carte thermique": "heatmap", "tableau": "table", "matrice": "matrix",
}

# ---------------------------------
# Parsing helpers
# ---------------------------------
def normalize_text(t: str) -> str:
    t = t.lower().strip()

    # Map common FR phrases to EN keywords or normal forms
    fr2en = {
        "diagramme en secteurs": "pie",
        "camembert": "pie",
        "nuage de points": "scatter",
        "couleur par": "color by",
        "moyenne mobile": "rolling average",
        "tri descendant": "sort desc",
        "tri ascendant": "sort asc",
        "titre": "title",
        "somme de": "sum of",
        "moyenne de": "avg of",
        "par ": "by ",
        "région": "region",
        "bénéfice": "profit",
        "ventes": "revenue"
    }
    for fr, en in fr2en.items():
        t = t.replace(fr, en)

    # remove parentheticals like "(metric)"
    t = re.sub(r"\([^)]*\)", "", t)
    t = re.sub(r"\s*:\s*", ": ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def resolve_col(freetext: str, cols: List[str]) -> Optional[str]:
    if not freetext:
        return None
    key = freetext.strip().lower()
    key = COL_SYNONYMS.get(key, key)
    # exact
    for c in cols:
        if str(c).lower() == key:
            return c
    # startswith
    for c in cols:
        if str(c).lower().startswith(key):
            return c
    # contains (last resort)
    for c in cols:
        if key in str(c).lower():
            return c
    return None

def infer_chart_type(t: str) -> str:
    # try direct keywords/aliases
    for k, v in CHART_ALIASES.items():
        if k in t:
            return v
    # defaults
    if "trend" in t or "over " in t:
        return "line"
    return "bar"

def parse_dimensions(t: str, cols: List[str]) -> List[str]:
    """
    Accepts: "grouped by X", "by X", "by X and Y", "by X, Y"
    Returns up to two dims.
    """
    dims = []
    m = re.search(r"(?:group(?:ed)? by|by)\s+([a-z0-9_ ,&-]+)", t)
    if m:
        raw = m.group(1)
        parts = [p.strip() for p in re.split(r",| and | & ", raw) if p.strip()]
        for p in parts:
            c = resolve_col(p, cols)
            if c and c not in dims:
                dims.append(c)
    return dims[:2]

def parse_metric_and_agg(t: str, df: pd.DataFrame) -> Tuple[str, Optional[str]]:
    """
    Returns (agg_func, metric_col).
    Handles "sum of Sales", "total revenue", "count of description", "total description".
    If a text col is used with a numeric agg, switch to count.
    """
    cols = df.columns.tolist()
    dtypes = df.dtypes.to_dict()

    m = re.search(r"(sum|avg|average|mean|median|max|min|count|total)\s+(?:of\s+)?([a-z0-9_ -]+)", t)
    if m:
        agg_raw, metric_raw = m.groups()
        agg = AGG_SYNONYMS.get(agg_raw, agg_raw)
        metric_col = resolve_col(metric_raw, cols)
        if metric_col and not pd.api.types.is_numeric_dtype(dtypes[metric_col]) and agg in ["sum", "mean", "median", "max", "min"]:
            agg = "count"
        return agg, metric_col

    m2 = re.search(r"(total|count)\s+([a-z0-9_ -]+)", t)
    if m2:
        agg = AGG_SYNONYMS.get(m2.group(1), "sum")
        metric_col = resolve_col(m2.group(2), cols)
        if metric_col and not pd.api.types.is_numeric_dtype(dtypes[metric_col]) and agg != "count":
            agg = "count"
        return agg, metric_col

    # fallback: first numeric (sum) or count rows
    num_like = [c for c in cols if pd.api.types.is_numeric_dtype(dtypes[c])]
    if num_like:
        return "sum", num_like[0]
    return "count", cols[0] if cols else None

def parse_sort(t: str, df: pd.DataFrame) -> Tuple[Optional[str], List[str]]:
    """
    Returns (order: 'ascending'/'descending'/None, [cols]).
    Example: "sort descending by revenue and profit"
    """
    order = None
    cols_order: List[str] = []
    m = re.search(r"sort\s+(ascending|descending|asc|desc)(?:\s+by)?\s+([a-z0-9_ ,&and]+)", t)
    if m:
        o_raw = m.group(1)
        order = "ascending" if o_raw in ["ascending", "asc"] else "descending"
        raw_cols = [c.strip() for c in re.split(r",|and|&", m.group(2)) if c.strip()]
        for rc in raw_cols:
            c = resolve_col(rc, df.columns.tolist())
            if c and c not in cols_order:
                cols_order.append(c)
    return order, cols_order

def parse_color_size(t: str, cols: List[str]) -> Tuple[Optional[str], Optional[str]]:
    color = size = None
    m = re.search(r"(?:color by|colour by|group by)\s+([a-z0-9_ -]+)", t)
    if m:
        color = resolve_col(m.group(1), cols)
    m2 = re.search(r"size by\s+([a-z0-9_ -]+)", t)
    if m2:
        size = resolve_col(m2.group(1), cols)
    return color, size

def parse_rolling(t: str) -> Optional[int]:
    m = re.search(r"(rolling average|moving average)\s+(\d+)", t)
    return int(m.group(2)) if m else None

def parse_title(t: str) -> Optional[str]:
    m = re.search(r"title\s*:\s*(.+)$", t)
    return m.group(1).strip() if m else None

def agg_name(func: str, col: str) -> str:
    if func == "count":
        return f"count_{col}" if col else "count"
    return f"{func}_{col}"

# ---------------------------------
# Aggregation layer
# ---------------------------------
def aggregate_for_chart(
    df: pd.DataFrame,
    dims: List[str],
    agg_func: str,
    metric_col: Optional[str],
    extra_sort_cols: List[str]
) -> Tuple[pd.DataFrame, List[str], str, List[str]]:
    """
    Returns (work_df, used_dims, measure_col, sortable_cols)
    - Aggregates df by up to two dims
    - Creates main measure column
    - Aggregates extra_sort_cols (sum/count) for sorting purpose
    """
    dtypes = df.dtypes.to_dict()
    cols = df.columns.tolist()

    # Choose default dims if none
    used_dims = list(dims) if dims else []
    if not used_dims:
        cat_like = [c for c in cols if not pd.api.types.is_numeric_dtype(dtypes[c])]
        used_dims = [cat_like[0]] if cat_like else [cols[0]]
    used_dims = used_dims[:2]

    # Main measure name
    if agg_func == "count":
        y_col = agg_name("count", metric_col if metric_col else used_dims[0])
    else:
        if metric_col is None or metric_col not in cols:
            num_like = [c for c in cols if pd.api.types.is_numeric_dtype(dtypes[c])]
            metric_col = num_like[0] if num_like else used_dims[0]
        y_col = agg_name(agg_func, metric_col)

    group_keys = used_dims

    # Build aggregation dict
    agg_dict = {}
    if agg_func == "count":
        if metric_col and metric_col in cols:
            agg_dict[metric_col] = "count"
        # else we'll use size() below
    else:
        agg_dict[metric_col] = agg_func

    # extra sort metrics
    for sc in extra_sort_cols:
        if sc == metric_col:
            # already aggregated; keep as is (we will rename consistently below)
            continue
        if sc in cols:
            if pd.api.types.is_numeric_dtype(dtypes[sc]):
                agg_dict[sc] = "sum"
            else:
                agg_dict[sc] = "count"

    # Perform aggregation
    if agg_func == "count" and (not metric_col or metric_col not in cols):
        work = df.groupby(group_keys, as_index=False).size().rename(columns={"size": y_col})
    else:
        work = df.groupby(group_keys, as_index=False).agg(agg_dict)
        # normalize the main measure name
        if agg_func != "count" and metric_col in work.columns and metric_col != y_col:
            work = work.rename(columns={metric_col: y_col})

    # Rename extra sort columns to consistent names (sum_*, count_*)
    sortable_cols: List[str] = [y_col]
    for sc in extra_sort_cols:
        if sc in work.columns and sc != y_col:
            # decide if it was summed or counted
            if pd.api.types.is_numeric_dtype(dtypes.get(sc, None)):
                new_name = agg_name("sum", sc)
            else:
                new_name = agg_name("count", sc)
            if new_name not in work.columns:
                work = work.rename(columns={sc: new_name})
            sortable_cols.append(new_name)
        else:
            # if it was the metric itself and renamed to y_col, we can still sort by y_col
            pass

    # For one‑dim heatmaps/matrix we may need a second dim later, but here we just return.
    return work, used_dims, y_col, list(dict.fromkeys(sortable_cols))  # dedup while preserving order

# ---------------------------------
# UI controls
# ---------------------------------
desc = st.text_area(
    "Describe the chart you want (EN/FR supported)",
    height=120,
    placeholder="e.g., 'Bar chart of count of Description by Region; sort descending by revenue and profit'"
)
generate = st.button("Generate chart", type="primary")

# ---------------------------------
# Data preview
# ---------------------------------
if uploaded:
    try:
        df = load_df(uploaded)
        st.success(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
        show_head_only = st.toggle("Preview only first 5 rows", value=False)
        st.dataframe(df.head() if show_head_only else df, use_container_width=True, height=320)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        df = None
else:
    df = None

# ---------------------------------
# Render only on click
# ---------------------------------
if generate:
    if df is None:
        st.warning("Please upload a dataset first.")
    elif not desc or not desc.strip():
        st.info("Please describe your chart above, then click Generate.")
    else:
        try:
            t = normalize_text(desc)
            chart_type = infer_chart_type(t)

            # Parse components
            dims = parse_dimensions(t, df.columns.tolist())
            agg_func, metric_col = parse_metric_and_agg(t, df)
            order, sort_cols_raw = parse_sort(t, df)
            color_by, size_by = parse_color_size(t, df.columns.tolist())
            roll_n = parse_rolling(t)
            title = parse_title(t)

            # Build aggregation (also prepares extra sort cols)
            work, used_dims, y_col, sortable_cols = aggregate_for_chart(df, dims, agg_func, metric_col, sort_cols_raw)

            # If heatmap/matrix require 2 dims and we only have 1, try to infer a second categorical
            if chart_type in ["heatmap", "matrix"] and len(used_dims) < 2:
                dtypes = df.dtypes.to_dict()
                cat_like = [c for c in df.columns if (not pd.api.types.is_numeric_dtype(dtypes[c])) and c not in used_dims]
                if cat_like:
                    used_dims = used_dims + [cat_like[0]]
                    work, used_dims, y_col, sortable_cols = aggregate_for_chart(df, used_dims, agg_func, metric_col, sort_cols_raw)

            # Sorting
            effective_sort_cols: List[str] = []
            for sc in sort_cols_raw:
                # map to aggregated names if needed
                candidates = [agg_name("sum", sc), agg_name("count", sc), agg_name("mean", sc), agg_name("median", sc),
                              agg_name("max", sc), agg_name("min", sc), sc]
                chosen = next((c for c in candidates if c in work.columns), None)
                if chosen and chosen not in effective_sort_cols:
                    effective_sort_cols.append(chosen)
            if not effective_sort_cols:
                effective_sort_cols = [y_col]
            if order is None:
                order = "descending"  # sensible default for measures
            work = work.sort_values(by=effective_sort_cols, ascending=[order == "ascending"] * len(effective_sort_cols))

            # Rolling average (line charts): apply on y after sorting by first dim if that dim looks time-like
            if chart_type == "line" and roll_n and len(used_dims) >= 1:
                dim0 = used_dims[0]
                if pd.api.types.is_datetime64_any_dtype(df[dim0]) or any(k in str(dim0).lower() for k in ["date", "time", "month", "year"]):
                    work = work.sort_values(by=dim0)
                    ra_col = f"{y_col}_rolling_{roll_n}"
                    work[ra_col] = work[y_col].rolling(roll_n, min_periods=1).mean()
                    y_col = ra_col

            # Category order (for x axis)
            dim1 = used_dims[0]
            dim2 = used_dims[1] if len(used_dims) > 1 else None
            ordered_dim1 = work[dim1].astype(str).tolist()
            seen = set()
            ordered_dim1 = [c for c in ordered_dim1 if not (c in seen or seen.add(c))]

            # -------- Chart builders --------
            chart = None
            default_title = title or f"{chart_type.capitalize()} of {y_col}" + (f" by {dim1}" if dim1 else "") + (f" and {dim2}" if dim2 else "")

            if chart_type in ["pie", "donut"]:
                # Pie/Donut need categorical + numeric
                cat = dim1
                if y_col not in work.columns or not pd.api.types.is_numeric_dtype(work[y_col]):
                    st.warning("Pie/Donut needs a numeric measure (e.g., 'sum of Sales by Region').")
                else:
                    base = alt.Chart(work).encode(
                        theta=alt.Theta(field=y_col, type="quantitative"),
                        color=alt.Color(field=cat, type="nominal", sort=ordered_dim1),
                        tooltip=[cat, y_col]
                    )
                    chart = base.mark_arc(innerRadius=80) if chart_type == "donut" else base.mark_arc()

            elif chart_type == "heatmap":
                if dim2 is None:
                    st.warning("Heatmap needs two categorical dimensions (e.g., 'by Region and Description').")
                else:
                    chart = alt.Chart(work).mark_rect().encode(
                        x=alt.X(dim1, type="nominal", sort=ordered_dim1),
                        y=alt.Y(dim2, type="nominal"),
                        color=alt.Color(y_col, type="quantitative"),
                        tooltip=[dim1, dim2, y_col]
                    )

            elif chart_type == "hist":
                # Use raw df and chosen metric (or first numeric)
                metric_for_hist = metric_col
                if metric_for_hist is None or metric_for_hist not in df.columns or not pd.api.types.is_numeric_dtype(df[metric_for_hist]):
                    num_like = [c for c in df.columns if pd.api.types.is_numeric_dtype(df.dtypes[c])]
                    metric_for_hist = num_like[0] if num_like else None
                if metric_for_hist is None:
                    st.warning("No numeric column available for histogram.")
                else:
                    chart = alt.Chart(df).mark_bar().encode(
                        x=alt.X(alt.Bin(maxbins=30), field=metric_for_hist, type='quantitative'),
                        y='count()',
                        tooltip=[metric_for_hist]
                    )
                    default_title = title or f"Histogram of {metric_for_hist}"

            elif chart_type == "box":
                if not pd.api.types.is_numeric_dtype(work[y_col]):
                    st.warning("Box plot requires a numeric measure. Try 'sum of Sales by Region'.")
                else:
                    base = work.copy()
                    if dim1 is None:
                        base["All"] = "All"
                        dim_x = "All"
                    else:
                        dim_x = dim1
                    chart = alt.Chart(base).mark_boxplot().encode(
                        x=alt.X(dim_x, sort=ordered_dim1 if dim_x == dim1 else None),
                        y=y_col,
                        color=(color_by if color_by in base.columns else alt.value(None)),
                        tooltip=list(base.columns)
                    )

            elif chart_type == "scatter":
                # Try to use two numeric columns: y_col and another numeric
                num_like = [c for c in work.columns if pd.api.types.is_numeric_dtype(work[c])]
                xnum = next((c for c in num_like if c != y_col), None)
                if xnum is None:
                    st.warning("Scatter needs two numeric axes. Try 'scatter of Profit vs Sales'.")
                else:
                    chart = alt.Chart(work).mark_point(filled=True, size=80).encode(
                        x=xnum, y=y_col,
                        color=(color_by if color_by in work.columns else alt.value(None)),
                        size=(size_by if (size_by in work.columns and pd.api.types.is_numeric_dtype(work[size_by])) else alt.value(60)),
                        tooltip=list(work.columns)
                    )
                    default_title = title or f"Scatter: {y_col} vs {xnum}"

            elif chart_type == "line":
                chart = alt.Chart(work).mark_line(point=True).encode(
                    x=alt.X(dim1, sort=ordered_dim1),
                    y=y_col,
                    color=(color_by if color_by in work.columns else alt.value(None)),
                    tooltip=list(work.columns)
                )

            elif chart_type == "area":
                chart = alt.Chart(work).mark_area(opacity=0.7).encode(
                    x=alt.X(dim1, sort=ordered_dim1),
                    y=y_col,
                    color=(color_by if color_by in work.columns else alt.value(None)),
                    tooltip=list(work.columns)
                )

            elif chart_type in ["bar", "bar_stacked"]:
                enc = {
                    "x": alt.X(dim1, sort=ordered_dim1),
                    "y": y_col,
                    "tooltip": list(work.columns)
                }
                if chart_type == "bar_stacked" and dim2:
                    enc["color"] = dim2
                elif color_by and color_by in work.columns:
                    enc["color"] = color_by
                chart = alt.Chart(work).mark_bar().encode(**enc)

            elif chart_type == "table":
                st.subheader(title or "Table")
                st.dataframe(work, use_container_width=True, height=420)
                default_title = None  # already shown as table

            elif chart_type == "matrix":
                if dim2 is None:
                    st.warning("Matrix needs two dimensions (e.g., 'by Region and Description').")
                else:
                    mat = work.pivot_table(index=dim1, columns=dim2, values=y_col, aggfunc="sum", fill_value=0)
                    st.subheader(title or "Matrix")
                    st.dataframe(mat, use_container_width=True, height=420)
                    default_title = None

            elif chart_type == "kpi":
                # Show single value (aggregate without group), so recompute w/o dims
                aframe, _, ykpi, _ = aggregate_for_chart(df, [], agg_func, metric_col, [])
                val = aframe[ykpi].iloc[0]
                st.metric(label=title or ykpi, value=f"{val:,.2f}" if isinstance(val, (int, float, np.number)) else str(val))
                default_title = None

            # Render chart if built
            if chart is not None:
                st.subheader(default_title or "")
                st.altair_chart(chart.properties(width='container', height=420), use_container_width=True)

            # Download aggregated data (when applicable)
            if chart_type not in ["kpi"] and work is not None and not work.empty:
                st.download_button(
                    "Download chart data as CSV",
                    data=work.to_csv(index=False).encode("utf-8"),
                    file_name="chart_data.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Could not build the chart: {e}")
