# app.py
# Streamlit "Describe to Chart" – a mini-Power BI-like tool
# EN + FR parsing, synonyms, multi-sort, many visual types (Altair + Plotly)

import re
import io
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Describe2Chart (mini Power BI)", page_icon="📊", layout="wide")
st.title("📊 Describe2Chart — Create a chart from your data + description")

st.markdown("""
Upload your data, describe the visual you want (EN/FR), then click **Generate chart**.  
**Examples (EN):**
- "Bar chart of **count of Description** grouped by **Region**; **sort descending by revenue and profit**"
- "Pie (donut) of **sum of Sales by Region**; title: Sales share"
- "Line chart of **Sales over Date**; rolling average 7; color by Region"
- "Heatmap of **sum of Profit by Month and Region**"

**Exemples (FR):**
- "Un **camembert** de la **somme des ventes par région**"
- "Barres du **nombre de Description** par **Région**; **tri descendant par bénéfice**"
- "Courbe des **ventes sur Date**; **moyenne mobile 7**; **couleur par** Région"
""")

# ----------------------------
# File upload & loading
# ----------------------------
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

# ----------------------------
# Parsing dictionaries
# ----------------------------
COL_SYNONYMS = {
    # EN
    "revenue": "Sales", "revenues": "Sales", "amount": "Sales", "sale": "Sales",
    "profit": "Profit", "margin": "Profit",
    "region": "Region", "date": "Date", "description": "Description",
    # FR
    "chiffre d'affaires": "Sales", "ca": "Sales", "ventes": "Sales",
    "bénéfice": "Profit", "marge": "Profit",
    "région": "Region",
}

AGG_SYNONYMS = {
    # EN
    "total": "sum", "sum": "sum", "average": "mean", "avg": "mean", "mean": "mean",
    "median": "median", "max": "max", "min": "min",
    "count": "count",
    # FR
    "somme": "sum", "moyenne": "mean", "mediane": "median", "nombre": "count",
    "compte": "count", "nb": "count",
}

CHART_KEYWORDS = {
    # EN
    "bar": "bar", "column": "bar", "line": "line", "area": "area", "scatter": "scatter",
    "bubble": "bubble", "histogram": "hist", "hist": "hist", "box": "box", "boxplot": "box",
    "pie": "pie", "donut": "donut", "heatmap": "heatmap", "treemap": "treemap",
    "waterfall": "waterfall", "funnel": "funnel", "kpi": "kpi", "table": "table", "matrix": "matrix",
    # FR
    "barres": "bar", "colonne": "bar", "courbe": "line", "aire": "area", "nuage de points": "scatter",
    "bulles": "bubble", "camembert": "pie", "anneau": "donut", "carte thermique": "heatmap",
    "arborescente": "treemap", "cascade": "waterfall", "entonnoir": "funnel",
}

# ----------------------------
# Parsing helpers
# ----------------------------
def normalize_text(t: str) -> str:
    """Lowercases, maps common FR phrases to EN keywords, strips noise."""
    t = t.lower().strip()

    # Map a few common FR phrases
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
        "par ": "by ",   # normalize "par" → "by" (safe in most contexts)
        "région": "region",
        "bénéfice": "profit",
        "ventes": "revenue",  # so later it maps to Sales
    }
    for fr, en in fr2en.items():
        t = t.replace(fr, en)

    # unify spacing, remove parenthetical hints "(metric)"
    t = re.sub(r"\([^)]*\)", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def resolve_col(freetext: str, cols: List[str]) -> Optional[str]:
    """Map free text to a real column using synonyms + loose matching."""
    if not freetext:
        return None
    name = freetext.strip().lower()
    name = COL_SYNONYMS.get(name, name)
    # exact
    for c in cols:
        if str(c).lower() == name:
            return c
    # startswith fallback
    for c in cols:
        if str(c).lower().startswith(name):
            return c
    # contains fallback (risky but helpful)
    for c in cols:
        if name in str(c).lower():
            return c
    return None

def infer_chart_type(t: str) -> str:
    for key, val in CHART_KEYWORDS.items():
        if key in t:
            return val
    # defaults
    if "trend" in t or "over" in t:
        return "line"
    return "bar"

def parse_group_dims(t: str, cols: List[str]) -> List[str]:
    """
    Capture one or multiple dimensions:
      - 'grouped by Region'
      - 'by Region and Month'
      - 'by Region, Product'
    """
    m = re.search(r"(?:group(?:ed)? by|by)\s+([a-z0-9_ ,>-]+)", t)
    if not m:
        return []
    raw = m.group(1)
    parts = [p.strip() for p in re.split(r",|and|>|→|->", raw) if p.strip()]
    dims = []
    for p in parts:
        c = resolve_col(p, cols)
        if c and c not in dims:
            dims.append(c)
    return dims

def parse_metric_and_agg(t: str, df: pd.DataFrame) -> Tuple[str, Optional[str]]:
    """
    Returns (agg_func, metric_col).
    Handles 'sum of Sales', 'total revenue', 'count of description', 'total description'.
    If metric text column with sum/mean → switch to count.
    If nothing explicit → (sum, first numeric) or (count, first col).
    """
    cols = df.columns.tolist()
    dtypes = df.dtypes.to_dict()

    # sum/avg/etc of <metric>
    m = re.search(r"(sum|avg|average|mean|median|max|min|count|total)\s+(?:of\s+)?([a-z0-9_ -]+)", t)
    if m:
        agg_raw, metric_raw = m.groups()
        agg = AGG_SYNONYMS.get(agg_raw, agg_raw)
        metric_col = resolve_col(metric_raw, cols)
        # if text + numeric agg → count
        if metric_col and not pd.api.types.is_numeric_dtype(dtypes[metric_col]) and agg in ["sum", "mean", "median", "max", "min"]:
            agg = "count"
        return agg, metric_col

    # 'total description' form without "of"
    m2 = re.search(r"(total|count)\s+([a-z0-9_ -]+)", t)
    if m2:
        agg = AGG_SYNONYMS.get(m2.group(1), "sum")
        metric_col = resolve_col(m2.group(2), cols)
        if metric_col and not pd.api.types.is_numeric_dtype(dtypes[metric_col]) and agg != "count":
            agg = "count"
        return agg, metric_col

    # default
    num_like = [c for c in cols if pd.api.types.is_numeric_dtype(dtypes[c])]
    if num_like:
        return "sum", num_like[0]
    return "count", (cols[0] if cols else None)

def parse_color_size(t: str, cols: List[str]) -> Tuple[Optional[str], Optional[str]]:
    color = size = None
    m = re.search(r"(?:color by|group by|colour by)\s+([a-z0-9_ -]+)", t)
    if m:
        color = resolve_col(m.group(1), cols)
    m2 = re.search(r"size by\s+([a-z0-9_ -]+)", t)
    if m2:
        size = resolve_col(m2.group(1), cols)
    return color, size

def parse_sort(t: str, df: pd.DataFrame) -> Tuple[Optional[str], List[str]]:
    """
    Returns (order: 'ascending'/'descending', [cols])
    Examples:
      - 'sort descending by revenue and profit'
      - 'sort asc by Sales'
    """
    order = None
    cols_order = []
    m = re.search(r"sort\s+(ascending|descending|asc|desc)(?:\s+by)?\s+([a-z0-9_, and]+)", t)
    if m:
        o_raw = m.group(1)
        order = "ascending" if o_raw in ["ascending", "asc"] else "descending"
        raw_cols = [c.strip() for c in re.split(r",|and", m.group(2)) if c.strip()]
        for rc in raw_cols:
            c = resolve_col(rc, df.columns.tolist())
            if c and c not in cols_order:
                cols_order.append(c)
    return order, cols_order

def parse_rolling(t: str) -> Optional[int]:
    m = re.search(r"(rolling average|moving average)\s+(\d+)", t)
    if m:
        return int(m.group(2))
    return None

def parse_title(t: str) -> Optional[str]:
    m = re.search(r"title\s*:\s*(.+)$", t)
    if m:
        return m.group(1).strip()
    return None

# ----------------------------
# Aggregation & prep
# ----------------------------
def aggregate_for_chart(
    df: pd.DataFrame,
    dims: List[str],
    agg_func: str,
    metric_col: Optional[str],
    extra_sort_cols: List[str],
) -> Tuple[pd.DataFrame, List[str], str, List[str]]:
    """
    Returns (work_df, used_dims, measure_col_name, available_sort_columns).
    - Aggregates df by dims.
    - Creates main measure column (sum/mean/... of metric OR count).
    - Also aggregates any extra_sort_cols (sum/count) to allow sorting by them.
    - If multiple dims, also adds a 'Category' column for simple X axis concatenation.
    """
    dtypes = df.dtypes.to_dict()
    used_dims = dims[:] if dims else []

    # If no dim, choose first categorical as default dim for charts that need a category
    if not used_dims:
        cat_like = [c for c in df.columns if not pd.api.types.is_numeric_dtype(dtypes[c])]
        if cat_like:
            used_dims = [cat_like[0]]

    # Build aggregation mapping
    agg_map = {}
    measure_col = None

    if agg_func == "count":
        # count of metric if given, else count of rows
        if used_dims:
            if metric_col and metric_col in df.columns:
                grouped = df.groupby(used_dims, as_index=False)[metric_col].count()
                measure_col = f"count_{metric_col}"
                grouped = grouped.rename(columns={metric_col: measure_col})
            else:
                grouped = df.groupby(used_dims, as_index=False).size().rename(columns={"size": "count_rows"})
                measure_col = "count_rows"
        else:
            # whole table count
            grouped = pd.DataFrame({ "count_rows": [len(df)] })
            measure_col = "count_rows"
    else:
        # numeric aggregate
        if metric_col is None or metric_col not in df.columns:
            # pick first numeric
            num_like = [c for c in df.columns if pd.api.types.is_numeric_dtype(dtypes[c])]
            metric_col = num_like[0] if num_like else None

        if used_dims:
            agg_map[metric_col] = agg_func
            grouped = df.groupby(used_dims, as_index=False).agg(agg_map)
            # Name normalization for clarity
            measure_col = f"{agg_func}_{metric_col}"
            # if pandas gave same name as metric_col, rename explicitly
            if metric_col in grouped.columns:
                grouped = grouped.rename(columns={metric_col: measure_col})
        else:
            value = getattr(df[metric_col], agg_func)()
            grouped = pd.DataFrame({ measure_col if (measure_col:=f"{agg_func}_{metric_col}") else "value": [value] })

    # Aggregate extra sort columns
    for sc in extra_sort_cols:
        if sc not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(dtypes[sc]):
            add = df.groupby(used_dims, as_index=False)[sc].sum() if used_dims else pd.DataFrame({sc: [df[sc].sum()]})
            colname = f"sum_{sc}"
        else:
            add = df.groupby(used_dims, as_index=False)[sc].count() if used_dims else pd.DataFrame({sc: [df[sc].count()]})
            colname = f"count_{sc}"
        # merge
        if used_dims:
            grouped = grouped.merge(add if used_dims else add, on=used_dims, how="left")
        else:
            grouped[colname] = add.iloc[:,0]
        # ensure final column name is our aggregated label
        if sc in grouped.columns:
            grouped = grouped.rename(columns={sc: colname})

    # Create concatenated 'Category' for multi-dim X axis in bar/column
    if used_dims and len(used_dims) > 1:
        grouped["Category"] = grouped[used_dims].astype(str).agg(" / ".join, axis=1)
        cat_axis = "Category"
    elif used_dims:
        cat_axis = used_dims[0]
    else:
        cat_axis = measure_col  # single-card

    # Available sort columns (main measure + aggregated extras)
    sortables = [measure_col] + [c for c in grouped.columns if c.startswith(("sum_", "count_")) and c not in [measure_col]]

    return grouped, used_dims, measure_col, [c for c in sortables if c in grouped.columns]

# ----------------------------
# Chart builders
# ----------------------------
def build_chart(
    chart_type: str,
    df: pd.DataFrame,
    dims: List[str],
    measure_col: str,
    order: Optional[str],
    sort_by_cols: List[str],
    color: Optional[str],
    size: Optional[str],
    title: Optional[str],
):
    # Determine category axis
    if dims and len(dims) > 1 and "Category" in df.columns:
        xcat = "Category"
    elif dims:
        xcat = dims[0]
    else:
        xcat = measure_col

    # Sorting
    if sort_by_cols:
        ascending = True if order == "ascending" else False
        # Use the first valid sort column present in df; if several, sort by all
        valid_sort = [c for c in sort_by_cols if c in df.columns]
        if not valid_sort:
            valid_sort = [measure_col]
        df = df.sort_values(valid_sort, ascending=ascending)
    elif order is not None:
        df = df.sort_values([measure_col], ascending=(order == "ascending"))

    # Order categories to match dataframe order
    ordered_categories = df[xcat].astype(str).tolist() if xcat in df.columns else None
    if ordered_categories:
        seen = set()
        ordered_categories = [c for c in ordered_categories if not (c in seen or seen.add(c))]

    # Build visuals
    if chart_type == "pie":
        if dims:
            cat = xcat
        else:
            st.warning("Pie needs a categorical field. Add 'by <Category>'.")
            return
        if not pd.api.types.is_numeric_dtype(df[measure_col]):
            st.warning("Pie needs a numeric measure. Try 'sum of <metric> by <category>'.")
            return
        chart = alt.Chart(df).mark_arc().encode(
            theta=alt.Theta(field=measure_col, type="quantitative"),
            color=alt.Color(field=cat, type="nominal", sort=ordered_categories),
            tooltip=[cat, measure_col]
        )
        st.altair_chart(chart.properties(width='container', height=420, title=title or f"Pie of {measure_col} by {cat}"), use_container_width=True)

    elif chart_type == "donut":
        if dims:
            cat = xcat
        else:
            st.warning("Donut needs a categorical field. Add 'by <Category>'.")
            return
        chart = alt.Chart(df).mark_arc(innerRadius=60).encode(
            theta=alt.Theta(field=measure_col, type="quantitative"),
            color=alt.Color(field=cat, type="nominal", sort=ordered_categories),
            tooltip=[cat, measure_col]
        )
        st.altair_chart(chart.properties(width='container', height=420, title=title or f"Donut of {measure_col} by {cat}"), use_container_width=True)

    elif chart_type in ["bar"]:  # bar/column
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X(xcat, sort=ordered_categories),
            y=alt.Y(measure_col, type="quantitative"),
            color=(color if color in df.columns else alt.value(None)),
            tooltip=list(df.columns)
        )
        st.altair_chart(chart.properties(width='container', height=420, title=title or f"Bar of {measure_col} by {xcat}"), use_container_width=True)

    elif chart_type == "line":
        chart = alt.Chart(df).mark_line(point=True).encode(
            x=alt.X(xcat, sort=ordered_categories),
            y=alt.Y(measure_col, type="quantitative"),
            color=(color if color in df.columns else alt.value(None)),
            tooltip=list(df.columns)
        )
        st.altair_chart(chart.properties(width='container', height=420, title=title or f"Line of {measure_col} by {xcat}"), use_container_width=True)

    elif chart_type == "area":
        chart = alt.Chart(df).mark_area(opacity=0.7).encode(
            x=alt.X(xcat, sort=ordered_categories),
            y=alt.Y(measure_col, type="quantitative"),
            color=(color if color in df.columns else alt.value(None)),
            tooltip=list(df.columns)
        )
        st.altair_chart(chart.properties(width='container', height=420, title=title or f"Area of {measure_col} by {xcat}"), use_container_width=True)

    elif chart_type in ["scatter", "bubble"]:
        # Need two numeric axes; try to find them
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(num_cols) < 2:
            st.warning("Scatter needs two numeric columns. Try 'X vs Y' in description.")
            return
        xnum = num_cols[0]
        ynum = num_cols[1] if len(num_cols) > 1 else measure_col
        size_enc = size if (size in df.columns and pd.api.types.is_numeric_dtype(df[size])) else None
        chart = alt.Chart(df).mark_point(filled=True).encode(
            x=xnum, y=ynum,
            size=size_enc if size_enc else alt.value(60),
            color=color if (color in df.columns) else alt.value(None),
            tooltip=list(df.columns)
        )
        st.altair_chart(chart.properties(width='container', height=420, title=title or f"Scatter: {ynum} vs {xnum}"), use_container_width=True)

    elif chart_type == "hist":
        # Histogram on the main measure unless metric_col detected separately
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X(alt.Bin(maxbins=30), field=measure_col, type='quantitative'),
            y='count()'
        )
        st.altair_chart(chart.properties(width='container', height=420, title=title or f"Histogram of {measure_col}"), use_container_width=True)

    elif chart_type == "box":
        base = df.copy()
        if dims:
            x_field = xcat
        else:
            # If no dim, create a constant to show distribution
            base["All"] = "All"
            x_field = "All"
        chart = alt.Chart(base).mark_boxplot().encode(
            x=x_field,
            y=measure_col,
            color=(color if color in df.columns else alt.value(None)),
            tooltip=list(base.columns)
        )
        st.altair_chart(chart.properties(width='container', height=420, title=title or f"Box plot of {measure_col}"), use_container_width=True)

    elif chart_type == "heatmap":
        if len(dims) < 2Absolutely, Nouhaila—here’s the **full, drop‑in `app.py`** with the improved natural‑language parser, French/English synonyms (incl. *camembert*), explicit **Generate** button (no auto‑chart), **full data preview** toggle, **multi‑field sorting**, and support for the core Power BI‑style visuals that Altair handles well: **bar/column, line, area, scatter/bubble, histogram, box plot, pie, donut, heatmap**.

> ✅ Your sentence  
> “**Create a bar chart showing total description (metric) grouped by Region. Sort descending by revenue and profit**”  
> now resolves to: **bar chart** with **y = count of Description**, **x = Region**, **sorted descending by Sales (revenue) then Profit**.

---

## How to run

```bash
pip install streamlit altair pandas openpyxl xlrd
streamlit run app.py
