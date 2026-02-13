# app.py
import re
import pandas as pd
import streamlit as st
import altair as alt

# ----------------------------
# Streamlit page setup
# ----------------------------
st.set_page_config(page_title="Describe2Chart", page_icon="📈", layout="wide")
st.title("📈 Describe2Chart – Create a graph from your data + description")

st.markdown("""
Upload your dataset, **describe** the chart you want (EN/FR supported), then click **Generate chart**.

**Examples (EN)**
- "Bar chart of **count of Description** grouped by **Region**; **sort descending by revenue and profit**"
- "Line chart of **Sales over Date**; rolling average 7; **color by Region**"
- "Pie of **sum of Sales by Region**; title: Sales share"
- "Heatmap of **sum of Profit by Region and Description**"

**Exemples (FR)**
- "Je veux un **camembert** des **ventes** par **région**; **titre: Répartition des ventes**"
- "Courbe des **ventes** sur **Date**; **moyenne mobile 7**; **couleur par Région**"
- "Barres du **nombre de Description** par **Région**; **tri descendant par bénéfice**"
""")

# ----------------------------
# File upload
# ----------------------------
uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

@st.cache_data
def load_df(file):
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
# Synonyms and parsing helpers
# ----------------------------

# Column synonyms (EN/FR) → map to your real column names
# Adjust these to your schema as needed.
COL_SYNONYMS = {
    "revenue": "Sales", "revenues": "Sales", "amount": "Sales",
    "chiffre d'affaires": "Sales", "ca": "Sales", "vente": "Sales", "ventes": "Sales",
    "profit": "Profit", "benefice": "Profit", "bénéfice": "Profit", "marge": "Profit",
    "region": "Region", "région": "Region",
    "date": "Date",
    "description": "Description",
}

# Aggregation synonyms
AGG_SYNONYMS = {
    "total": "sum", "sum": "sum", "somme": "sum",
    "avg": "mean", "average": "mean", "mean": "mean", "moyenne": "mean",
    "median": "median", "mediane": "median",
    "max": "max", "min": "min",
    "count": "count", "nombre": "count", "compte": "count", "nb": "count"
}

def resolve_col(name: str, cols):
    """Map a free-text name to a real column using synonyms + exact/startswith matches."""
    if name is None: return None
    key = name.strip().lower()
    key = COL_SYNONYMS.get(key, key)  # apply synonym if defined
    for c in cols:
        if str(c).lower() == key:
            return c
    for c in cols:  # startswith fallback
        if str(c).lower().startswith(key):
            return c
    return None

def normalize_text(t: str) -> str:
    """Lowercase, normalize FR→EN chart keywords, clean up punctuation and parentheticals."""
    t = t.lower().strip()
    fr2en = {
        "camembert": "pie",
        "diagramme en secteurs": "pie",
        "secteurs": "pie",
        "nuage de points": "scatter",
        "courbe": "line",
        "aire": "area",
        "barres": "bar",
        "colonne": "column",
        "empilé": "stacked",
        "couleur par": "color by",
        "couleur": "color",
        "moyenne mobile": "rolling average",
        "tri descendant": "sort desc",
        "tri ascendant": "sort asc",
        "titre": "title",
        "somme de": "sum of",
        "moyenne de": "avg of",
        "anneau": "donut",  # donut chart
        "carte thermique": "heatmap"
    }
    for fr, en in fr2en.items():
        t = t.replace(fr, en)
    t = re.sub(r"\([^)]*\)", "", t)     # remove content in parentheses
    t = re.sub(r"\s*:\s*", ": ", t)     # normalize colons
    t = re.sub(r"\s+", " ", t).strip()  # collapse spaces
    return t

def infer_chart_type(text):
    t = text
    if "donut" in t: return "donut"
    if "pie" in t: return "pie"
    if "heatmap" in t: return "heatmap"
    if "stacked bar" in t or ("bar" in t and "stacked" in t): return "bar_stacked"
    if "column" in t or "bar" in t: return "bar"
    if "line" in t: return "line"
    if "scatter" in t or "bubble" in t: return "scatter"
    if "area" in t: return "area"
    if "hist" in t or "histogram" in t: return "hist"
    if "box" in t or "boxplot" in t: return "box"
    # Default to bar
    return "bar"

def parse_dimensions(text, cols):
    """
    Parse up to two grouping dimensions.
    Accepts: "grouped by X", "by X", "by X and Y", "by X, Y"
    Returns list: [dim1] or [dim1, dim2]
    """
    dims = []
    m = re.search(r"(?:group(?:ed)? by|by)\s+([a-z0-9_ ,&-]+)", text)
    if m:
        raw = m.group(1)
        # split on ",", "and", "&"
        parts = [p.strip() for p in re.split(r",| and | & ", raw) if p.strip()]
        for p in parts:
            col = resolve_col(p, cols)
            if col and col not in dims:
                dims.append(col)
    return dims[:2]

def parse_metric_and_agg(text, df):
    """
    Returns (agg_func, metric_col)
    - If metric is non-numeric and agg is sum/avg/etc. → switch to count
    - If not specified → pick first numeric col and 'sum'; else count of first col
    """
    cols = df.columns.tolist()
    dtypes = df.dtypes.to_dict()

    # Patterns: "sum of Sales", "total sales", "count of description"
    m = re.search(r"(sum|avg|average|mean|median|max|min|count|total)\s+(?:of\s+)?([a-z0-9_ -]+)", text)
    if m:
        agg_raw, metric_raw = m.groups()
        agg = AGG_SYNONYMS.get(agg_raw, agg_raw)
        metric_col = resolve_col(metric_raw, cols)
        if metric_col:
            if not pd.api.types.is_numeric_dtype(dtypes[metric_col]) and agg in ["sum", "mean", "median", "max", "min"]:
                agg = "count"
        return agg, metric_col

    # "total description" form
    m2 = re.search(r"(total|count)\s+([a-z0-9_ -]+)", text)
    if m2:
        agg = AGG_SYNONYMS.get(m2.group(1), "sum")
        metric_col = resolve_col(m2.group(2), cols)
        if metric_col and not pd.api.types.is_numeric_dtype(dtypes[metric_col]) and agg != "count":
            agg = "count"
        return agg, metric_col

    # Fallback
    num_like = [c for c in cols if pd.api.types.is_numeric_dtype(dtypes[c])]
    if num_like:
        return "sum", num_like[0]
    return "count", cols[0] if cols else None

def parse_sort(text, df):
    """
    Returns (order: 'ascending'/'descending'/None, [list_of_columns_to_sort_by])
    Supports: "sort descending by revenue and profit", "sort asc by Sales"
    """
    order = None
    cols_order = []
    m = re.search(r"sort\s+(ascending|descending|asc|desc)(?:\s+by)?\s+([a-z0-9_ ,&and]+)", text)
    if m:
        o_raw = m.group(1)
        order = "ascending" if o_raw in ["ascending", "asc"] else "descending"
        raw_cols = [c.strip() for c in re.split(r",|and|&", m.group(2)) if c.strip()]
        for rc in raw_cols:
            c = resolve_col(rc, df.columns.tolist())
            if c: cols_order.append(c)
    return order, cols_order

def agg_name(func: str, col: str):
    if func == "count":
        return f"count_{col}" if col else "count"
    return f"{func}_{col}"

def build_aggregated(df, dims, agg_func, metric_col, extra_sort_cols):
    """
    Build an aggregated frame based on 0–2 dimensions, a main metric (agg_func(metric_col)),
    and any extra columns to aggregate for sorting (e.g., Sales, Profit).
    Returns (work_df, dim1, dim2, y_col_name)
    """
    cols = df.columns.tolist()
    dtypes = df.dtypes.to_dict()

    # Choose dimensions if missing
    if not dims:
        # Prefer non-numeric for grouping
        cat_like = [c for c in cols if not pd.api.types.is_numeric_dtype(dtypes[c])]
        dims = [cat_like[0]] if cat_like else [cols[0]]

    # Ensure we have at most 2 dims
    dims = dims[:2]
    dim1 = dims[0]
    dim2 = dims[1] if len(dims) > 1 else None

    group_keys = [dim1] + ([dim2] if dim2 else [])

    # Determine main measure column name
    if agg_func == "count":
        y_col = agg_name("count", metric_col if metric_col else dim1)
    else:
        # If metric_col None → pick first numeric
        if metric_col is None:
            num_like = [c for c in cols if pd.api.types.is_numeric_dtype(dtypes[c])]
            metric_col = num_like[0] if num_like else dim1
        y_col = agg_name(agg_func, metric_col)

    # Build aggregation dict
    agg_dict = {}

    if agg_func == "count":
        # Count of metric (if provided) else of first dimension
        if metric_col:
            agg_dict[metric_col] = "count"
        else:
            # We'll use size() for count of rows per group later
            pass
    else:
        agg_dict[metric_col] = agg_func

    # Extra sort columns (aggregate them too so we can sort by them)
    for sc in extra_sort_cols:
        if sc not in agg_dict:
            if pd.api.types.is_numeric_dtype(dtypes[sc]):
                agg_dict[sc] = "sum"
            else:
                agg_dict[sc] = "count"

    # Perform groupby aggregation
    if agg_func == "count" and not metric_col:
        work = df.groupby(group_keys, as_index=False).size().rename(columns={"size": y_col})
    else:
        work = df.groupby(group_keys, as_index=False).agg(agg_dict)
        # standardize main measure name
        if agg_func != "count" and metric_col in work.columns:
            if metric_col != y_col:
                work = work.rename(columns={metric_col: y_col})

        # Ensure extra sort columns get standard names (sum_Sales etc.)
        for sc in extra_sort_cols:
            target_name = agg_name("sum", sc) if pd.api.types.is_numeric_dtype(dtypes[sc]) else agg_name("count", sc)
            if sc in work.columns and sc != target_name:
                # Only rename if not already in target form
                # Avoid collisions
                if target_name in work.columns:
                    continue
                work = work.rename(columns={sc: target_name})

    return work, dim1, dim2, y_col

def choose_default_for_pie(df):
    """Pick a default (category, value) for pie/donut when user didn't specify a metric."""
    cols = df.columns.tolist()
    dtypes = df.dtypes.to_dict()
    cat_like = [c for c in cols if not pd.api.types.is_numeric_dtype(dtypes[c])]
    num_like = [c for c in cols if pd.api.types.is_numeric_dtype(dtypes[c])]
    cat = "Region" if "Region" in cols else (cat_like[0] if cat_like else cols[0])
    val = "Sales" if "Sales" in cols else (num_like[0] if num_like else None)
    return cat, val

# ----------------------------
# UI inputs
# ----------------------------
desc = st.text_area(
    "Describe the chart you want (EN/FR supported)",
    height=120,
    placeholder="e.g., 'Bar chart of count of Description by Region; sort descending by revenue and profit'"
)
generate = st.button("Generate chart", type="primary")

# ----------------------------
# Data preview
# ----------------------------
if uploaded:
    try:
        df = load_df(uploaded)
        st.success(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
        show_head_only = st.toggle("Preview only first 5 rows", value=False)
        st.dataframe(df.head() if show_head_only else df, use_container_width=True, height=300)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        df = None
else:
    df = None

# ----------------------------
# Chart render (only on click)
# ----------------------------
if generate:
    if df is None:
        st.warning("Please upload a dataset first.")
    elif not desc or not desc.strip():
        st.info("Please describe your chart above, then click Generate.")
    else:
        try:
            t = normalize_text(desc)
            chart_type = infer_chart_type(t)

            # 1) Dimensions (up to 2)
            dims = parse_dimensions(t, df.columns.tolist())

            # 2) Metric + aggregation
            agg_func, metric_col = parse_metric_and_agg(t, df)

            # 3) Sorting
            sort_order, sort_cols = parse_sort(t, df)  # columns are raw; will be mapped in aggregation step

            # 4) Build aggregated dataset (also aggregates sort columns)
            work, dim1, dim2, y_col = build_aggregated(df, dims, agg_func, metric_col, sort_cols)

            # 5) Resolve sort column names after aggregation (they may be renamed to sum_/count_)
            effective_sort_cols = []
            for sc in sort_cols:
                candidates = [agg_name("sum", sc), agg_name("count", sc), agg_name("mean", sc), agg_name("median", sc),
                              agg_name("max", sc), agg_name("min", sc), sc]
                chosen = next((c for c in candidates if c in work.columns), None)
                if chosen and chosen not in effective_sort_cols:
                    effective_sort_cols.append(chosen)

            # Default sort: by main measure
            if not effective_sort_cols:
                effective_sort_cols = [y_col]

            if sort_order is None:
                # fallback: descending for measures
                sort_order = "descending"

            ascending_flags = [sort_order == "ascending"] * len(effective_sort_cols)
            work = work.sort_values(by=effective_sort_cols, ascending=ascending_flags)

            # Keep category order from sorted data
            ordered_dim1 = work[dim1].astype(str).tolist()
            seen = set()
            ordered_dim1 = [c for c in ordered_dim1 if not (c in seen or seen.add(c))]

            # 6) Build chart per type
            chart = None

            if chart_type in ["pie", "donut"]:
                # Need (category, value)
                cat = dim1
                val = None

                # If main measure is numeric, use y_col
                if y_col in work.columns and pd.api.types.is_numeric_dtype(work[y_col]):
                    val = y_col
                else:
                    # Choose default numeric
                    cat_def, val_def = choose_default_for_pie(df)
                    cat = cat or cat_def
                    val = val or val_def

                if val is None or not pd.api.types.is_numeric_dtype(work[val]):
                    st.warning("Pie/Donut needs a numeric measure (e.g., 'sum of Sales by Region').")
                else:
                    base = alt.Chart(work).encode(
                        theta=alt.Theta(field=val, type="quantitative"),
                        color=alt.Color(field=cat, type="nominal", sort=ordered_dim1),
                        tooltip=[cat, val]
                    )
                    mark = alt.MarkDef(type="arc", innerRadius=80) if chart_type == "donut" else alt.MarkDef(type="arc")
                    chart = base.mark_arc(innerRadius=80) if chart_type == "donut" else base.mark_arc()

            elif chart_type == "heatmap":
                # Requires two dimensions; if we only have one, try to infer a second
                if dim2 is None:
                    # Guess another categorical column different from dim1
                    dtypes = df.dtypes.to_dict()
                    cat_like = [c for c in df.columns if (not pd.api.types.is_numeric_dtype(dtypes[c])) and c != dim1]
                    if cat_like:
                        dim2 = cat_like[0]
                        # Re-aggregate with second dim
                        work, dim1, dim2, y_col = build_aggregated(df, [dim1, dim2], agg_func, metric_col, sort_cols)
                    else:
                        st.warning("Heatmap needs two categorical dimensions (e.g., 'by Region and Description').")
                if dim2:
                    chart = alt.Chart(work).mark_rect().encode(
                        x=alt.X(dim1, type="nominal", sort=ordered_dim1),
                        y=alt.Y(dim2, type="nominal"),
                        color=alt.Color(y_col, type="quantitative"),
                        tooltip=[dim1, dim2, y_col]
                    )

            elif chart_type == "hist":
                # Histogram uses the raw df and chosen metric (or first numeric)
                metric_for_hist = metric_col
                if metric_for_hist is None or not pd.api.types.is_numeric_dtype(df[metric_for_hist]):
                    # pick first numeric
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

            elif chart_type == "box":
                # Box plot of main metric by dim1
                if not pd.api.types.is_numeric_dtype(work[y_col]):
                    st.warning("Box plot requires a numeric measure. Try 'sum of Sales by Region'.")
                else:
                    chart = alt.Chart(work).mark_boxplot().encode(
                        x=alt.X(dim1, sort=ordered_dim1),
