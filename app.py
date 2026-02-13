# app.py
import re
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Describe2Chart", page_icon="📈", layout="wide")
st.title("📈 Describe2Chart – Create a graph from your data + description")

st.markdown("""
Upload your data and **describe** the chart you want.  
Then click **Generate chart**.

**Examples (EN):**
- "Line chart of Sales over Date; color by Region; rolling average 7"
- "Stacked bar of Revenue by Region; sort descending; data labels"
- "Pie of sum of Sales by Region; title: Sales by Region"

**Exemples (FR):**
- "Je veux un **camembert** des **ventes** par **région**"
- "Courbe des ventes sur la **date**; moyenne mobile 7; **couleur par** région"
- "Histogramme du **profit**; **bins 20**; **titre: Distribution du profit**"
""")

# ----------------------------
# 1) File upload
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
# Helpers
# ----------------------------
def normalize_text(t: str) -> str:
    t = t.lower().strip()
    # map a few French keywords to English equivalents used internally
    fr2en = {
        "camembert": "pie",
        "diagramme en secteurs": "pie",
        "secteurs": "pie",
        "nuage de points": "scatter",
        "courbe": "line",
        "aire": "area",
        "barres": "bar",
        "empilé": "stacked",
        "couleur par": "color by",
        "couleur": "color",
        "moyenne mobile": "rolling average",
        "tri descendant": "sort desc",
        "tri ascendant": "sort asc",
        "titre": "title",
        "somme de": "sum of",
        "moyenne de": "avg of"
    }
    for fr, en in fr2en.items():
        t = t.replace(fr, en)
    # fix some punctuation spacing
    t = re.sub(r"\s*:\s*", ": ", t)
    return t

def infer_chart_type(text):
    t = text
    if "stacked bar" in t or ("bar" in t and "stacked" in t): return "bar_stacked"
    if "line" in t: return "line"
    if "scatter" in t or "bubble" in t: return "scatter"
    if "area" in t: return "area"
    if "hist" in t or "histogram" in t: return "hist"
    if "box" in t or "boxplot" in t: return "box"
    if "pie" in t: return "pie"
    # defaults:
    if "over" in t or "trend" in t: return "line"
    return "bar"

def _find_exact_col(name, cols):
    for col in cols:
        if str(col).lower().strip() == name.lower().strip():
            return col
    return None

def guess_columns_from_text(t, cols, dtypes):
    # Parse patterns like:
    # - "of Y over X"
    # - "Y vs X"
    # - "sum of Sales by Region"
    m_over = re.search(r"of\s+(.+?)\s+over\s+([a-z0-9_ -]+)", t)
    m_vs   = re.search(r"(.+?)\s+vs\s+([a-z0-9_ -]+)", t)
    m_by   = re.search(r"(sum|avg|mean|median|max|min)\s+of\s+([a-z0-9_ -]+)\s+by\s+([a-z0-9_ -]+)", t)

    x = y = color = size = None

    if m_by:
        func, val, cat = m_by.groups()
        y = _find_exact_col(val, cols)
        x = _find_exact_col(cat, cols)

    if m_over and (x is None or y is None):
        y = y or _find_exact_col(m_over.group(1), cols)
        x = x or _find_exact_col(m_over.group(2), cols)

    if m_vs and (x is None or y is None):
        y = y or _find_exact_col(m_vs.group(1), cols)
        x = x or _find_exact_col(m_vs.group(2), cols)

    # color by / group by / size by
    for key in ["color by", "group by", "colour by"]:
        m = re.search(fr"{key}\s+([a-z0-9_ -]+)", t)
        if m and color is None:
            cand = _find_exact_col(m.group(1), cols)
            if cand: color = cand

    m = re.search(r"size by\s+([a-z0-9_ -]+)", t)
    if m:
        cand = _find_exact_col(m.group(1), cols)
        if cand: size = cand

    # fallbacks based on types
    num_like = [c for c in cols if pd.api.types.is_numeric_dtype(dtypes[c])]
    cat_like = [c for c in cols if not pd.api.types.is_numeric_dtype(dtypes[c])]
    time_like = [c for c in cols if any(k in str(c).lower() for k in ["date", "time", "month", "year"])]

    if x is None:
        x = (time_like[0] if time_like else (cat_like[0] if cat_like else cols[0]))
    if y is None:
        y = (num_like[0] if num_like else (cols[1] if len(cols) > 1 else cols[0]))
    if color is None and len(cat_like) > 0:
        # color is optional; leave None unless specified
        pass
    return x, y, color, size

def parse_options(text):
    t = text
    opts = {
        "stacked": "stack" in t,
        "markers": "marker" in t or "dot" in t,
        "labels": "label" in t,
        "sort_desc": "sort desc" in t or "descending" in t,
        "sort_asc": "sort asc" in t or "ascending" in t,
        "rolling": None,
        "agg": None,
        "bins": None,
        "title": None,
        "theme": "dark" if "dark" in t else "light" if "light theme" in t else None
    }
    m_roll = re.search(r"(rolling average|moving average)\s+(\d+)", t)
    if m_roll:
        opts["rolling"] = int(m_roll.group(2))

    m_bins = re.search(r"bins?\s+(\d+)", t)
    if m_bins:
        opts["bins"] = int(m_bins.group(1))

    m_agg = re.search(r"(sum|avg|mean|median|max|min)\s+of\s+([a-z0-9_ -]+)", t)
    if m_agg:
        opts["agg"] = (m_agg.group(1), m_agg.group(2))

    m_title = re.search(r"title\s*:\s*(.+)$", t)
    if m_title:
        opts["title"] = m_title.group(1).strip()
    return opts

def choose_default_for_pie(df, cols, dtypes):
    cat_like = [c for c in cols if not pd.api.types.is_numeric_dtype(dtypes[c])]
    num_like = [c for c in cols if pd.api.types.is_numeric_dtype(dtypes[c])]
    # Prefer "Region" + first numeric (e.g., Sales)
    cat = "Region" if "Region" in cols else (cat_like[0] if cat_like else None)
    val = "Sales" if "Sales" in cols else (num_like[0] if num_like else None)
    return cat, val

# ----------------------------
# 2) UI – description + controls
# ----------------------------
desc = st.text_area(
    "Describe the chart you want (EN/FR supported)",
    height=120,
    placeholder="e.g., 'Je veux un camembert des ventes par région' or 'Line chart of Sales over Date; color by Region'"
)

# Prevent auto-render: require explicit click
generate = st.button("Generate chart", type="primary", use_container_width=False)

# ----------------------------
# 3) Data preview section
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
# 4) Render chart only on click
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
            x, y, color, size = guess_columns_from_text(t, df.columns.tolist(), df.dtypes.to_dict())
            opts = parse_options(t)

            work = df.copy()

            # For pie charts: if not explicit, choose defaults
            if chart_type == "pie":
                # If the user's description did not provide a clear category/value
                if not pd.api.types.is_numeric_dtype(work[y]) or pd.api.types.is_numeric_dtype(work[x]):
                    cat, val = choose_default_for_pie(work, work.columns.tolist(), work.dtypes.to_dict())
                    if cat and val:
                        x, y = cat, val

            # Optional aggregation from description (e.g., "sum of Sales by Region")
            if opts["agg"] and x in work.columns:
                func, col = opts["agg"]
                col_match = next((c for c in work.columns if c.lower() == col.lower()), None)
                if col_match:
                    agg_map = {"avg":"mean", "mean":"mean", "sum":"sum", "median":"median", "max":"max", "min":"min"}
                    f = agg_map.get(func, func)
                    work = work.groupby(x, as_index=False)[col_match].agg(f)
                    y = col_match

            # Optional rolling average (for numeric y)
            if opts["rolling"] and pd.api.types.is_numeric_dtype(work[y]):
                if pd.api.types.is_datetime64_any_dtype(work[x]) or "date" in str(x).lower():
                    work = work.sort_values(by=x)
                work[f"{y}_rolling_{opts['rolling']}"] = work[y].rolling(opts["rolling"], min_periods=1).mean()
                y = f"{y}_rolling_{opts['rolling']}"

            # Build chart with Altair
            if chart_type == "hist":
                bins = opts["bins"] or 30
                chart = alt.Chart(work).mark_bar().encode(
                    x=alt.X(alt.Bin(maxbins=bins), field=y, type='quantitative'),
                    y='count()',
                    color=color if color else alt.value(None),
                    tooltip=list(work.columns)
                )
            elif chart_type == "box":
                chart = alt.Chart(work).mark_boxplot().encode(
                    x=color if color else x,
                    y=y,
                    color=color if color else alt.value(None),
                    tooltip=list(work.columns)
                )
            elif chart_type == "scatter":
                chart = alt.Chart(work).mark_point(filled=True).encode(
                    x=x, y=y,
                    color=color if color else alt.value(None),
                    size=size if size else alt.value(60),
                    tooltip=list(work.columns)
                )
            elif chart_type == "line":
                chart = alt.Chart(work).mark_line(point=opts["markers"]).encode(
                    x=x, y=y,
                    color=color if color else alt.value(None),
                    tooltip=list(work.columns)
                )
            elif chart_type in ["bar", "bar_stacked"]:
                enc = {
                    "x": alt.X(x, sort="-y" if opts["sort_desc"] else ("y" if opts["sort_asc"] else None)),
                    "y": y,
                    "tooltip": list(work.columns)
                }
                if color:
                    enc["color"] = color
                chart = alt.Chart(work).mark_bar().encode(**enc)
                if chart_type == "bar_stacked" or opts["stacked"]:
                    # Altair stacks by default if color provided; keep as-is
                    pass
            elif chart_type == "area":
                chart = alt.Chart(work).mark_area().encode(
                    x=x, y=y,
                    color=color if color else alt.value(None),
                    tooltip=list(work.columns)
                )
            elif chart_type == "pie":
                cat = color or x  # category
                val = y           # value
                if not pd.api.types.is_numeric_dtype(work[val]):
                    st.warning("Pie chart needs a numeric value column for sizes. Try 'sum of <Value> by <Category>'.")
                    chart = None
                else:
                    chart = alt.Chart(work).mark_arc().encode(
                        theta=alt.Theta(field=val, type="quantitative"),
                        color=alt.Color(field=cat, type="nominal"),
                        tooltip=[cat, val]
                    )
            else:
                chart = alt.Chart(work).mark_bar().encode(x=x, y=y)

            if chart is not None:
                title = opts["title"] or f"{chart_type.capitalize()} of {y} by {x}"
                st.subheader("Chart")
                st.altair_chart(chart.properties(width='container', height=420, title=title), use_container_width=True)
            else:
                st.info("Adjust your description and click Generate again.")

            st.download_button(
                "Download chart data as CSV",
                data=work.to_csv(index=False).encode("utf-8"),
                file_name="chart_data.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Could not build the chart: {e}")
