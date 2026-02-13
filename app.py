# app.py
import re
import io
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Describe2Chart", page_icon="📈", layout="wide")
st.title("📈 Describe2Chart – Create a graph from your data + description")

st.markdown("""
Upload your data and describe the chart you want.<br>
**Examples**:
- "Line chart of Sales over Month; color by Region; show markers"
- "Stacked bar of Revenue by Region; sort descending; data labels"
- "Scatter of CO2 vs Distance; size by Weight; color by Route"
""", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
desc = st.text_area("Describe the chart you want", height=120,
                    placeholder="e.g., 'line chart of Sales over Month; color by Region; rolling average 7'")

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

def infer_chart_type(text):
    t = text.lower()
    if "line" in t: return "line"
    if "stacked bar" in t: return "bar_stacked"
    if "bar" in t: return "bar"
    if "area" in t: return "area"
    if "scatter" in t or "bubble" in t: return "scatter"
    if "hist" in t: return "hist"
    if "box" in t: return "box"
    if "pie" in t: return "pie"
    # defaults:
    if "over" in t or "trend" in t: return "line"
    return "bar"

def find_column(candidates, cols):
    # Find first column name that appears in text, fallback to heuristic by dtype
    for c in candidates:
        for col in cols:
            if c.strip().lower() == str(col).lower():
                return col
    return None

def extract_columns(text, cols, dtypes):
    t = text.lower()

    # guess x by keywords
    time_like = [c for c in cols if "date" in str(c).lower() or "time" in str(c).lower() or "month" in str(c).lower() or "year" in str(c).lower()]
    num_like = [c for c in cols if pd.api.types.is_numeric_dtype(dtypes[c])]
    cat_like = [c for c in cols if not pd.api.types.is_numeric_dtype(dtypes[c])]

    # parse "of Y over X", "Y vs X"
    m_over = re.search(r"of\s+(.+?)\s+over\s+([a-zA-Z0-9_ -]+)", t)
    m_vs   = re.search(r"(.+?)\s+vs\s+([a-zA-Z0-9_ -]+)", t)

    y_guess = None
    x_guess = None

    if m_over:
        y_guess = find_column([m_over.group(1)], cols)
        x_guess = find_column([m_over.group(2)], cols)
    elif m_vs:
        y_guess = find_column([m_vs.group(1)], cols)
        x_guess = find_column([m_vs.group(2)], cols)

    # Fallbacks
    if x_guess is None:
        x_guess = time_like[0] if len(time_like) else (cat_like[0] if len(cat_like) else cols[0])
    if y_guess is None:
        y_guess = num_like[0] if len(num_like) else (cols[1] if len(cols) > 1 else cols[0])

    # color/size/row/column
    color = None; size = None
    for key, var in [("color by", "color"), ("colour by", "color"), ("group by", "color")]:
        m = re.search(fr"{key}\s+([a-zA-Z0-9_ -]+)", t)
        if m:
            cand = find_column([m.group(1)], cols)
            if cand: color = cand; break

    m = re.search(r"size by\s+([a-zA-Z0-9_ -]+)", t)
    if m:
        cand = find_column([m.group(1)], cols)
        if cand: size = cand

    return x_guess, y_guess, color, size

def parse_options(text):
    t = text.lower()
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
    m_roll = re.search(r"rolling average\s+(\d+)|moving average\s+(\d+)", t)
    if m_roll:
        opts["rolling"] = int([g for g in m_roll.groups() if g][0])

    m_bins = re.search(r"bins?\s+(\d+)", t)
    if m_bins:
        opts["bins"] = int(m_bins.group(1))

    m_agg = re.search(r"(sum|avg|mean|median|max|min)\s+of\s+([a-zA-Z0-9_ -]+)", t)
    if m_agg:
        opts["agg"] = (m_agg.group(1), m_agg.group(2))

    m_title = re.search(r"title\s*:\s*(.+)$", t)
    if m_title:
        opts["title"] = m_title.group(1).strip()

    return opts

if uploaded:
    try:
        df = load_df(uploaded)
        st.success(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
        st.dataframe(df.head())

        if desc.strip():
            chart_type = infer_chart_type(desc)
            x, y, color, size = extract_columns(desc, df.columns.tolist(), df.dtypes.to_dict())
            opts = parse_options(desc)

            work = df.copy()

            # optional aggregation
            if opts["agg"]:
                func, col = opts["agg"]
                col_match = next((c for c in df.columns if c.lower() == col.lower()), None)
                if col_match:
                    agg_map = {"avg":"mean", "mean":"mean", "sum":"sum", "median":"median", "max":"max", "min":"min"}
                    f = agg_map.get(func, func)
                    if x in work.columns and col_match in work.columns:
                        work = work.groupby(x, as_index=False)[col_match].agg(f)
                        y = col_match

            # optional rolling average
            if opts["rolling"] and pd.api.types.is_numeric_dtype(work[y]):
                work = work.sort_values(by=x)
                work[f"{y}_rolling_{opts['rolling']}"] = work[y].rolling(opts["rolling"], min_periods=1).mean()
                y = f"{y}_rolling_{opts['rolling']}"

            # Build chart
            enc = {}
            if chart_type in ["bar", "bar_stacked", "area"]:
                enc["x"] = alt.X(x, sort="-y" if opts["sort_desc"] else "y" if opts["sort_asc"] else None)
            else:
                enc["x"] = alt.X(x)

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
                mark = alt.MarkDef(type="bar")
                chart = alt.Chart(work).mark_bar().encode(
                    x=x, y=y,
                    color=color if color else alt.value("#4C78A8"),
                    tooltip=list(work.columns),
                    order=y
                )
                if chart_type == "bar_stacked" or opts["stacked"]:
                    chart = chart.encode(x=x, y=y, color=color).transform_aggregate(total=f"sum({y})", groupby=[x, color] if color else [x])
            elif chart_type == "area":
                chart = alt.Chart(work).mark_area().encode(
                    x=x, y=y,
                    color=color if color else alt.value(None),
                    tooltip=list(work.columns)
                )
            elif chart_type == "pie":
                # requires categorical + numeric
                cat = color or x
                val = y
                if not pd.api.types.is_numeric_dtype(work[val]):
                    st.warning("Pie chart needs a numeric value column for sizes. Try 'sum of <Value> by <Category>'.")
                else:
                    chart = alt.Chart(work).mark_arc().encode(
                        theta=alt.Theta(field=val, type="quantitative"),
                        color=alt.Color(field=cat, type="nominal"),
                        tooltip=[cat, val]
                    )
            else:
                chart = alt.Chart(work).mark_bar().encode(x=x, y=y)

            if opts["labels"] and chart_type not in ["pie", "hist", "box"]:
                text = chart.mark_text(align='center', baseline='bottom', dy=-5).encode(text=y)
                chart = chart + text

            if opts["theme"] == "dark":
                st.checkbox("Dark theme requested in description", value=True, disabled=True)
                st.write("Use Streamlit settings or a custom theme for full dark mode.")
            title = opts["title"] or f"{chart_type.capitalize()} of {y} by {x}"
            st.subheader("Chart")
            st.altair_chart(chart.properties(width='container', height=420, title=title), use_container_width=True)

            st.download_button(
                "Download chart data as CSV",
                data=work.to_csv(index=False).encode("utf-8"),
                file_name="chart_data.csv",
                mime="text/csv"
            )
        else:
            st.info("Describe the chart you want in the text area above.")
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload a CSV or Excel file to get started.")
