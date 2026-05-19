import streamlit as st
import pandas as pd
import altair as alt

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Cap Visual Intelligence", layout="wide")

st.title("Cap Visual Intelligence powered by GenAI")
st.write("Upload your dataset, describe the visual, then click Generate.")


# ---------------------------
# LOAD DATA
# ---------------------------
uploaded = st.file_uploader("Upload CSV file", type=["csv"])

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# ---------------------------
# INTERPRET REQUEST (SMART)
# ---------------------------
def interpret_request(desc, df):
    desc = desc.lower()

    cols = df.columns.tolist()

    spec = {
        "chart_type": "bar",
        "metric": None,
        "aggregation": "sum",
        "dimension": None,
        "sort": []
    }

    # Chart detection
    if "line" in desc:
        spec["chart_type"] = "line"
    elif "pie" in desc or "camembert" in desc:
        spec["chart_type"] = "pie"
    elif "scatter" in desc:
        spec["chart_type"] = "scatter"
    elif "heatmap" in desc:
        spec["chart_type"] = "heatmap"
    else:
        spec["chart_type"] = "bar"

    # Synonyms
    synonyms = {
        "revenue": "Sales",
        "ventes": "Sales",
        "profit": "Profit",
        "bénéfice": "Profit",
        "région": "Region",
        "region": "Region",
        "description": "Description"
    }

    for key, val in synonyms.items():
        if key in desc:
            desc = desc.replace(key, val.lower())

    # Aggregation
    if "count" in desc or "nombre" in desc:
        spec["aggregation"] = "count"
    elif "average" in desc or "mean" in desc:
        spec["aggregation"] = "mean"
    elif "sum" in desc or "total" in desc:
        spec["aggregation"] = "sum"

    # Metric detection
    for col in cols:
        if col.lower() in desc:
            spec["metric"] = col

    # ⚡ Fix: text column → count
    if spec["metric"] and not pd.api.types.is_numeric_dtype(df[spec["metric"]]):
        spec["aggregation"] = "count"

    # Dimension
    for col in cols:
        if col.lower() in desc and col != spec["metric"]:
            spec["dimension"] = col
            break

    # Sorting
    if "sort" in desc or "descending" in desc:
        if "sales" in desc:
            spec["sort"].append(("Sales", False))
        if "profit" in desc:
            spec["sort"].append(("Profit", False))

    return spec


# ---------------------------
# VALIDATION
# ---------------------------
def validate_spec(spec, df):
    errors = []

    if not spec["metric"]:
        errors.append("No metric detected")

    if spec["chart_type"] in ["bar", "line", "pie"] and not spec["dimension"]:
        errors.append("No dimension detected")

    if spec["aggregation"] != "count":
        if not pd.api.types.is_numeric_dtype(df[spec["metric"]]):
            errors.append("Metric must be numeric")

    if spec["chart_type"] == "pie":
        if spec["aggregation"] != "count" and not pd.api.types.is_numeric_dtype(df[spec["metric"]]):
            errors.append("Pie requires numeric metric")

    return errors


# ---------------------------
# PROCESS DATA
# ---------------------------
def process_data(df, spec):
    metric = spec["metric"]
    dim = spec["dimension"]
    agg = spec["aggregation"]

    if agg == "count":
        grouped = df.groupby(dim)[metric].count().reset_index(name=f"count_{metric}")
        y = f"count_{metric}"
    else:
        grouped = df.groupby(dim)[metric].agg(agg).reset_index(name=f"{agg}_{metric}")
        y = f"{agg}_{metric}"

    # Sorting
    for col, asc in spec["sort"]:
        if col in grouped.columns:
            grouped = grouped.sort_values(by=col, ascending=asc)

    return grouped, dim, y


# ---------------------------
# RENDER CHART
# ---------------------------
def render_chart(spec, data, x, y):
    chart_type = spec["chart_type"]

    if chart_type == "bar":
        chart = alt.Chart(data).mark_bar().encode(
            x=alt.X(x, sort='-y'),
            y=y,
            tooltip=[x, y]
        )

    elif chart_type == "line":
        chart = alt.Chart(data).mark_line(point=True).encode(
            x=x,
            y=y,
            tooltip=[x, y]
        )

    elif chart_type == "pie":
        chart = alt.Chart(data).mark_arc().encode(
            theta=y,
            color=x,
            tooltip=[x, y]
        )

    elif chart_type == "scatter":
        chart = alt.Chart(data).mark_point(size=100).encode(
            x=x,
            y=y,
            tooltip=[x, y]
        )

    elif chart_type == "heatmap":
        chart = alt.Chart(data).mark_rect().encode(
            x=x,
            y=y,
            color=y
        )

    else:
        chart = alt.Chart(data).mark_bar().encode(x=x, y=y)

    return chart


# ---------------------------
# UI INPUT
# ---------------------------
if uploaded:
    df = load_data(uploaded)

    st.write("Dataset Preview:")
    st.dataframe(df.head())

    desc = st.text_area("Describe your chart")

    if st.button("Generate chart"):

        spec = interpret_request(desc, df)

        st.subheader("Detected Interpretation")
        st.json(spec)

        errors = validate_spec(spec, df)

        if errors:
            st.error("❌ " + " | ".join(errors))

        else:
            data, x, y = process_data(df, spec)

            st.success("✅ Chart generated successfully")

            chart = render_chart(spec, data, x, y)
            st.altair_chart(chart, use_container_width=True)

            st.dataframe(data)

else:
    st.info("Upload a dataset to start")
