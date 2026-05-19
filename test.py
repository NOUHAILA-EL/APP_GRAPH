import streamlit as st
import pandas as pd
import altair as alt

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Cap Visual Intelligence", layout="wide")

st.title("Cap Visual Intelligence powered by GenAI")

st.write("Upload dataset, choose guided mode OR describe the chart.")

# ---------------------------
# LOAD DATA
# ---------------------------
uploaded = st.file_uploader("Upload CSV file", type=["csv"])

@st.cache_data
def load_data(file):
    return pd.read_csv(file)


# ---------------------------
# AI INTERPRETATION (SIMPLE)
# ---------------------------
def interpret_request(desc, df):

    desc = desc.lower()

    spec = {
        "chart_type": "Bar",
        "metric": None,
        "dimension": None,
        "aggregation": "sum"
    }

    # chart type
    if "line" in desc:
        spec["chart_type"] = "Line"
    elif "pie" in desc or "camembert" in desc:
        spec["chart_type"] = "Pie"

    # synonyms
    synonyms = {
        "revenue": "Sales",
        "ventes": "Sales",
        "profit": "Profit",
        "bénéfice": "Profit",
        "region": "Region",
        "région": "Region",
        "description": "Description"
    }

    for k, v in synonyms.items():
        if k in desc:
            desc = desc.replace(k, v.lower())

    # aggregation
    if "count" in desc:
        spec["aggregation"] = "count"
    elif "mean" in desc:
        spec["aggregation"] = "mean"
    elif "sum" in desc or "total" in desc:
        spec["aggregation"] = "sum"

    # detect columns
    for col in df.columns:
        if col.lower() in desc:
            if spec["metric"] is None:
                spec["metric"] = col
            else:
                spec["dimension"] = col

    # fix text aggregation
    if spec["metric"] and not pd.api.types.is_numeric_dtype(df[spec["metric"]]):
        spec["aggregation"] = "count"

    return spec


# ---------------------------
# PROCESS DATA
# ---------------------------
def process_data(df, metric, dimension, aggregation):

    if aggregation == "count":
        data = df.groupby(dimension)[metric].count().reset_index(name="value")

    elif aggregation == "mean":
        data = df.groupby(dimension)[metric].mean().reset_index(name="value")

    else:
        data = df.groupby(dimension)[metric].sum().reset_index(name="value")

    return data


# ---------------------------
# RENDER CHART
# ---------------------------
def render_chart(chart_type, data, x, y):

    if chart_type == "Bar":
        chart = alt.Chart(data).mark_bar().encode(
            x=x,
            y=y,
            tooltip=[x, y]
        )

    elif chart_type == "Line":
        chart = alt.Chart(data).mark_line(point=True).encode(
            x=x,
            y=y
        )

    elif chart_type == "Pie":
        chart = alt.Chart(data).mark_arc().encode(
            theta=y,
            color=x
        )

    else:
        chart = alt.Chart(data).mark_bar().encode(x=x, y=y)

    return chart


# ---------------------------
# MAIN APP
# ---------------------------
if uploaded:

    df = load_data(uploaded)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ---------------------------
    # MODE SWITCH
    # ---------------------------
    mode = st.radio("Choose Mode", ["Guided Mode", "AI Mode"])

    # ===========================
    # ✅ GUIDED MODE
    # ===========================
    if mode == "Guided Mode":

        st.subheader("Build your chart")

        chart_type = st.selectbox(
            "Chart Type",
            ["Bar", "Line", "Pie"]
        )

        # numeric columns
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        # categorical columns
        cat_cols = df.select_dtypes(exclude="number").columns.tolist()

        metric = st.selectbox("Metric (Y-axis)", numeric_cols)

        dimension = st.selectbox("Dimension (X-axis)", cat_cols)

        aggregation = st.selectbox(
            "Aggregation",
            ["sum", "mean", "count"]
        )

        if st.button("Generate chart:

"):
            data = process_data(df, metric, dimension, aggregation)

            st.success("✅ Chart created")

            chart = render_chart(chart_type, data, dimension, "value")
            st.altair_chart(chart, use_container_width=True)

            st.dataframe(data)

    # ===========================
    # ✅ AI MODE
    # ===========================
    else:

        desc = st.text_area("Describe your chart")

        if st.button("Generate chart"):

            spec = interpret_request(desc, df)

            st.subheader("AI Interpretation")
            st.json(spec)

            # validation
            if not spec["metric"] or not spec["dimension"]:
                st.error("❌ Could not understand request clearly")
            else:
                data = process_data(
                    df,
                    spec["metric"],
                    spec["dimension"],
                    spec["aggregation"]
                )

                chart = render_chart(
                    spec["chart_type"],
                    data,
                    spec["dimension"],
                    "value"
                )

                st.success("✅ Chart generated")
                st.altair_chart(chart, use_container_width=True)

                st.dataframe(data)

else:
    st.info("Upload a dataset to start")
``
