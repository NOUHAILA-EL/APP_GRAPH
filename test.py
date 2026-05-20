import streamlit as st
import pandas as pd
import altair as alt

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Cap Visual Intelligence", layout="wide")

st.title("Cap Visual Intelligence powered by GenAI")
st.write("Upload your dataset, then use Guided Mode or AI Mode to create charts.")

# ---------------------------
# LOAD DATA
# ---------------------------
uploaded = st.file_uploader("Upload CSV file", type=["csv"])

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# ---------------------------
# AI INTERPRETATION (IMPROVED ✅)
# ---------------------------
def interpret_request(desc, df):

    desc = desc.lower()

    spec = {
        "chart_type": "Bar",
        "metric": None,
        "dimension": None,
        "aggregation": "sum"
    }

    # ✅ Chart type detection
    if any(x in desc for x in ["line", "courbe"]):
        spec["chart_type"] = "Line"
    elif any(x in desc for x in ["pie", "camembert"]):
        spec["chart_type"] = "Pie"
    else:
        spec["chart_type"] = "Bar"

    # ✅ Synonyms mapping
    synonyms = {
        "revenue": "Sales",
        "ventes": "Sales",
        "sales": "Sales",
        "profit": "Profit",
        "bénéfice": "Profit",
        "region": "Region",
        "région": "Region",
        "description": "Description",
        "date": "Date"
    }

    for k, v in synonyms.items():
        if k in desc:
            desc = desc.replace(k, v.lower())

    # ✅ Aggregation
    if any(x in desc for x in ["count", "nombre"]):
        spec["aggregation"] = "count"
    elif any(x in desc for x in ["mean", "moyenne"]):
        spec["aggregation"] = "mean"
    elif any(x in desc for x in ["sum", "somme", "total"]):
        spec["aggregation"] = "sum"

    # ✅ Detect columns (SAFE: ignore bad column)
    for col in df.columns:
        if col == "MixedColumn":  # ❌ skip invalid column
            continue

        if col.lower() in desc:
            if spec["metric"] is None:
                spec["metric"] = col
            else:
                spec["dimension"] = col

    # ✅ Fix: text metric → count
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
            x=alt.X(x, sort="-y"),
            y=y,
            tooltip=[x, y]
        )

    elif chart_type == "Line":
        chart = alt.Chart(data).mark_line(point=True).encode(
            x=x,
            y=y,
            tooltip=[x, y]
        )

    elif chart_type == "Pie":
        chart = alt.Chart(data).mark_arc().encode(
            theta=y,
            color=x,
            tooltip=[x, y]
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

    mode = st.radio("Choose Mode", ["Guided Mode", "AI Mode"])

    # =========================
    # ✅ GUIDED MODE (SAFE ✅)
    # =========================
    if mode == "Guided Mode":

        st.subheader("Build your chart manually")

        chart_type = st.selectbox("Chart Type", ["Bar", "Line", "Pie"])

        # ✅ Safe numeric columns
        numeric_cols = [
            col for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col])
        ]

        # ✅ Safe categorical columns (exclude MixedColumn)
        categorical_cols = [
            col for col in df.columns
            if not pd.api.types.is_numeric_dtype(df[col])
            and col != "MixedColumn"
        ]

        if not numeric_cols:
            st.error("No numeric columns found")
        elif not categorical_cols:
            st.error("No categorical columns found")
        else:
            metric = st.selectbox("Metric (Y-axis)", numeric_cols)
            dimension = st.selectbox("Dimension (X-axis)", categorical_cols)

            aggregation = st.selectbox("Aggregation", ["sum", "mean", "count"])

            if st.button("Generate chart"):
                data = process_data(df, metric, dimension, aggregation)

                st.success("✅ Chart created")

                chart = render_chart(chart_type, data, dimension, "value")
                st.altair_chart(chart, use_container_width=True)

                st.dataframe(data)

    # =========================
    # ✅ AI MODE (UPDATED ✅)
    # =========================
    elif mode == "AI Mode":

        st.subheader("Describe your chart")

        st.markdown("""
### ✅ Supported format  
👉 **Chart + aggregation + metric + by + dimension**

---

### 📘 Working Examples (EN)

- Bar chart of sum Sales by Region  
- Bar chart of sum Profit by Region  
- Line chart of sum Sales by Date  
- Pie chart of sum Sales by Region  

---

### 📗 Exemples (FR)

- Barres somme ventes par région  
- Barres somme profit par région  
- Courbe somme ventes par date  
- Camembert somme ventes par région  
""")

        example = st.selectbox(
            "Choose an example",
            [
                "",
                "Bar chart of sum Sales by Region",
                "Bar chart of sum Profit by Region",
                "Line chart of sum Sales by Date",
                "Pie chart of sum Sales by Region",
                "Barres somme ventes par région"
            ]
        )

        if "ai_input" not in st.session_state:
            st.session_state.ai_input = ""

        if example:
            st.session_state.ai_input = example

        desc = st.text_area("Your request", value=st.session_state.ai_input)

        if st.button("Generate chart"):

            if not desc.strip():
                st.warning("Please enter a description")
            else:
                spec = interpret_request(desc, df)

                st.subheader("AI Interpretation")
                st.json(spec)

                if not spec["metric"] or not spec["dimension"]:
                    st.error("❌ Could not clearly interpret your request")
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

                    st.success("✅ Chart generated successfully")
                    st.altair_chart(chart, use_container_width=True)

                    st.dataframe(data)

else:
    st.info("Upload a dataset to start")
