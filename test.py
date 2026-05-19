# =========================
# ✅ AI MODE (UPDATED)
# =========================
elif mode == "AI Mode":

    st.subheader("Describe your chart")

    st.markdown("""
Upload your dataset, describe the visual (EN/FR), then click Generate chart.

### 📘 Examples (EN)
- Bar chart of count of Description grouped by Region; sort descending by revenue and profit  
- Line chart of Sales over Date; rolling average 7; color by Region  
- Pie (donut) of sum of Sales by Region; title: Sales share  
- Heatmap of sum of Profit by Region and Description  
- Table of sum of Sales by Region  
- Matrix of sum of Sales by Region and Description  
- KPI of sum of Sales; title: Total Sales  

### 📗 Exemples (FR)
- Un camembert de la somme des ventes par région  
- Barres du nombre de Description par Région; tri descendant par bénéfice  
- Courbe des ventes sur Date; moyenne mobile 7; couleur par Région  
- Tableau de la somme des ventes par région  
- Matrice de la somme des ventes par région et description  
- KPI de la somme des ventes; titre: CA total  
""")

    # ✅ Predefined examples dropdown
    example_choice = st.selectbox(
        "Choose an example (optional)",
        [
            "",
            "Bar chart of count of Description grouped by Region; sort descending by revenue and profit",
            "Line chart of Sales over Date; rolling average 7; color by Region",
            "Pie of sum of Sales by Region",
            "Heatmap of sum of Profit by Region and Description",
            "Table of sum of Sales by Region",
            "Matrix of sum of Sales by Region and Description",
            "KPI of sum of Sales"
        ]
    )

    # ✅ Text input (auto-filled if example selected)
    if "ai_input" not in st.session_state:
        st.session_state.ai_input = ""

    if example_choice:
        st.session_state.ai_input = example_choice

    desc = st.text_area("Your request", value=st.session_state.ai_input)

    # ✅ Generate
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
