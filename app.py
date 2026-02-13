if generate:
    if df is None:
        st.warning("Please upload a dataset first.")
    elif not desc or not desc.strip():
        st.info("Please describe your chart above, then click Generate.")
    else:
        try:
            t = normalize_text(desc)

            chart_type = infer_chart_type(t)
            # 1) parse group (dimension)
            dim = parse_group(t, df.columns.tolist())

            # 2) parse metric + aggregation
            agg_func, metric_col = parse_metric_and_agg(t, df)

            # 3) parse sorting (order + columns)
            sort_order, sort_cols = parse_sort(t, df)
            if sort_order is None:
                sort_order = "descending" if "descending" in t or "desc" in t else None

            # 4) build aggregated frame (also aggregate sort columns if needed)
            work, x, y = build_aggregated(df, dim, agg_func, metric_col, sort_cols)

            # 5) compute multi-key sort
            sort_keys = []
            ascending = []
            # if user asked "sort by revenue, profit", convert synonyms and ensure those cols exist in work
            for sc in sort_cols:
                col = sc
                # try to match expected aggregated name if needed
                if col not in work.columns and f"sum_{col}" in work.columns:
                    col = f"sum_{col}"
                if col in work.columns:
                    sort_keys.append(col)
                    ascending.append(sort_order == "ascending")
            # always add main measure last if nothing else parsed
            if not sort_keys:
                sort_keys = [y]
                ascending = [sort_order == "ascending"] if sort_order else [False]

            work = work.sort_values(by=sort_keys, ascending=ascending)

            # 6) order categories in chart following sorted dataframe
            ordered_categories = work[x].astype(str).tolist()
            # deduplicate while keeping order
            seen = set()
            ordered_categories = [c for c in ordered_categories if not (c in seen or seen.add(c))]

            # 7) build the chart
            if chart_type == "pie":
                # For pie, need (category, value)
                value_col = y if y in work.columns else None
                if value_col is None or not pd.api.types.is_numeric_dtype(work[value_col]):
                    st.warning("Pie needs a numeric measure. Try 'sum of Sales by Region'.")
                    chart = None
                else:
                    chart = alt.Chart(work).mark_arc().encode(
                        theta=alt.Theta(field=value_col, type="quantitative"),
                        color=alt.Color(field=x, type="nominal", sort=ordered_categories),
                        tooltip=[x, value_col]
                    )
            elif chart_type == "line":
                chart = alt.Chart(work).mark_line(point=True).encode(
                    x=alt.X(x, sort=ordered_categories),
                    y=y,
                    tooltip=list(work.columns)
                )
            elif chart_type in ["bar", "bar_stacked"]:
                enc = {
                    "x": alt.X(x, sort=ordered_categories),
                    "y": y,
                    "tooltip": list(work.columns)
                }
                chart = alt.Chart(work).mark_bar().encode(**enc)
            elif chart_type == "scatter":
                # If you need size/color, you can extend parsing similarly
                chart = alt.Chart(work).mark_point(filled=True).encode(
                    x=x, y=y, tooltip=list(work.columns)
                )
            elif chart_type == "hist":
                bins = 30
                chart = alt.Chart(df).mark_bar().encode(
                    x=alt.X(alt.Bin(maxbins=bins), field=metric_col, type='quantitative'),
                    y='count()'
                )
            else:
                chart = alt.Chart(work).mark_bar().encode(
                    x=alt.X(x, sort=ordered_categories),
                    y=y,
                    tooltip=list(work.columns)
                )

            if chart is not None:
                title = f"{chart_type.capitalize()} of {y} by {x}"
                st.subheader("Chart")
                st.altair_chart(
                    chart.properties(width='container', height=420, title=title),
                    use_container_width=True
                )
                st.download_button(
                    "Download chart data as CSV",
                    data=work.to_csv(index=False).encode("utf-8"),
                    file_name="chart_data.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Could not build the chart: {e}")
``
