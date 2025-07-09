import streamlit as st
import pandas as pd
from datetime import datetime
import os
import altair as alt
from dateutil.relativedelta import relativedelta
from src.promo_analyzer import analyze_promotion_lift
from src.model_handler import  prepare_prediction_data,load_latest_predictor,generate_future_covariates,generate_static_features,load_prediction_artifacts
from pymongo import MongoClient
from dotenv import load_dotenv
from src.data_loader import load_latest_recommendation_data, get_last_n_months_sales,load_dataframe_from_mongo
from typing import Optional
from src import config 
from autogluon.timeseries import TimeSeriesDataFrame  
from src.seasonal_analysis import identify_seasonal_skus,calculate_seasonal_strength
from statsmodels.tsa.seasonal import seasonal_decompose

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://root:example@localhost:27017/")
MONGO_DB = os.getenv("MONGO_DB", "sales_automl")

st.set_page_config(
    page_title="Inventory Recommendations Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_recommendation_data_from_mongo() -> Optional[pd.DataFrame]:
    """
    Loads the latest inventory recommendations from MongoDB using data_loader.
    """
    try:
        df = load_latest_recommendation_data(mongo_uri=MONGO_URI, db_name=MONGO_DB)
        if df.empty:
            st.warning("No data found in the latest recommendations collection.")
            return None
        return df
    except Exception as e:
        st.error(f"⚠️ An error occurred while loading the latest recommendations: {str(e)}", icon="🚨")
        return None

def save_feedback_to_mongo(feedback_df: pd.DataFrame, collection_name="feedback_data", mongo_uri=MONGO_URI, db_name=MONGO_DB):
    """
    Saves feedback data to MongoDB.
    """
    try:
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]
        collection.insert_many(feedback_df.to_dict("records"))
        client.close()
    except Exception as e:
        st.error(f"⚠️ Could not save feedback to MongoDB: {str(e)}", icon="🚨")

def display_sku_overview(sku_data: pd.Series) -> None:
    """Displays the key recommendation metrics for a selected SKU across all horizons."""
    st.subheader("Forecast Horizons & Recommendations", divider="blue")

    
    channel_order = ['App', 'Web', 'Offline']
    ordered_data = sku_data[sku_data['channel'].isin(channel_order)].sort_values(by='channel', key=lambda x: x.map({channel: i for i, channel in enumerate(channel_order)}))

    num_metrics = len(ordered_data)
    cols = st.columns(3)  
    for i, (index, row) in enumerate(ordered_data.iterrows()):
        with cols[i % 3]:  
            st.metric(
                label=f"**{row['horizon']} Forecast** ({row['forecast_days']} days) - **Channel:** `{row['channel']}`",
                value=f"{row['total_forecasted_demand']:,.0f} units"
            )
            st.markdown(f"**Safety Stock:** `{row['safety_stock']}`")
            st.markdown(f"**Reorder Point:** `{row['reorder_point']}`")
            st.markdown(f"**Order Quantity (EOQ):** `{row['economic_order_quantity (EOQ)']}`")

def display_demand_chart(sku_data: pd.Series, historical_data: pd.DataFrame) -> None:
    st.subheader("Demand Forecast Comparison", divider="gray")

    col1, col2 = st.columns(2)

    with col1:
        agg_sku_data = sku_data.groupby('horizon')['total_forecasted_demand'].sum().reset_index()

        current_chart = alt.Chart(agg_sku_data).mark_bar(
            cornerRadiusTopLeft=3,
            cornerRadiusTopRight=3
        ).encode(
            x=alt.X('horizon:N', title='Forecast Horizon', sort=['1-Month', '3-Month', '6-Month']),
            y=alt.Y('total_forecasted_demand:Q', title='Total Forecasted Demand (Units)'),
            color=alt.Color('horizon:N', legend=None, scale=alt.Scale(scheme='viridis')),
            tooltip=[
                alt.Tooltip('horizon', title='Horizon'),
                alt.Tooltip('total_forecasted_demand', title='Total Forecasted Demand', format=',.0f'),
            ]
        ).properties(
            title='Total Demand by Forecast Horizon'
        ).configure_axis(
            labelFontSize=12,
            titleFontSize=14
        ).configure_title(
            fontSize=16,
            anchor='start'
        )
        st.altair_chart(current_chart, use_container_width=True)

    
    with col2:
        demand_by_horizon = sku_data.groupby('horizon')['total_forecasted_demand'].sum()

        d1 = demand_by_horizon.get('1-Month', 0)
        d3 = demand_by_horizon.get('3-Month', 0)
        d6 = demand_by_horizon.get('6-Month', 0)

        
        monthly_demands = []
        
        monthly_demands.append(d1)
        
        demand_month_2_3 = (d3 - d1) / 2 if d3 > d1 else 0
        monthly_demands.extend([demand_month_2_3] * 2)
        
        demand_month_4_6 = (d6 - d3) / 3 if d6 > d3 else 0
        monthly_demands.extend([demand_month_4_6] * 3)

        
        future_dates = [(datetime.today() + relativedelta(months=i)).strftime('%Y-%m') for i in range(6)]

        
        forecast_df = pd.DataFrame({
            'Month': future_dates,
            'Implied Monthly Demand': monthly_demands
        })

        
        forecast_chart = alt.Chart(forecast_df).mark_line(point=True).encode(
            x=alt.X('Month:N', title='Month', sort=None),
            y=alt.Y('Implied Monthly Demand:Q', title='Implied Monthly Demand (Units)'),
            tooltip=[
                alt.Tooltip('Month', title='Month'),
                alt.Tooltip('Implied Monthly Demand', title='Forecast Demand', format=',.0f')
            ]
        ).properties(
            title='Implied Monthly Demand for Next 6 Months'
        ).configure_axis(
            labelFontSize=12,
            titleFontSize=14
        ).configure_title(
            fontSize=16,
            anchor='start'
        )
        st.altair_chart(forecast_chart, use_container_width=True)

def display_chatbot(selected_sku: str, full_df: pd.DataFrame) -> None:
    """Displays a simple chatbot for asking questions."""
    st.markdown("---")
    with st.expander("💬 Chat with your Data Assistant", expanded=False):
        if f"messages_{selected_sku}" not in st.session_state:
            st.session_state[f"messages_{selected_sku}"] = [{"role": "assistant", "content": f"How can I help you analyze SKU `{selected_sku}`?"}]

        for msg in st.session_state[f"messages_{selected_sku}"]:
            st.chat_message(msg["role"]).write(msg["content"])

        if prompt := st.chat_input("Ask about this SKU..."):
            st.session_state[f"messages_{selected_sku}"].append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            
            if "reorder point" in prompt.lower():
                
                reorder_point_data = full_df[(full_df['item_id'] == selected_sku) & (full_df['horizon'] == '1-Month')]
                if not reorder_point_data.empty:
                    
                    total_reorder_point = reorder_point_data['reorder_point'].sum()
                    response = f"For SKU `{selected_sku}`, the combined 1-Month forecast suggests a total reorder point of **{total_reorder_point} units** across all channels."
                else:
                    response = f"Sorry, I could not find the reorder point for `{selected_sku}` for the 1-Month horizon."
            else:
                response = f"Analyzing your request about '{prompt}' for SKU `{selected_sku}`. I can provide details on demand, safety stock, and reorder points."

            st.session_state[f"messages_{selected_sku}"].append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)

def save_feedback(selected_sku: str, sku_data: pd.DataFrame, feedback: str):
    """Saves the user feedback to MongoDB."""
    
    feedback_records = []
    for _, row in sku_data.iterrows():
        feedback_records.append({
            "timestamp": datetime.now(),
            "selected_sku": selected_sku,
            "feedback": feedback,
            "horizon": row['horizon'],
            "channel": row['channel'],
            "total_forecasted_demand": row['total_forecasted_demand'],
            "safety_stock": row['safety_stock'],
            "reorder_point": row['reorder_point'],
            "eoq": row['economic_order_quantity (EOQ)']
        })
    
    new_feedback_df = pd.DataFrame(feedback_records)

    try:
        
        save_feedback_to_mongo(new_feedback_df)
        st.success("Thank you for your feedback! It has been recorded.")
    except Exception as e:
        st.error(f"⚠️ An error occurred while saving feedback: {str(e)}", icon="🚨")

def capture_feedback(selected_sku: str, sku_data: pd.Series):
    """Displays feedback buttons and saves the feedback to MongoDB."""
    st.subheader("Was this forecast helpful?", divider="green")
    
    feedback_col1, feedback_col2 = st.columns(2)

    with feedback_col1:
        if st.button("👍 Good Forecast", use_container_width=True):
            save_feedback(selected_sku, sku_data, "Good")

    with feedback_col2:
        if st.button("👎 Bad Forecast", use_container_width=True):
            save_feedback(selected_sku, sku_data, "Bad")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Recommendations", "Analyze Data", "New SKU Forecast","Promotion Analysis","Seasonal Analysis"])

if page == "Recommendations":
    reco_df = load_latest_recommendation_data()

    if reco_df is None or reco_df.empty:
        st.info("ℹ️ Awaiting recommendation data. Please ensure the main pipeline has run.", icon="⏳")
    else:
        with st.sidebar:
            st.header("⚙️ SKU Selection")
            sku_list = sorted(reco_df['item_id'].unique())
            selected_sku = st.selectbox("Select or Search SKU:", sku_list, index=0)

        if selected_sku:
            st.header(f"Analysis for Item: `{selected_sku}`", divider="rainbow")
            
            
            sku_reco_data = reco_df[reco_df['item_id'] == selected_sku]
            
            
            sku_historical_data = get_last_n_months_sales(sku_list=[selected_sku], quantity_col='qty', months_back=6)
            if sku_reco_data.empty:
                st.error(f"No recommendation data found for item: {selected_sku}")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    total_historical = sku_historical_data['qty'].sum() if not sku_historical_data.empty else 0
                    st.metric(label="Last 6-Month Actual Sales", value=f"{total_historical:,.0f} units")
                with col2:
                    total_forecasted = sku_reco_data[sku_reco_data['horizon'] == '6-Month']['total_forecasted_demand'].sum()
                    st.metric(label="Next 6-Month Forecasted Sales", value=f"{total_forecasted:,.0f} units")

                display_sku_overview(sku_reco_data)
                
                display_demand_chart(sku_reco_data, sku_historical_data)

                with st.expander("📋 View Raw Recommendation Data"):
                    st.dataframe(sku_reco_data, use_container_width=True, hide_index=True)

                display_chatbot(selected_sku, reco_df)
                st.markdown("---")
                capture_feedback(selected_sku, sku_reco_data)
        else:
            st.info("Please select an item from the sidebar to see the detailed analysis.", icon="👈")

elif page == "Analyze Data":
    st.header("Analyze Data with EDA")
    
    eda_dir = os.path.join(config.PROJECT_SRC, "eda", "target")
    if not os.path.isdir(eda_dir):
        os.makedirs(eda_dir, exist_ok=True)
        st.warning(f"Created EDA directory at: {eda_dir}")
        st.info("Please run the EDA script to generate charts.")
    else:
        charts = sorted([f for f in os.listdir(eda_dir) if f.endswith(".png")])
        if not charts:
            st.warning("No PNG charts found in the EDA directory. Please run the EDA script first.")
        else:
            for chart in charts:
                chart_path = os.path.join(eda_dir, chart)
                try:
                    st.image(chart_path, caption=chart, use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to load chart {chart}: {str(e)}")

elif page == "New SKU Forecast":
    st.header("📤 Forecast on New Data")

    st.info("""
    **Required CSV format:**
    - `sku`: Product identifier (e.g., A0215)
    - `timestamp`: Date (e.g., `2025-07-04`)
    - `target`: Sales quantity (numeric)
    - `disc`: Discount percentage (numeric)
    """)

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        try:
            user_data_raw = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview:")
            st.dataframe(user_data_raw.head())

            required_columns = ["sku", "timestamp", "target", "disc"]
            if not all(col in user_data_raw.columns for col in required_columns):
                st.error(f"Missing required columns. Please ensure your CSV has: {required_columns}")
            else:
                with st.status("🚀 Initializing forecast process...", expanded=True) as status:
                    try:
                        
                        status.write("1. Loading trained model...")
                        predictor = load_latest_predictor()
                        if predictor is None:
                            status.update(label="🚨 Error: Could not load model.", state="error")
                            st.error("Could not load a trained model. Please run the main training pipeline first.")
                            st.stop()

                        
                        status.write("2. Loading training artifacts (features & holidays)...")
                        static_feature_columns, holidays_df = load_prediction_artifacts()

                        
                        status.write("3. Preparing and enriching uploaded data...")
                        enriched_data = prepare_prediction_data(user_data_raw, holidays_df)
                        
                        
                        status.write("4. Creating time series structure...")
                        enriched_data['channel'] = 'Online'
                        enriched_data['item_id'] = enriched_data['sku'].astype(str) + "_" + enriched_data['channel']
                        static_features = generate_static_features(enriched_data, all_training_columns=static_feature_columns)
                        static_features.reset_index(inplace=True)
                        ts_upload = TimeSeriesDataFrame.from_data_frame(
                            enriched_data,
                            id_column='item_id',
                            timestamp_column='timestamp',
                            static_features_df=static_features
                        )

                        
                        status.write("5. Evaluating model performance on historical data...")
                        min_series_length = ts_upload.index.get_level_values('item_id').value_counts().min()
                        if min_series_length > predictor.prediction_length:
                            
                            
                            metrics = predictor.evaluate(ts_upload, display=False)
                            status.write("   -> ✅ Performance evaluation complete.")
                        else:
                            reason = f"Uploaded data history ({min_series_length} points) is not longer than the model's prediction length ({predictor.prediction_length} points)."
                            metrics = pd.DataFrame([{"info": "Evaluation skipped", "reason": reason}])
                            status.write("   -> ⚠️ Performance evaluation skipped (data too short).")

                        
                        status.write("6. Generating future forecast...")
                        future_known_covariates = generate_future_covariates(predictor, ts_upload, holidays_df)
                        predictions = predictor.predict(
                            ts_upload,
                            known_covariates=future_known_covariates
                        )
                    
                        predictions = predictions.clip(lower=0).round(0).astype(int)
                        
                        status.update(label="✅ Process Complete!", state="complete", expanded=False)

                    except Exception as e:
                        status.update(label=f"🚨 Error: {e}", state="error")
                        st.error(f"An error occurred during forecast generation: {e}")
                        st.stop()
                
                
                st.subheader("Performance on Uploaded Data (Backtest)")
                st.dataframe(metrics)

                st.subheader("Future Forecast")
                st.dataframe(predictions)

                st.download_button(
                   label="Download Forecast Data (CSV)",
                   data=predictions.to_csv(index=True).encode('utf-8'),
                   file_name='forecast_results.csv',
                   mime='text/csv',
                )

        except Exception as e:
            st.error(f"Error reading the uploaded file: {str(e)}")

elif page == "Promotion Analysis":
    st.header("🔬 Promotion Effectiveness Analysis")
    
    lookback_days = st.slider(
        "Select analysis period (days):", 
        min_value=184, 
        max_value=365, 
        value=200, 
        step=7
    )

    if st.button("🚀 Run Analysis", use_container_width=True):
        with st.spinner("Analyzing promotion performance... This may take a moment."):
            try:
                predictor = load_latest_predictor()
                if predictor is None:
                    st.error("🚨 Could not load a trained model. Please run the main training pipeline first.")
                    st.stop()
                
                lift_summary_df = analyze_promotion_lift(predictor, lookback_days)
                if lift_summary_df.empty:
                    st.write(lift_summary_df)
                    st.warning("No promotion data was found in the selected period to analyze.")
                else:
                    st.subheader("Promotion Performance Summary")
                    st.caption(f"Comparing actual sales vs. forecasted sales without promotions over the last {lookback_days} days.")
                    
                    
                    st.dataframe(lift_summary_df,
                        column_config={
                            "total_actual_sales": st.column_config.NumberColumn("Actual Sales", format="%d units"),
                            "total_forecasted_sales_no_promo": st.column_config.NumberColumn("Est. Sales w/o Promo", format="%d units"),
                            "total_lift_units": st.column_config.NumberColumn("Sales Lift (Units)", format="%d"),
                            "percentage_lift": st.column_config.ProgressColumn(
                                "Sales Lift (%)",
                                help="The percentage increase in sales attributable to promotions.",
                                format="%.2f%%",
                                min_value=lift_summary_df['percentage_lift'].min(),
                                max_value=lift_summary_df['percentage_lift'].max(),
                            ),
                        },
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")

elif page == "Seasonal Analysis":
    st.header("❄️ Advanced Seasonal SKU Analysis Dashboard")

    st.markdown("""
    Analyze SKUs for seasonal trends and their performance during major sales events.
    """)

    seasonal_strength_threshold = st.slider(
        "Minimum Seasonal Strength (0.0 - 1.0):",
        min_value=0.0,
        max_value=1.0,
        value=config.MIN_SEASONAL_STRENGTH,
        step=0.05,
        key='seasonal_strength_slider'
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        seasonal_period_days = st.selectbox(
            "Seasonal Period (Days):",
            options=[7, 30, 90, 365],
            index=3,
            format_func=lambda x: f"{x} days ({'Weekly' if x==7 else 'Monthly (approx)' if x==30 else 'Quarterly (approx)' if x==90 else 'Annual'})",
            key='seasonal_period_select'
        )

    with col2:
        seasonal_model = st.selectbox(
            "Decomposition Model:",
            options=['additive', 'multiplicative'],
            index=0,
            help="Multiplicative is better for percentage-based seasonality."
        )

    with col3:
        fill_method = st.selectbox(
            "Missing Date Fill Method:",
            options=['interpolate', 'zero', 'ffill'],
            index=0,
            help="How to fill gaps in time series before decomposition."
        )

    sales_events = {
        "Valentine's Day": ("02-10", "02-15"),
        "Holi": ("03-01", "03-31"),
        "Eid": ("04-01", "04-30"),
        "Mother's Day": ("05-07", "05-14"),
        "Father's Day": ("06-10", "06-20"),
        "Raksha Bandhan": ("08-01", "08-31"),
        "Independence Day (India)": ("08-10", "08-20"),
        "Ganesh Chaturthi": ("09-01", "09-30"),
        "Navratri": ("10-01", "10-24"),
        "Dussehra": ("10-15", "10-25"),
        "Diwali": ("10-15", "11-15"),
        "Black Friday": ("11-20", "11-30"),
        "Cyber Monday": ("12-01", "12-05"),
        "Christmas": ("12-20", "12-31"),
        "New Year": ("01-01", "01-07"),
        "Republic Day (India)": ("01-20", "01-27"),
        "Women's Day": ("03-05", "03-10"),
        "Baisakhi": ("04-10", "04-20"),
        "Onam": ("08-15", "09-10"),
        "Durga Puja": ("10-10", "10-20"),
        "Thanksgiving": ("11-20", "11-30"),
        "Makar Sankranti": ("01-10", "01-20"),
        "Christmas/New Year Sale": ("12-20", "01-05")
    }

    st.markdown("**Special Sales Seasons:**")
    selected_events = st.multiselect(
        "Select sales events to analyze SKU performance:",
        list(sales_events.keys()),
        default=["Valentine's Day", "Diwali", "Black Friday"]
    )

    if st.button("🔍 Analyze SKUs", key='run_advanced_seasonal_analysis'):
        st.info(
            f"Analyzing seasonality with strength ≥ {seasonal_strength_threshold}, "
            f"{seasonal_period_days}-day period, model '{seasonal_model}', fill '{fill_method}'..."
        )

        try:
            sales_data = load_dataframe_from_mongo("sales_data")
            sales_data = sales_data.rename(columns={
                'qty': 'target',
                'sku': 'item_id',
                'created_at': 'timestamp'
            })
            sales_data['timestamp'] = pd.to_datetime(sales_data['timestamp'])
            analysis_data = sales_data.copy()
        except Exception as e:
            st.error(f"Failed to load sales data: {e}")
            analysis_data = pd.DataFrame()

        if analysis_data.empty:
            st.warning("No data to analyze.")
        else:
            with st.spinner("Running decomposition and aggregating results..."):
                seasonal_sku_info_df = identify_seasonal_skus(
                    sales_data=analysis_data[['item_id', 'timestamp', 'target']],
                    min_seasonal_strength=seasonal_strength_threshold,
                    period_days=seasonal_period_days,
                    time_col='timestamp',
                    id_col='item_id',
                    target_col='target',
                    model=seasonal_model,
                    fill_method=fill_method
                )

                # Compute sales during special events
                event_sales_summary = []
                for event_name in selected_events:
                    start_mmdd, end_mmdd = sales_events[event_name]
                    mask = analysis_data['timestamp'].dt.strftime("%m-%d").between(start_mmdd, end_mmdd)
                    event_data = analysis_data.loc[mask]
                    event_agg = (
                        event_data.groupby('item_id')['target']
                        .sum()
                        .reset_index()
                        .rename(columns={'target': f'{event_name}_Sales'})
                    )
                    if event_sales_summary:
                        event_sales_summary[0] = event_sales_summary[0].merge(event_agg, on='item_id', how='outer')
                    else:
                        event_sales_summary.append(event_agg)

                # Combine with seasonal strength
                seasonal_df = seasonal_sku_info_df.copy()
                if event_sales_summary:
                    seasonal_df = seasonal_df.merge(event_sales_summary[0], on='item_id', how='left')

                # Fill NaNs with 0
                seasonal_df = seasonal_df.fillna(0)
                st.session_state['seasonal_skus'] = seasonal_df

                st.success(f"✅ Analysis complete. {len(seasonal_df)} SKUs evaluated.")

    if 'seasonal_skus' in st.session_state and not st.session_state['seasonal_skus'].empty:
        df = st.session_state['seasonal_skus']

        st.subheader("Top 100 SKUs by Seasonal Strength")
        top100 = df.sort_values(by="seasonal_strength", ascending=False).head(100)
        st.dataframe(top100, use_container_width=True)

        st.subheader("Seasonal vs Non-Seasonal SKU Distribution")
        seasonal_count = (df['is_seasonal'] == True).sum()
        non_seasonal_count = len(df) - seasonal_count
        st.write(f"Seasonal: {seasonal_count}")
        st.write(f"Non-Seasonal: {non_seasonal_count}")

        pie_data = pd.DataFrame({
            'Category': ['Seasonal', 'Non-Seasonal'],
            'Count': [seasonal_count, non_seasonal_count]
        })

        pie_chart = alt.Chart(pie_data).mark_arc().encode(
            theta=alt.Theta(field="Count", type="quantitative"),
            color=alt.Color(field="Category", type="nominal"),
            tooltip=["Category", "Count"]
        )
        st.altair_chart(pie_chart, use_container_width=True)

        st.subheader("SKU Performance During Selected Events")
        event_cols = [col for col in df.columns if col.endswith('_Sales')]
        if event_cols:
            st.dataframe(df[['item_id'] + event_cols].sort_values(by=event_cols, ascending=False), use_container_width=True)
        else:
            st.info("No event sales columns to display.")

        st.subheader("Detailed SKU Viewer")
        selected_sku = st.selectbox("Select SKU to view decomposition and sales history:", df['item_id'].unique())
        sku_data = load_dataframe_from_mongo("sales_data")
        sku_data = sku_data.rename(columns={'qty': 'target', 'sku': 'item_id', 'created_at': 'timestamp'})
        sku_data['timestamp'] = pd.to_datetime(sku_data['timestamp'])

        sku_series = (
            sku_data[sku_data['item_id'] == selected_sku]
            .groupby('timestamp')
            ['target']
            .sum()
            .sort_index()
        )

        sku_series_reindexed = sku_series.reindex(
            pd.date_range(sku_series.index.min(), sku_series.index.max(), freq="D"),
            fill_value=0
        )

        if not sku_series.empty:
            decomposition = seasonal_decompose(
                sku_series_reindexed,
                model=seasonal_model,
                period=seasonal_period_days,
                extrapolate_trend=seasonal_period_days
                )

            st.caption("**Trend:** The long-term movement in sales, ignoring seasonality and noise.")
            st.line_chart(decomposition.trend, height=200)

            st.caption("**Seasonal:** The repeating seasonal pattern (e.g., weekly, yearly) in sales.")
            st.line_chart(decomposition.seasonal, height=200)

            st.caption("**Residual:** The random noise or irregularities not explained by trend or seasonality.")
            st.line_chart(decomposition.resid, height=200)
        else:
            st.warning("No data available for the selected SKU.")

    else:
        st.info("Run analysis to see seasonal insights.")
