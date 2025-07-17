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
from src.data_loader import load_latest_recommendation_data, get_last_n_months_sales,load_dataframe_from_mongo,get_latest_model_metrics,get_top_skus_by_forecast
from typing import Optional
from src import config 
from autogluon.timeseries import TimeSeriesDataFrame  
from src.seasonal_analysis import identify_seasonal_skus,calculate_seasonal_strength
from statsmodels.tsa.seasonal import seasonal_decompose
from urllib.parse import quote_plus
from src import dbConnect


load_dotenv()

mongo_user = os.getenv("MONGO_USERNAME")
mongo_pass = quote_plus(os.getenv("MONGO_PASSWORD", ""))
mongo_host = os.getenv("MONGO_HOST")
mongo_port = int(os.getenv("MONGO_PORT", 27017))
MONGO_DB = os.getenv("MONGO_DB", "sales_automl")

# Build the full, correct connection string
MONGO_URI = f"mongodb://{mongo_user}:{mongo_pass}@{mongo_host}:{mongo_port}/"


st.set_page_config(
    page_title="Inventory Recommendations Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    st.subheader("Was this forecast helpful?", divider="green")
    
    feedback_col1, feedback_col2 = st.columns(2)

    with feedback_col1:
        if st.button("👍 Good Forecast", use_container_width=True):
            save_feedback(selected_sku, sku_data, "Good")

    with feedback_col2:
        if st.button("👎 Bad Forecast", use_container_width=True):
            save_feedback(selected_sku, sku_data, "Bad")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Recommendations", "New SKU Forecast","Promotion Analysis","Seasonal Analysis","Executive Summary","Inventory Optimization"])

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

            last_1 = get_last_n_months_sales(sku_list=[selected_sku], quantity_col='qty', months_back=1)
            last_3 = get_last_n_months_sales(sku_list=[selected_sku], quantity_col='qty', months_back=3)
            last_6 = get_last_n_months_sales(sku_list=[selected_sku], quantity_col='qty', months_back=6)

            if sku_reco_data.empty:
                st.error(f"No recommendation data found for item: {selected_sku}")
            else:
                col1, col2 ,col3 = st.columns(3)
                with col1:
                    st.metric(label="Last 6-Month Actual Sales", value=f"{last_6['qty'].sum():,.0f} units")
                with col2:
                    st.metric(label="Last 3-Month Actual Sales", value=f"{last_3['qty'].sum():,.0f} units")
                with col3:
                    st.metric(label="Last 1-Month Actual Sales", value=f"{last_1['qty'].sum():,.0f} units")
                with col1:
                    total_forecasted = sku_reco_data[sku_reco_data['horizon'] == '6-Month']['total_forecasted_demand'].sum()
                    st.metric(label="Next 6-Month Forecasted Sales", value=f"{total_forecasted:,.0f} units")
                with col2:
                    total_forecasted = sku_reco_data[sku_reco_data['horizon'] == '3-Month']['total_forecasted_demand'].sum()
                    st.metric(label="Next 3-Month Forecasted Sales", value=f"{total_forecasted:,.0f} units")
                with col3:
                    total_forecasted = sku_reco_data[sku_reco_data['horizon'] == '1-Month']['total_forecasted_demand'].sum()
                    st.metric(label="Next 1-Month Forecasted Sales", value=f"{total_forecasted:,.0f} units")
                
                display_sku_overview(sku_reco_data)
                
                with st.expander("📋 View Raw Recommendation Data"):
                    st.dataframe(sku_reco_data, use_container_width=True, hide_index=True)
                display_chatbot(selected_sku, reco_df)
                st.markdown("---")
                capture_feedback(selected_sku, sku_reco_data)
        else:
            st.info("Please select an item from the sidebar to see the detailed analysis.", icon="👈")

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
            key='seasonal_period_select',
            help="""
            **What is Seasonal Period?**
            This tells the system how often your sales patterns repeat.
            
            **Examples:**
            • **7 days (Weekly)**: For products with weekly patterns
              - Example: Office supplies that sell more on weekdays
            • **30 days (Monthly)**: For products with monthly cycles
              - Example: Groceries that peak at month-end (salary days)
            • **90 days (Quarterly)**: For products with seasonal business cycles
              - Example: School supplies that peak every quarter
            • **365 days (Annual)**: For products with yearly seasons
              - Example: Winter clothing, festival items, AC units
            
            **Which to choose?**
            - Start with **365 days** for most products
            - Use **7 days** if you notice weekly patterns
            - Use **30 days** for monthly salary-driven purchases
            """
        )

    with col2:
        seasonal_model = st.selectbox(
            "Decomposition Model:",
            options=['additive', 'multiplicative'],
            index=0,
            help="""
            **What is Decomposition Model?**
            This determines how seasonal patterns affect your sales.
            
            **Additive Model:**
            - Seasonal effect is the SAME regardless of sales volume
            - Example: Ice cream sales increase by +100 units every summer
            - Use when seasonal boost is consistent in absolute numbers
            
            **Multiplicative Model:**
            - Seasonal effect is a PERCENTAGE of current sales
            - Example: Festival sales increase by 50% regardless of base sales
            - Use when seasonal boost grows with your business size
            
            **Which to choose?**
            - **Additive**: If seasonal increase is always the same amount
            - **Multiplicative**: If seasonal increase is a percentage (more common)
            
            **Example:**
            - Product A sells 100 units normally, 150 in festival → Use Multiplicative (50% increase)
            - Product B sells 100 units normally, 120 in festival → Use Additive (+20 units)
            """
        )

    with col3:
        fill_method = st.selectbox(
            "Missing Date Fill Method:",
            options=['interpolate', 'zero', 'ffill'],
            index=0,
            help="""
            **What is Missing Date Fill Method?**
            Sometimes your sales data has gaps (missing dates). This tells the system how to handle them.
            
            **Methods Explained:**
            
            **1. Interpolate (Recommended)**
            - Estimates missing values based on nearby days
            - Example: If Mon=10, Wed=30, system estimates Tue=20
            - Best for: Most business scenarios
            
            **2. Zero**
            - Treats missing dates as zero sales
            - Example: If Tuesday is missing, assumes 0 sales
            - Best for: When missing data means no sales (store closures)
            
            **3. Forward Fill (ffill)**
            - Uses the last known value for missing dates
            - Example: If Mon=10, Tue missing, system uses Tue=10
            - Best for: When you expect sales to remain constant for missing days
            
            **Which to choose?**
            - **Interpolate**: Generally best, especially for regular sales patterns
            - **Zero**: If missing data clearly means no sales
            - **Forward Fill**: If you expect consistent sales and want to avoid underestimating
            """
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

    st.markdown("**Set Event Date and Range for Analysis**")
    event_date_inputs = {}
    event_range_inputs = {}
    analysis_data = pd.DataFrame()  # Initialize empty DataFrame for analysis

    # Arrange event inputs in rows, each with 2 events (6 columns: event1-date, event1-before, event1-after, event2-date, event2-before, event2-after)
    events_per_row = 2
    num_rows = (len(selected_events) + events_per_row - 1) // events_per_row

    for row in range(num_rows):
        cols = st.columns(6)  # 6 columns for 2 events per row, 3 inputs each
        for i in range(2):  # 2 events per row
            idx = row * 2 + i
            if idx < len(selected_events):
                event = selected_events[idx]
                with cols[i*3]:
                    st.markdown(f"**{event}**")
                with cols[i*3 + 1]:
                    event_date = st.date_input(
                        "Date",
                        value=datetime.now(),
                        key=f"{event}_date_{idx}"  # unique key
                    )
                with cols[i*3 + 2]:
                    minus_days = st.number_input(
                        "Days before",
                        min_value=0,
                        max_value=30,
                        value=3,
                        step=1,
                        key=f"{event}_minus_{idx}"  # unique key
                    )
                    plus_days = st.number_input(
                        "Days after",
                        min_value=0,
                        max_value=30,
                        value=3,
                        step=1,
                        key=f"{event}_plus_{idx}"  # unique key
                    )
                event_date_inputs[event] = event_date
                event_range_inputs[event] = (minus_days, plus_days)

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
            sales_data['timestamp'] = pd.to_datetime(sales_data['timestamp'], format='%d/%m/%Y', errors='coerce')
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

            # Fixed event sales aggregation logic
            event_sales_data = []
            for event_name in selected_events:
                event_date = event_date_inputs[event_name]
                minus_days, plus_days = event_range_inputs[event_name]
                
                # Convert event_date to pandas Timestamp if it's not already
                if isinstance(event_date, str):
                    event_date = pd.to_datetime(event_date)
                
                start_date = pd.Timestamp(event_date) - pd.Timedelta(days=minus_days)
                end_date = pd.Timestamp(event_date) + pd.Timedelta(days=plus_days)
                
                # Debug print
                st.write(f"Analyzing {event_name}: {start_date} to {end_date}")

                analysis_data['timestamp'] = pd.to_datetime(analysis_data['timestamp'])
                mask = (analysis_data['timestamp'] >= start_date) & (analysis_data['timestamp'] <= end_date)
                event_data = analysis_data.loc[mask]
                
                # Debug print
                st.write(f"Found {len(event_data)} records for {event_name}")
                
                if not event_data.empty:
                    event_agg = (
                        event_data.groupby('item_id')['target']
                        .sum()
                        .reset_index()
                        .rename(columns={'target': f'{event_name}_Sales'})
                    )
                    event_sales_data.append(event_agg)

            # Merge all event sales data
            seasonal_df = seasonal_sku_info_df.copy()
            
            if event_sales_data:
                # Start with the first event data
                merged_events = event_sales_data[0]
                
                # Merge with remaining events
                for event_df in event_sales_data[1:]:
                    merged_events = merged_events.merge(event_df, on='item_id', how='outer')
                
                # Merge with seasonal data
                seasonal_df = seasonal_df.merge(merged_events, on='item_id', how='left')
            
            # Fill missing values with 0
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

elif page == "Executive Summary":
    st.header("📈 Executive Summary Dashboard")
    st.markdown("A high-level overview of critical business metrics.")
    
    reco_df = load_latest_recommendation_data()
    model_metrics = get_latest_model_metrics()
    sales_df = load_dataframe_from_mongo("sales_data")
    
    if model_metrics:
        with st.container():
            st.subheader("📈 Model Accuracy Metrics")
            
            # Add model trained date at the top
            model_trained_date = model_metrics.get('trained_date', 'N/A')
            print("model trained date",model_trained_date)
            if model_trained_date != 'N/A':
                try:
                    # Parse the date and format it nicely
                    if isinstance(model_trained_date, str):
                        trained_dt = pd.to_datetime(model_trained_date)
                    else:
                        trained_dt = model_trained_date
                    formatted_date = trained_dt.strftime("%B %d, %Y at %I:%M %p")
                    st.info(f"🤖 **Model Last Trained:** {formatted_date}")
                except:
                    st.info(f"🤖 **Model Last Trained:** {model_trained_date}")
            else:
                st.info("🤖 **Model Last Trained:** Information not available")
            
            col1, col2, col3 = st.columns(3)
            def safe_float_fmt(val, fmt):
                try:
                    return format(abs(float(val)), fmt)
                except Exception:
                    return str(val)

            col1.metric("MASE", safe_float_fmt(model_metrics.get('MASE', 'N/A'), ".3f"),help=(
        "MASE (Mean Absolute Scaled Error) compares the model's forecast error to a simple baseline (like last month's sales). "
        "A value below 1 means the model is better than guessing last period's sales. "
        "Example: MASE = 0.7 means the model is 30% better than the naive forecast."
    ))
            col2.metric("RMSE", safe_float_fmt(model_metrics.get('RMSE', 'N/A'), ".2f"),help=(
        "RMSE (Root Mean Squared Error) shows the typical difference between predicted and actual sales, in units. "
        "Lower is better. "
        "Example: RMSE = 12 means, on average, the forecast is off by 12 units."
    ))
            col3.metric("MAPE", safe_float_fmt(model_metrics.get('MAPE', 'N/A'), ".2%"),help=(
        "MAPE (Mean Absolute Percentage Error) shows the average error as a percentage of actual sales. "
        "Lower is better. "
        "Example: MAPE = 8% means forecasts are off by 8% on average."
    ))
            st.markdown("---")

            col1, col2, col3 = st.columns(3)
            sku_list = sorted(reco_df['item_id'].unique())
            all_sku_last_6 = get_last_n_months_sales(sku_list=sku_list, quantity_col='qty', months_back=6)
            all_sku_last_3 = get_last_n_months_sales(sku_list=sku_list, quantity_col='qty', months_back=3)
            all_sku_last_1 = get_last_n_months_sales(sku_list=sku_list, quantity_col='qty', months_back=1)
            col1.metric("All SKUs Last 6-Month Actual Sales", f"{all_sku_last_6['qty'].sum():,.0f} units")
            col2.metric("All SKUs Last 3-Month Actual Sales", f"{all_sku_last_3['qty'].sum():,.0f} units")
            col3.metric("All SKUs Last 1-Month Actual Sales", f"{all_sku_last_1['qty'].sum():,.0f} units")
            
            col1.metric("All SKUs Total Forecasted Demand (Next 6-Months)", f"{reco_df[reco_df['horizon'] == '6-Month']['total_forecasted_demand'].sum():,.0f} units")
            col2.metric("All SKUs Total Forecasted Demand (Next 3-Months)", f"{reco_df[reco_df['horizon'] == '3-Month']['total_forecasted_demand'].sum():,.0f} units")
            col3.metric("All SKUs Total Forecasted Demand (Next 1-Month)", f"{reco_df[reco_df['horizon'] == '1-Month']['total_forecasted_demand'].sum():,.0f} units")
            st.markdown("---")
           
            st.subheader("📊 Top SKUs by Forecasted Demand", divider="blue")

            # Add a number input for user to select top_n
            top_n = st.number_input("Select number of top SKUs to display:", min_value=1, max_value=100, value=50, step=1)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**1-Month Forecast**")
                top_1m = get_top_skus_by_forecast(reco_df, top_n=top_n, months=1)
                if not top_1m.empty:
                    # Only convert numeric columns to int
                    for col in top_1m.select_dtypes(include='number').columns:
                        top_1m[col] = top_1m[col].round(0).astype(int)
                    st.dataframe(top_1m, use_container_width=True, hide_index=True)
                else:
                    st.info(f"No SKUs found with forecasted demand data for the next 1 month.")

            with col2:
                st.markdown("**3-Month Forecast**")
                top_3m = get_top_skus_by_forecast(reco_df, top_n=top_n, months=3)
                if not top_3m.empty:
                    for col in top_3m.select_dtypes(include='number').columns:
                        top_3m[col] = top_3m[col].round(0).astype(int)
                    st.dataframe(top_3m, use_container_width=True, hide_index=True)
                else:
                    st.info(f"No SKUs found with forecasted demand data for the next 3 months.")

            with col3:
                st.markdown("**6-Month Forecast**")
                top_6m = get_top_skus_by_forecast(reco_df, top_n=top_n, months=6)
                if not top_6m.empty:
                    for col in top_6m.select_dtypes(include='number').columns:
                        top_6m[col] = top_6m[col].round(0).astype(int)
                    st.dataframe(top_6m, use_container_width=True, hide_index=True)
                else:
                    st.info(f"No SKUs found with forecasted demand data for the next 6 months.")

            # Add download button for all SKUs forecast data
            st.markdown("---")
            st.subheader("📥 Download Complete Forecast Data", divider="blue")
            
            if not reco_df.empty:
                # Prepare download data with timestamp
                download_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                download_filename = f"all_skus_forecast_{download_timestamp}.csv"
                
                # Create a clean version of the data for download
                download_data = reco_df.copy()
                
                # Round numeric columns for cleaner output
                numeric_columns = download_data.select_dtypes(include='number').columns
                for col in numeric_columns:
                    download_data[col] = download_data[col].round(2)
                
                # Sort by item_id and horizon for better organization
                download_data = download_data.sort_values(['item_id', 'horizon'])
                
                col1, col2, col3 = st.columns([1, 1, 2])
                
                with col1:
                    st.metric("Total SKUs", f"{len(download_data['item_id'].unique()):,}")
                
                with col2:
                    st.metric("Total Records", f"{len(download_data):,}")
                
                with col3:
                    st.download_button(
                        label="📥 Download All SKUs Forecast (CSV)",
                        data=download_data.to_csv(index=False).encode('utf-8'),
                        file_name=download_filename,
                        mime='text/csv',
                        use_container_width=True,
                        help="Download complete forecast data for all SKUs across all horizons and channels"
                    )
                
                # Show preview of download data
                with st.expander("📋 Preview Download Data", expanded=False):
                    st.dataframe(download_data.head(20), use_container_width=True, hide_index=True)
                    st.caption(f"Showing first 20 rows of {len(download_data)} total records")
            else:
                st.info("No forecast data available for download.")
    
    # --- 2. Define Functions to Use Real Data ---

    def get_total_forecasted_units(recommendations, horizon_str):
        """Calculates total forecasted units for a given horizon."""
        if recommendations is None or recommendations.empty:
            return 0
        
        # Filter for the specific forecast horizon (e.g., '1-Month')
        horizon_demand = recommendations[recommendations['horizon'] == horizon_str]
        return horizon_demand['total_forecasted_demand'].sum()

    def get_top_skus_by_forecast(recommendations):
        """Gets the top 5 SKUs based on the 1-month demand forecast."""
        if recommendations is None or recommendations.empty:
            return pd.DataFrame({'SKU': [], 'Forecasted Demand (Next 30D)': []})

        # Get 1-Month forecast, group by SKU, and sum demand
        one_month_forecast = recommendations[recommendations['horizon'] == '1-Month']
        top_skus = one_month_forecast.groupby('item_id')['total_forecasted_demand'].sum()
        top_skus = top_skus.sort_values(ascending=False).head(5).reset_index()
        top_skus.rename(columns={'item_id': 'SKU', 'total_forecasted_demand': 'Forecasted Demand (Next 30D)'}, inplace=True)
        return top_skus
    
    def get_actual_sales_data(sales):
        """Gets actual sales data for the last 6 months."""
        if sales is None or sales.empty:
            return pd.DataFrame({'Month': [], 'Actual Sales': []})
        
        sales['timestamp'] = pd.to_datetime(sales['created_at'])
        # Ensure we only look at the last 6 full months
        end_date = datetime.now().replace(day=1) - pd.Timedelta(days=1)
        start_date = end_date - pd.DateOffset(months=6)
        
        # Filter, group by month, and sum sales
        monthly_sales = sales[(sales['timestamp'] >= start_date) & (sales['timestamp'] <= end_date)]
        monthly_sales = monthly_sales.set_index('timestamp').groupby(pd.Grouper(freq='ME'))['qty'].sum().reset_index()
        monthly_sales['Month'] = monthly_sales['timestamp'].dt.strftime('%Y-%m')
        return monthly_sales[['Month', 'qty']].rename(columns={'qty': 'Actual Sales'})

    # --- Placeholder functions for complex KPIs ---
    def get_inventory_turnover():
        # NOTE: Requires COGS and Average Inventory data, which is not available.
        return 4.2

    def get_sell_through_rate():
        # NOTE: Requires Units Received data, which is not available.
        return 65.4

    # --- 3. Build the Dashboard ---
    st.subheader("Key Performance Indicators (KPIs)", divider='blue')
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Inventory Turnover", value=f"{get_inventory_turnover()}x", help="NOTE: This is a placeholder value.")
        
    with col2:
        st.metric(label="Sell-Through Rate", value=f"{get_sell_through_rate()}%", help="NOTE: This is a placeholder value.")

    with col3:
        # Using real data for forecasted units
        forecasted_units_30d = get_total_forecasted_units(reco_df, '1-Month')
        st.metric(label="Forecasted Units (Next 30D)", value=f"{forecasted_units_30d:,.0f}")

# elif page == "Inventory Optimization":
#     st.header("📦 Inventory Optimization & Simulation")
#     st.markdown("Tools for strategic inventory analysis, including ABC classification and ordering simulation.")

#     # --- 1. Define Function to Use Real Data ---
#     def load_sales_data_for_abc():
#         """
#         Loads sales data and calculates total revenue per SKU for ABC analysis.
#         """
#         sales_df = load_dataframe_from_mongo("sales_data")
#         if sales_df is None or sales_df.empty:
#             st.warning("Could not load sales data for ABC analysis.")
#             return pd.DataFrame()

#         # ABC analysis requires revenue (quantity * price)
#         if 'price' not in sales_df.columns:
#             st.error("Error: 'price' column not found in sales_data. Cannot perform revenue-based ABC analysis.")
#             return pd.DataFrame()
            
#         sales_df['total_revenue'] = sales_df['qty'] * sales_df['price']
        
#         # Group by item_id (SKU) and sum the revenue
#         abc_data = sales_df.groupby('sku')['total_revenue'].sum().reset_index()
#         abc_data = abc_data.rename(columns={'sku': 'item_id'})
#         return abc_data

#     # --- 2. Build the Dashboard ---
#     st.subheader("ABC Analysis", divider='blue')
#     st.markdown("""
#     Classify your products into A, B, and C categories based on their revenue contribution.
#     - **A-Items**: High-value products (top 80% of revenue).
#     - **B-Items**: Moderate-value products (next 15% of revenue).
#     - **C-Items**: Low-value products (bottom 5% of revenue).
#     """)

#     # Use the new function to load real data
#     abc_data = load_sales_data_for_abc()

#     if not abc_data.empty:
#         abc_data = abc_data.sort_values(by='total_revenue', ascending=False)
#         abc_data['cumulative_revenue'] = abc_data['total_revenue'].cumsum()
#         total_revenue = abc_data['total_revenue'].sum()
#         abc_data['cumulative_percentage'] = (abc_data['cumulative_revenue'] / total_revenue) * 100

#         def assign_abc_category(percentage):
#             if percentage <= 80:
#                 return 'A'
#             elif percentage <= 95:
#                 return 'B'
#             else:
#                 return 'C'

#         abc_data['category'] = abc_data['cumulative_percentage'].apply(assign_abc_category)
        
#         viz_col, data_col = st.columns([1, 2])

#         with viz_col:
#             st.markdown("##### SKU Count by Category")
#             category_counts = abc_data['category'].value_counts().reset_index()
#             category_counts.columns = ['Category', 'Count']
            
#             pie_chart = alt.Chart(category_counts).mark_arc(innerRadius=50).encode(
#                 theta=alt.Theta(field="Count", type="quantitative"),
#                 color=alt.Color(field="Category", type="nominal", scale=alt.Scale(scheme='viridis')),
#                 tooltip=['Category', 'Count']
#             ).properties(height=250)
#             st.altair_chart(pie_chart, use_container_width=True)

#         with data_col:
#             st.markdown("##### Top 10 SKUs by Revenue")
#             st.dataframe(
#                 abc_data.head(10),
#                 column_config={
#                     "item_id": "SKU",
#                     "total_revenue": st.column_config.NumberColumn("Total Revenue", format="$%.2f"),
#                     "cumulative_percentage": st.column_config.ProgressColumn("Revenue Contribution", format="%.2f%%", min_value=0, max_value=100),
#                     "category": "Category"
#                 },
#                 use_container_width=True,
#                 hide_index=True
#             )

#     # --- The EOQ simulation remains the same as it is interactive ---
#     st.subheader("EOQ & Reorder Point Simulation", divider='blue')
#     st.markdown("Interactively calculate the Economic Order Quantity (EOQ) and Reorder Point (ROP).")

#     sim_col1, sim_col2 = st.columns(2)

#     with sim_col1:
#         annual_demand = st.number_input("Annual Demand (Units)", min_value=100, value=10000, step=100)
#         ordering_cost = st.number_input("Cost per Order ($)", min_value=1.0, value=50.0, step=5.0)
#         holding_cost = st.number_input("Annual Holding Cost per Unit ($)", min_value=0.1, value=5.0, step=0.5)
#         lead_time_days = st.slider("Supplier Lead Time (Days)", min_value=1, max_value=90, value=14)

#     daily_demand = annual_demand / 365
#     if ordering_cost > 0 and holding_cost > 0:
#         eoq = (2 * annual_demand * ordering_cost / holding_cost)**0.5
#     else:
#         eoq = 0
        
#     reorder_point = daily_demand * lead_time_days
    
#     with sim_col2:
#         st.metric("Economic Order Quantity (EOQ)", f"{eoq:,.0f} units", help="The optimal order size to minimize total inventory costs.")
#         st.metric("Reorder Point (ROP)", f"{reorder_point:,.0f} units", help="The inventory level at which a new order should be placed.")

#     st.write("This chart simulates the inventory cycle over 90 days based on the EOQ and ROP calculated above.")

#     if eoq > 0:
#         inventory_levels = []
#         current_inventory = eoq
#         for day in range(90):
#             inventory_levels.append({'day': day, 'inventory': current_inventory, 'level': 'Inventory Level'})
#             inventory_levels.append({'day': day, 'inventory': reorder_point, 'level': 'Reorder Point'})
#             current_inventory -= daily_demand
#             if current_inventory <= 0:
#                 current_inventory = eoq 

#         sim_df = pd.DataFrame(inventory_levels)

#         inventory_chart = alt.Chart(sim_df).mark_line(interpolate='step-after').encode(
#             x=alt.X('day:Q', title='Day'),
#             y=alt.Y('inventory:Q', title='Inventory Level (Units)'),
#             color=alt.Color('level:N', title='Metric', scale=alt.Scale(
#                 domain=['Inventory Level', 'Reorder Point']
#             ))
#         )
#         st.altair_chart(inventory_chart, use_container_width=True)
#     else:
#         st.warning("EOQ is zero. Cannot run simulation.")
