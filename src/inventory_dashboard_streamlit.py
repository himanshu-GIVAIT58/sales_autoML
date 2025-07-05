
import streamlit as st
import pandas as pd
from datetime import datetime
import os
import altair as alt
from dateutil.relativedelta import relativedelta


from src.model_handler import make_fast_predictions, prepare_prediction_data,load_latest_predictor,generate_future_covariates,generate_static_features,load_prediction_artifacts
from pymongo import MongoClient
from dotenv import load_dotenv
from src.data_loader import load_latest_recommendation_data, load_dataframe_from_mongo
from typing import Optional
from src import config 
from autogluon.timeseries import TimeSeriesDataFrame  

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


def display_sku_overview(sku_data: pd.DataFrame) -> None:
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

def display_demand_chart(sku_data: pd.DataFrame) -> None:
    """Visualizes the total forecasted demand across different horizons."""
    st.subheader("Demand Forecast Comparison", divider="gray")

    
    col1, col2 = st.columns(2)

    
    with col1:
        st.markdown("**Total Demand by Forecast Horizon**")
        
        
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
        st.markdown("**Implied Monthly Demand for Next 6 Months**")
        
        
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

def capture_feedback(selected_sku: str, sku_data: pd.DataFrame):
    """Displays feedback buttons and saves the feedback to MongoDB."""
    st.subheader("Was this forecast helpful?", divider="green")
    
    feedback_col1, feedback_col2 = st.columns(2)

    with feedback_col1:
        if st.button("👍 Good Forecast", use_container_width=True):
            save_feedback(selected_sku, sku_data, "Good")

    with feedback_col2:
        if st.button("👎 Bad Forecast", use_container_width=True):
            save_feedback(selected_sku, sku_data, "Bad")


st.title('📦 Inventory Recommendations Dashboard')


st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Recommendations", "Analyze Data", "New SKU Forecast"])

if page == "Recommendations":
    
    
    reco_df = load_recommendation_data_from_mongo()

    if reco_df is None or reco_df.empty:
        st.info("ℹ️ Awaiting data. Please ensure recommendations are available in MongoDB.", icon="⏳")
    else:
        
        with st.sidebar:
            st.header("⚙️ SKU Selection")
            sku_list = sorted(reco_df['item_id'].unique())
            search_term = st.text_input("Search SKU:", placeholder="Enter Item ID...")
            if search_term:
                filtered_sku_list = [sku for sku in sku_list if search_term.lower() in str(sku).lower()]
            else:
                filtered_sku_list = sku_list

            if not filtered_sku_list:
                st.warning("No SKUs found for your search term.")
                selected_sku = None
            else:
                selected_sku = st.radio("Select Item:", filtered_sku_list, index=0)

        
        if selected_sku:
            st.header(f"Analysis for Item: `{selected_sku}`", divider="rainbow")
            sku_data = reco_df[reco_df['item_id'] == selected_sku].sort_values('forecast_days')

            if sku_data.empty:
                st.error(f"No data found for the selected item: {selected_sku}")
            else:
                st.subheader(f"Overall 6-Month Total Forecast: {sku_data[sku_data['horizon'] == '6-Month']['total_forecasted_demand'].sum():,.0f} units")
                display_sku_overview(sku_data)
                display_demand_chart(sku_data)

                with st.expander("📋 View Raw Data"):
                    st.dataframe(sku_data, use_container_width=True, hide_index=True)

                display_chatbot(selected_sku, reco_df)
                st.markdown("---")
                capture_feedback(selected_sku, sku_data)
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
                        # Step 1: Load Model
                        status.write("1. Loading trained model...")
                        predictor = load_latest_predictor()
                        if predictor is None:
                            status.update(label="🚨 Error: Could not load model.", state="error")
                            st.error("Could not load a trained model. Please run the main training pipeline first.")
                            st.stop()

                        # Step 2: Load Artifacts
                        status.write("2. Loading training artifacts (features & holidays)...")
                        static_feature_columns, holidays_df = load_prediction_artifacts()

                        # Step 3: Prepare Data
                        status.write("3. Preparing and enriching uploaded data...")
                        enriched_data = prepare_prediction_data(user_data_raw, holidays_df)
                        
                        # Step 4: Create TimeSeries Structure
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

                        # Step 5: Evaluate Performance (with safety check)
                        status.write("5. Evaluating model performance on historical data...")
                        min_series_length = ts_upload.index.get_level_values('item_id').value_counts().min()
                        if min_series_length > predictor.prediction_length:
                            # --- KEY FIX IS HERE ---
                            # Assign the single output of predictor.evaluate() to the metrics variable.
                            metrics = predictor.evaluate(ts_upload, display=False)
                            status.write("   -> ✅ Performance evaluation complete.")
                        else:
                            reason = f"Uploaded data history ({min_series_length} points) is not longer than the model's prediction length ({predictor.prediction_length} points)."
                            metrics = pd.DataFrame([{"info": "Evaluation skipped", "reason": reason}])
                            status.write("   -> ⚠️ Performance evaluation skipped (data too short).")

                        # Step 6: Generate Future Forecast
                        status.write("6. Generating future forecast...")
                        future_known_covariates = generate_future_covariates(predictor, ts_upload, holidays_df)
                        predictions = predictor.predict(
                            ts_upload,
                            known_covariates=future_known_covariates
                        )
                        
                        status.update(label="✅ Process Complete!", state="complete", expanded=False)

                    except Exception as e:
                        status.update(label=f"🚨 Error: {e}", state="error")
                        st.error(f"An error occurred during forecast generation: {e}")
                        st.stop()
                
                # --- Display Results ---
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
