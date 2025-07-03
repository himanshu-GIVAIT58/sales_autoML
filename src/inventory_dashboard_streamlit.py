# --- Imports ---
import streamlit as st
import pandas as pd
from datetime import datetime
import os
import altair as alt
from dateutil.relativedelta import relativedelta

# Fix relative imports to absolute imports
from model_handler import make_predictions, load_latest_predictor
from pymongo import MongoClient
from dotenv import load_dotenv
from data_loader import load_latest_recommendation_data, load_dataframe_from_mongo
from typing import Optional

# --- Load environment variables ---
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://root:example@localhost:27017/")
MONGO_DB = os.getenv("MONGO_DB", "sales_automl")

MODEL_SAVE_PATH = "./autogluon_models/"
EDA_DIR = "./eda/target/"  # Updated path to match the actual structure

# --- Page Configuration ---
st.set_page_config(
    page_title="Inventory Recommendations Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- Data Loading ---
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

# --- MongoDB Feedback Save ---
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
        st.success("Thank you for your feedback! It has been recorded.")
    except Exception as e:
        st.error(f"⚠️ Could not save feedback to MongoDB: {str(e)}", icon="🚨")

# --- UI Helper Functions ---
def display_sku_overview(sku_data: pd.DataFrame) -> None:
    """Displays the key recommendation metrics for a selected SKU across all horizons."""
    st.subheader("Forecast Horizons & Recommendations", divider="blue")

    # Create a metric card for each forecast horizon
    channel_order = ['App', 'Web', 'Offline']
    ordered_data = sku_data[sku_data['channel'].isin(channel_order)].sort_values(by='channel', key=lambda x: x.map({channel: i for i, channel in enumerate(channel_order)}))

    num_metrics = len(ordered_data)
    cols = st.columns(3)  # Create 3 equal-width columns
    for i, (index, row) in enumerate(ordered_data.iterrows()):
        with cols[i % 3]:  # Use modulo to cycle through the columns
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

    # Create two columns for side-by-side layout
    col1, col2 = st.columns(2)

    # Current Demand Bar Chart (Aggregated by Horizon)
    with col1:
        st.markdown("### Total Demand by Horizon")
        
        # Aggregate data by horizon
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

    # Corrected Forecasted Demand Chart for the Next 6 Months
    with col2:
        st.markdown("### Implied Monthly Demand for Next 6 Months")
        
        # Aggregate the total demand per horizon
        demand_by_horizon = sku_data.groupby('horizon')['total_forecasted_demand'].sum()

        d1 = demand_by_horizon.get('1-Month', 0)
        d3 = demand_by_horizon.get('3-Month', 0)
        d6 = demand_by_horizon.get('6-Month', 0)

        # Calculate implied monthly demand
        monthly_demands = []
        # Month 1
        monthly_demands.append(d1)
        # Months 2-3
        demand_month_2_3 = (d3 - d1) / 2 if d3 > d1 else 0
        monthly_demands.extend([demand_month_2_3] * 2)
        # Months 4-6
        demand_month_4_6 = (d6 - d3) / 3 if d6 > d3 else 0
        monthly_demands.extend([demand_month_4_6] * 3)

        # Create future dates for x-axis labels
        future_dates = [(datetime.today() + relativedelta(months=i)).strftime('%Y-%m') for i in range(6)]

        # Create a DataFrame for the forecasted demand
        forecast_df = pd.DataFrame({
            'Month': future_dates,
            'Implied Monthly Demand': monthly_demands
        })

        # Create the forecasted demand chart
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

            # Mock response based on the prompt
            if "reorder point" in prompt.lower():
                # Aggregate reorder point data across channels for the 1-Month horizon
                reorder_point_data = full_df[(full_df['item_id'] == selected_sku) & (full_df['horizon'] == '1-Month')]
                if not reorder_point_data.empty:
                    # Sum reorder points across all channels for the SKU
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
    # Prepare the data to be saved
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
        # Save to MongoDB only
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

# --- Main Application ---
st.title('📦 Inventory Recommendations Dashboard')

# 1) Create a radio button in the sidebar to select a page
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Recommendations", "Analyze Data", "New SKU Forecast"])

if page == "Recommendations":
    # --- Main Content: Recommendations ---
    # Load data from MongoDB
    reco_df = load_recommendation_data_from_mongo()

    if reco_df is None or reco_df.empty:
        st.info("ℹ️ Awaiting data. Please ensure recommendations are available in MongoDB.", icon="⏳")
    else:
        # --- Sidebar for SKU Selection ---
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

        # --- Main Content Area ---
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
    if not os.path.isdir(EDA_DIR):
        os.makedirs(EDA_DIR, exist_ok=True)
        st.warning(f"Created EDA directory at: {EDA_DIR}")
        st.info("Please run the EDA script to generate charts.")
    else:
        charts = sorted([f for f in os.listdir(EDA_DIR) if f.endswith(".png")])
        if not charts:
            st.warning("No PNG charts found in the EDA directory. Please run the EDA script first.")
        else:
            for chart in charts:
                chart_path = os.path.join(EDA_DIR, chart)
                try:
                    st.image(chart_path, caption=chart, use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to load chart {chart}: {str(e)}")

elif page == "New SKU Forecast":
    st.subheader("📤 Upload Data for New SKU Forecasting")

    st.info("""
    **Required CSV format:**
    - `sku`: Product identifier
    - `timestamp`: Date (e.g., `2025-07-02` or `2025-07-02T02:54:20+05:30`)
    - `target`: Sales quantity (numeric)
    - `disc`: Discount percentage (numeric, e.g., `10.5`)
    """)

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        try:
            user_data = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview:")
            st.dataframe(user_data.head())

            required_columns = ["sku", "timestamp", "target", "disc"]
            if not all(col in user_data.columns for col in required_columns):
                st.error(f"Missing required columns. Please ensure your CSV has: {required_columns}")
            else:
                with st.spinner("Processing data and generating forecast..."):
                    try:
                        # Standardize timestamp format
                        user_data["timestamp"] = pd.to_datetime(user_data["timestamp"], format='mixed', utc=True).dt.tz_localize(None)

                        # Load forecasting model
                        predictor = load_latest_predictor(MODEL_SAVE_PATH)
                        if predictor is None:
                            st.error("🚨 Could not load pre-trained model. Please check the path and ensure it's available.")
                            st.stop()

                        # --- FIX STARTS HERE: Load and process holidays data ---
                        holidays_df = pd.DataFrame(columns=['timestamp', 'is_holiday']) # Default empty DF
                        try:
                            holidays_from_db = load_dataframe_from_mongo("holidays_data")
                            if not holidays_from_db.empty and 'Date' in holidays_from_db.columns:
                                holidays_df['timestamp'] = pd.to_datetime(holidays_from_db['Date'], format='mixed', utc=True).dt.tz_localize(None)
                                holidays_df['is_holiday'] = 1
                                holidays_df = holidays_df[['timestamp', 'is_holiday']].drop_duplicates()
                            else:
                                st.warning("Holiday data not found or is empty in MongoDB. Proceeding without it.")
                        except Exception as e:
                            st.warning(f"Could not load holiday data: {e}. Proceeding without it.")
                        # --- FIX ENDS HERE ---

                        # Generate predictions
                        predictions = make_predictions(
                            predictor=predictor,
                            user_uploaded_data=user_data,
                            holidays_df=holidays_df
                        )

                        st.success("✅ Forecasts generated successfully!")
                        st.dataframe(predictions)

                    except Exception as e:
                        st.error(f"Error during forecast generation: {str(e)}")
                        st.info("Please check model configuration and data formats.")

        except Exception as e:
            st.error(f"Error reading the uploaded file: {str(e)}")
            st.info("Please ensure your CSV file is properly formatted and not corrupt.")
