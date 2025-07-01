# --- Imports ---
import streamlit as st
import pandas as pd
import altair as alt
from typing import Optional, Dict, Any, List
import os
from io import StringIO
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

# --- Page Configuration ---
st.set_page_config(
    page_title="Inventory Recommendations Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants & Data ---
CSV_FILEPATH = "inventory_recommendations.csv"
FEEDBACK_FILEPATH = "feedback_data.csv"

# --- Data Loading ---
@st.cache_data(show_spinner="Loading recommendation data...")
def load_recommendation_data(filepath: str) -> Optional[pd.DataFrame]:
    """Loads and processes the recommendation data from a CSV file."""
    try:
        # Create a dummy CSV if it doesn't exist for demonstration purposes
        if not os.path.exists(filepath):
            data = {
                'item_id': ['SKU001', 'SKU001', 'SKU001', 'SKU002', 'SKU002', 'SKU002', 'SKU003', 'SKU003', 'SKU003'],
                'channel': ['App', 'Web', 'Offline', 'App', 'Web', 'Offline', 'App', 'Web', 'Offline'],
                'horizon': ['1-Month', '3-Month', '6-Month', '1-Month', '3-Month', '6-Month', '1-Month', '3-Month', '6-Month'],
                'forecast_days': [30, 90, 180, 30, 90, 180, 30, 90, 180],
                'total_forecasted_demand': [100, 310, 650, 150, 460, 980, 80, 250, 500],
                'safety_stock': [10, 30, 60, 15, 45, 90, 8, 24, 48],
                'reorder_point': [20, 60, 120, 30, 90, 180, 16, 48, 96],
                'economic_order_quantity (EOQ)': [50, 150, 300, 75, 225, 450, 40, 120, 240]
            }
            dummy_df = pd.DataFrame(data)
            dummy_df.to_csv(filepath, index=False)

        df = pd.read_csv(filepath)
        # Ensure the column name is correctly handled if it already exists
        if 'economic_order_quantity (EOQ)' in df.columns:
            df.rename(columns={'economic_order_quantity (EOQ)': 'eoq'}, inplace=True)
        return df
    except FileNotFoundError:
        st.error(f"⚠️ File not found: `{filepath}`. Please ensure this file is in the same directory as your script.", icon="🚨")
        return None
    except Exception as e:
        st.error(f"⚠️ An error occurred while loading data: {str(e)}", icon="🚨")
        return None

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
            st.markdown(f"**Order Quantity (EOQ):** `{row['eoq']}`")

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
    """Saves the user feedback to a CSV file."""
    # Prepare the data to be saved
    feedback_records = []
    for index, row in sku_data.iterrows():
        feedback_records.append({
            "timestamp": datetime.now(),
            "selected_sku": selected_sku,
            "feedback": feedback,
            "horizon": row['horizon'],
            "channel": row['channel'],
            "total_forecasted_demand": row['total_forecasted_demand'],
            "safety_stock": row['safety_stock'],
            "reorder_point": row['reorder_point'],
            "eoq": row['eoq']
        })
    
    new_feedback_df = pd.DataFrame(feedback_records)

    try:
        if os.path.exists(FEEDBACK_FILEPATH):
            # Append to existing file
            feedback_df = pd.read_csv(FEEDBACK_FILEPATH)
            combined_df = pd.concat([feedback_df, new_feedback_df], ignore_index=True)
            combined_df.to_csv(FEEDBACK_FILEPATH, index=False)
        else:
            # Create a new file
            new_feedback_df.to_csv(FEEDBACK_FILEPATH, index=False)
        st.success("Thank you for your feedback! It has been recorded.")
    except Exception as e:
        st.error(f"⚠️ An error occurred while saving feedback: {str(e)}", icon="🚨")

def capture_feedback(selected_sku: str, sku_data: pd.DataFrame):
    """Displays feedback buttons and saves the feedback to a CSV file."""
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

# Load data from the CSV file
reco_df = load_recommendation_data(CSV_FILEPATH)

if reco_df is None or reco_df.empty:
    st.info("ℹ️ Awaiting data. Please ensure `inventory_recommendations.csv` is in the correct directory.", icon="⏳")
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
            # Calculate the overall total forecast for the selected SKU for the 6-month horizon
            total_forecast = sku_data[sku_data['horizon'] == '6-Month']['total_forecasted_demand'].sum()
            
            # Display the overall total forecast
            st.subheader(f"Overall 6-Month Total Forecast: {total_forecast:,.0f} units", 
                         help="This is the total forecasted demand across all channels for the selected SKU over the 6-month horizon.")

            # Display the main metrics and charts
            display_sku_overview(sku_data)
            st.markdown("<br>", unsafe_allow_html=True)  # Spacer
            display_demand_chart(sku_data)

            # Display the raw data in a tab
            with st.expander("📋 View Raw Data"):
                st.dataframe(sku_data, use_container_width=True, hide_index=True)

            # Display the chatbot, passing the full dataframe for context
            display_chatbot(selected_sku, reco_df)
            
            # Add a divider before the feedback section
            st.markdown("---")

            # Display the feedback section
            capture_feedback(selected_sku, sku_data)
    else:
        st.info("Please select an item from the sidebar to see the detailed analysis.", icon="👈")
