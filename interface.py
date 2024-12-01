import streamlit as st
import pandas as pd
import os
import time
from langchain_community.llms import OpenAI
import plotly.express as px  # Import Plotly for visualization

# Load environment variables
from dotenv import load_dotenv
load_dotenv()
key = os.getenv("KEY")

# Initialize the language model
llm = OpenAI(temperature=0, openai_api_key=key)

# Path to preprocessed CSV files
CSV_DIR = "./simulated_boat_data"

# Sidebar Navigation
st.sidebar.title("VesselVision Navigation")
page = st.sidebar.radio("Select a page", ["Analyze Media"])

# Home Page

# Analyze Media Page
if page == "Analyze Media":
    st.title("Analyze Vessel Data")
    st.write("**For demo purposes, you can type a query below to simulate analysis:**")
    demo_query = st.text_input("Type your analysis request here (e.g., 'Analyze vessel patterns on busy days')", "")

    st.write("Upload a video to simulate vessel data analysis.")

    # Initialize session state
    if "refine_analysis" not in st.session_state:
        st.session_state.refine_analysis = False
    if "trend_summary" not in st.session_state:
        st.session_state.trend_summary = None
    if "insight_summary" not in st.session_state:
        st.session_state.insight_summary = None
    if "analysis_started" not in st.session_state:
        st.session_state.analysis_started = False

    # Form for video upload
    with st.form(key='video_form'):
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mpeg4"])
        submit = st.form_submit_button("Submit")

    if submit and uploaded_file:
        st.session_state.analysis_started = True
        st.session_state.refine_analysis = False  # Reset refine analysis state
        st.session_state.trend_summary = None
        st.session_state.insight_summary = None

    # Handle video analysis
    if st.session_state.analysis_started:
        st.success("Video uploaded successfully!")
        st.write("Simulating analysis...")

        # Simulate processing time with a spinner
        with st.spinner("Analyzing the video..."):
            time.sleep(2)

        # Use preprocessed CSV file
        csv_file = "1_12-12-18.csv"  # Example CSV for the uploaded video
        csv_path = os.path.join(CSV_DIR, csv_file)

        if os.path.exists(csv_path):
            # Load CSV and count vessels
            vessel_data = pd.read_csv(csv_path)
            count = len(vessel_data)

            st.success(f"The number of vessels detected in the video is: **{count}**")

            st.write(f"Reading file: {csv_path}")

            # Ask the user if they want to refine analysis
            refine_analysis = st.radio(
                "Do you want to refine the analysis by looking at trends over previous days?",
                ("No", "Yes")
            )

            if refine_analysis == "Yes":
                st.session_state.refine_analysis = True

    # Handle trend analysis if "Yes" is selected
    if st.session_state.refine_analysis:
        st.write("Fetching data from all previous days...")

        # Load all CSVs in the folder
        previous_csvs = [
            os.path.join(CSV_DIR, f) for f in os.listdir(CSV_DIR) if f.endswith(".csv")
        ]

        valid_dataframes = []
        for csv_file in previous_csvs:
            try:
                # Check if the file is not empty
                if os.stat(csv_file).st_size > 0:
                    df = pd.read_csv(csv_file)
                    valid_dataframes.append(df)
                else:
                    st.warning(f"The file {os.path.basename(csv_file)} is empty and was skipped.")
            except pd.errors.EmptyDataError:
                st.warning(f"The file {os.path.basename(csv_file)} is malformed and was skipped.")
            except Exception as e:
                st.error(f"An error occurred while reading {os.path.basename(csv_file)}: {e}")

        if valid_dataframes:

            all_data = pd.concat(valid_dataframes, ignore_index=True)


            if 'timestamp_in' in all_data.columns:
                all_data['timestamp_in'] = pd.to_datetime(all_data['timestamp_in'], unit='s', origin='unix', errors='coerce')
                all_data['day'] = all_data['timestamp_in'].dt.date  # Extract day for grouping


                summary_data = all_data.groupby('day').agg(
                    boats_per_day=('ID', 'count'),
                    avg_duration=('duration', 'mean'),  # Duration remains in seconds
                    total_up=('direction', lambda x: (x == 'up').sum()),
                    total_down=('direction', lambda x: (x == 'down').sum())
                ).reset_index()


                summary_text = (
                    f"In the last 7 days, there were an average of {summary_data['boats_per_day'].mean():.1f} vessels per day. "
                    f"The average duration of a vessel in the port is {summary_data['avg_duration'].mean():.1f} seconds. "  # Updated to seconds
                    f"On average, {summary_data['total_up'].mean():.1f} vessels moved upstream, while {summary_data['total_down'].mean():.1f} "
                    "vessels moved downstream each day. "
                )
                
                if st.session_state.trend_summary is None:
                    st.session_state.trend_summary = llm.invoke(summary_text)
            

                trend_summary_prompt = (
                    f"The following data summarizes vessel activity in the Rotterdam port over multiple days:\n\n{summary_text}"
                    "Based on this data, provide insights relevant to commercial port operations, such as cargo movement patterns, "
                    "peak times, or traffic trends. Note: The duration values are in seconds."
                )

                if st.session_state.trend_summary is None:
                    st.session_state.trend_summary = llm.invoke(trend_summary_prompt)

                trend_summary = llm.invoke(trend_summary_prompt)
                st.write(trend_summary)  # Debugging step

                
            else:
                st.error("The data is missing the 'timestamp_in' column. Cannot perform trend analysis.")


        if st.session_state.trend_summary:
            st.write("**Trend Analysis:**")
            st.write(st.session_state.trend_summary)
