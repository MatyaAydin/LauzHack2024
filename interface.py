import streamlit as st
import requests
from dotenv import load_dotenv
import plotly.express as px
import os
from langchain_community.llms import OpenAI
import tempfile
import time
from sort.sort import Sort
from video_analyse import get_count_video
from pic_analyse import get_count
from prompt import get_prompt_request, get_prompt_answer,get_prompt_answer_image
load_dotenv()




page = st.sidebar.radio("Select a page", ["Home", "Request video","Request image"])
key = os.getenv("KEY")

llm = OpenAI(temperature=0, openai_api_key=key)

if page == "Home":
    st.title("VesselVision: Intelligent Port Monitoring")
    st.write("""
    ### Welcome to VesselVision

    **VesselVision** leverages cutting-edge computer vision technology to monitor and track boats within port areas. Our system accurately counts the number of vessels departing the port, providing valuable insights for port management and operational efficiency.

    - **Real-Time Tracking:** Continuously monitor boat movements with high precision.
    - **Automated Counting:** Seamlessly count the number of boats leaving the port.
    - **Data Analytics:** Gain actionable insights to optimize port operations.
    - **Enhanced Safety:** Improve maritime safety through accurate vessel monitoring.

    Explore how VesselVision can transform port management by integrating advanced computer vision solutions.
    """)
elif page == "Request video":
    st.title("Request video Page")
    with st.form(key='request_form'):
        usr_input = st.text_input("Enter your request here:")
        
        submit = st.form_submit_button("Submit")
        uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        file_path = tfile.name 
    if submit:
        if usr_input:
            st.success(f"Your request is: {usr_input}")
            prompt = get_prompt_request(usr_input)

            answer = llm.invoke(prompt)
            improve_answer = get_prompt_answer(prompt,answer)
            improve_answer = llm.invoke(improve_answer)
            
            with st.spinner('Processing your request...'):
                time.sleep(5)
            count = get_count_video(file_path)
            st.success(f"The number of boat is {count}")
            st.video("live_feed_5min.mp4")
            
        else:
            st.error("You didn't request anything.")
elif page == "Request image":
    st.title("Request image Page")
    with st.form(key='request_form_image'):
        usr_input = st.text_input("Enter your request here:")
        
        submit = st.form_submit_button("Submit")
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "pdf"])

    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        tfile_path = tfile.name
    if submit:
        if usr_input:
            st.success(f"Your request is: {usr_input}")
            #prompt = get_prompt_request(usr_input)

            
            improve_answer = get_prompt_answer_image(usr_input)
            improve_answer = llm.invoke(improve_answer).strip()
            print(f"The reformulated query is {improve_answer} and the type is {type(improve_answer)} of len {len(improve_answer)}")
            number = get_count(tfile_path,improve_answer)
            st.success(f"the number of {improve_answer} is {number}")
            
        else:
            st.error("You didn't request anything.")