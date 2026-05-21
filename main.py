import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import os
from dotenv import load_dotenv

from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(
    page_title='AI Data Analyst',
    page_icon='📊',
    layout='wide'
)

st.title('📈 AI Data Analysis Tool')
st.caption('Upload a CSV file and get AI-powered insights')

# ======================================
# OPENAI SETUP
# ======================================
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("⚠️ OpenAI API key not found in .env file!")
    st.stop()

client = OpenAI(
    api_key=openai_api_key
)

# ======================================
# AI FUNCTION
# ======================================
def ask_ai(prompt):

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


# ======================================
# FILE UPLOAD
# ======================================
uploaded_file = st.file_uploader(
    '📂 Upload CSV File',
    type=['csv']
)

# ======================================
# RUN APP ONLY IF FILE IS UPLOADED
# ======================================
if uploaded_file:

    # Read CSV File
    df = pd.read_csv(uploaded_file)

    # ======================================
    # SUMMARY METRICS
    # ======================================
    st.subheader('📊 Dataset Summary')

    c1, c2, c3 = st.columns(3)

    c1.metric(
        'Rows',
        df.shape[0]
    )

    c2.metric(
        'Columns',
        df.shape[1]
    )

    c3.metric(
        'Missing Values',
        int(df.isnull().sum().sum())
    )

    # ======================================
    # DATA PREVIEW
    # ======================================
    st.subheader('📋 Dataset Preview')

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    # ======================================
    # FIND NUMERIC COLUMNS
    # ======================================
    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    # ======================================
    # CHART GENERATOR
    # ======================================
    st.subheader('📈 Automatic Chart Generator')

    chart_type = st.selectbox(
        'Select Chart Type',
        ['Histogram', 'Box Plot', 'Heatmap', 'Scatter Plot']
    )

    # ======================================
    # HISTOGRAM
    # ======================================
    if chart_type == 'Histogram':

        if numeric_cols:

            col = st.selectbox(
                'Select Column',
                numeric_cols
            )

            fig = px.histogram(
                df,
                x=col,
                title=f'Histogram of {col}'
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # ======================================
    # BOX PLOT
    # ======================================
    elif chart_type == 'Box Plot':

        if numeric_cols:

            col = st.selectbox(
                'Select Column',
                numeric_cols
            )

            fig = px.box(
                df,
                y=col,
                title=f'Box Plot of {col}'
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # ======================================
    # HEATMAP
    # ======================================
    elif chart_type == 'Heatmap':

        if len(numeric_cols) > 1:

            corr = df[numeric_cols].corr()

            fig = px.imshow(
                corr,
                text_auto=True,
                title='Correlation Heatmap'
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:
            st.warning('Need at least 2 numeric columns for Heatmap.')

    # ======================================
    # SCATTER PLOT
    # ======================================
    elif chart_type == 'Scatter Plot':

        if len(numeric_cols) >= 2:

            x_col = st.selectbox(
                'X Axis',
                numeric_cols,
                index=0
            )

            y_col = st.selectbox(
                'Y Axis',
                numeric_cols,
                index=1
            )

            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                title=f'{x_col} vs {y_col}'
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:
            st.warning('Need at least 2 numeric columns for Scatter Plot.')

    # ======================================
    # AI AUTO ANALYSIS
    # ======================================
    st.divider()

    st.subheader('🤖 AI Auto Analysis')

    if st.button('Generate AI Insights'):

        with st.spinner('Analyzing Dataset...'):

            # Capture df.info()
            buffer = io.StringIO()

            df.info(buf=buffer)

            info_text = buffer.getvalue()

            # Create Prompt
            prompt = (
                "You are an expert data analyst.\n\n"

                "Dataset Information:\n"
                + info_text +

                "\n\nStatistical Summary:\n"
                + df.describe(include='all').to_string() +

                "\n\nPlease provide:\n"
                "1. What the dataset is about\n"
                "2. Key trends and patterns\n"
                "3. Outliers or anomalies\n"
                "4. Business recommendations\n"
                "5. Most important columns"
            )

            # Get AI Response
            insights = ask_ai(prompt)

            st.success(insights)

    # ======================================
    # CUSTOM AI QUESTIONS
    # ======================================
    st.divider()

    st.subheader('💬 Ask Custom Questions')

    custom_q = st.text_area(
        'Ask anything about your dataset:',
        placeholder='What is the average salary by department?'
    )

    if st.button('Ask AI') and custom_q:

        with st.spinner('Processing Question...'):

            prompt = (
                "Dataset Sample:\n"
                + df.head(20).to_string()

                + "\n\nColumns:\n"
                + str(list(df.columns))

                + "\n\nQuestion:\n"
                + custom_q
            )

            answer = ask_ai(prompt)

            st.info(answer)

else:

    st.info('👆 Upload a CSV file to start analysis.')