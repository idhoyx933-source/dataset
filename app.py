import streamlit as st
import pandas as pd
import pickle
import re
import matplotlib.pyplot as plt
import altair as alt
from deep_translator import GoogleTranslator
from typing import Tuple, Any

# Constants
MODEL_PATH = 'model_sentiment_pilpres.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'

# Page Configuration
st.set_page_config(
    page_title="Presidential Election Sentiment Analysis",
    layout="wide"
)

@st.cache_resource
def ridho20221310028load_model_assets() -> Tuple[Any, Any]:
    """
    Loads the serialized machine learning model and TF-IDF vectorizer.
    """
    try:
        with open(MODEL_PATH, 'rb') as file_model:
            loaded_model = pickle.load(file_model)
        with open(VECTORIZER_PATH, 'rb') as file_vec:
            loaded_vec = pickle.load(file_vec)
        return loaded_model, loaded_vec
    except FileNotFoundError:
        return None, None

def ridho20221310028preprocess_text(text: str) -> str:
    """
    Cleans and normalizes the input text.
    """
    text = str(text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()

def ridho20221310028translate_to_english(text: str) -> str:
    """
    Translates input text to English using Google Translator.
    """
    try:
        translator = GoogleTranslator(source='auto', target='en')
        return translator.translate(text)
    except Exception:
        return text

def ridho20221310028render_sidebar() -> str:
    """
    Renders the sidebar navigation and model information.
    """
    with st.sidebar:
        st.title("Navigation")
        mode = st.radio("Select Mode:", ["Single Simulator", "Batch CSV Analysis"])

        st.divider()
        st.info(
            "Model Information:\n"
            "- Algorithm: Logistic Regression\n"
            "- Feature Extraction: TF-IDF\n"
            "- Context: 2024 Presidential Election"
        )
    return mode

def ridho20221310028render_single_mode(model: Any, vectorizer: Any):
    """
    Renders the interface for single text prediction.
    """
    st.title("Sentiment Analysis Simulator")
    st.markdown("Test the model by entering an opinion (Indonesian/English).")

    col1, col2 = st.columns([2, 1])

    with col1:
        user_input = st.text_area("Input Opinion:", height=150)

        if st.button("Analyze", type="primary"):
            if not user_input.strip():
                st.warning("Please enter text to analyze.")
                return

            with st.spinner("Processing..."):
                text_english = ridho20221310028translate_to_english(user_input)
                cleaned_text = ridho20221310028preprocess_text(text_english)
                vectorized_text = vectorizer.transform([cleaned_text])

                pred_class = model.predict(vectorized_text)[0]
                pred_proba = model.predict_proba(vectorized_text)[0]

                st.divider()
                st.subheader("Analysis Result")

                if pred_class == 1:
                    st.success("Sentiment: POSITIVE")
                else:
                    st.error("Sentiment: NEGATIVE")

                with st.expander("Translation Details"):
                    st.write(f"**Original:** {user_input}")
                    st.write(f"**Translated:** {text_english}")

                ridho20221310028render_probability_chart(col2, pred_proba)

def ridho20221310028render_probability_chart(column: Any, probabilities: list):
    """
    Renders the probability bar chart using Altair.
    """
    with column:
        st.write("#### Model Confidence")

        chart_data = pd.DataFrame({
            'Sentiment': ['Negative', 'Positive'],
            'Probability': [probabilities[0], probabilities[1]]
        })

        chart = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X('Sentiment', sort=None),
            y='Probability',
            color=alt.Color(
                'Sentiment',
                scale=alt.Scale(
                    domain=['Negative', 'Positive'],
                    range=['#FF4B4B', '#00CC96']
                ),
                legend=None
            ),
            tooltip=['Sentiment', alt.Tooltip('Probability', format='.1%')]
        ).properties(height=300)

        st.altair_chart(chart, use_container_width=True)
        st.caption(
            f"Positive: {probabilities[1]*100:.1f}% | "
            f"Negative: {probabilities[0]*100:.1f}%"
        )

def ridho20221310028render_batch_mode(model: Any, vectorizer: Any):
    """
    Renders the interface for batch CSV analysis.
    """
    st.title("Batch Sentiment Analysis")
    st.markdown("Upload a CSV file containing tweets to analyze sentiment distribution.")

    uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write(f"Data Loaded: {len(df)} rows.")

            possible_columns = ['text', 'tweet', 'content', 'tweet_normalized', 'clean_text']
            target_col = next((col for col in possible_columns if col in df.columns), None)

            if not target_col:
                st.error("Text column not found in CSV.")
                return

            if st.button("Start Analysis"):
                with st.spinner("Analyzing data..."):
                    df['processed_text'] = df[target_col].apply(
                        ridho20221310028preprocess_text
                    )

                    vectorized_data = vectorizer.transform(df['processed_text'])
                    df['prediction_label'] = model.predict(vectorized_data)
                    df['sentiment'] = df['prediction_label'].map(
                        {1: 'Positive', 0: 'Negative'}
                    )

                    st.divider()
                    col_chart, col_stats = st.columns(2)

                    with col_chart:
                        st.subheader("Sentiment Distribution")
                        counts = df['sentiment'].value_counts()

                        fig, ax = plt.subplots()
                        ax.pie(
                            counts,
                            labels=counts.index,
                            autopct='%1.1f%%',
                            startangle=90
                        )
                        ax.axis('equal')
                        st.pyplot(fig)

                    with col_stats:
                        st.subheader("Statistics")
                        st.metric("Total Rows", len(df))
                        st.metric("Positive Count", counts.get('Positive', 0))
                        st.metric("Negative Count", counts.get('Negative', 0))

                    st.subheader("Detailed Results")
                    st.dataframe(df[[target_col, 'sentiment']].head(100))

        except Exception as e:
            st.error(f"An error occurred: {e}")

def ridho20221310028main():
    """
    Application entry point.
    """
    model, vectorizer = ridho20221310028load_model_assets()

    if model is None or vectorizer is None:
        st.error("Model or vectorizer file not found.")
        return

    selected_mode = ridho20221310028render_sidebar()

    if selected_mode == "Single Simulator":
        ridho20221310028render_single_mode(model, vectorizer)
    else:
        ridho20221310028render_batch_mode(model, vectorizer)

    st.markdown("---")
    st.caption("Deep Learning Final Project - Sentiment Analysis 2024")

if __name__ == "__main__":
    ridho20221310028main()
