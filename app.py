import streamlit as st
import pandas as pd
from transformers import pipeline
import plotly.express as px
import io

st.set_page_config(page_title="Review Sentiment Analyzer", page_icon="📝", layout="wide")

if 'results' not in st.session_state:
    st.session_state.results = None

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", return_all_scores=True)

def analyze_text(text, classifier):
    if not text or pd.isna(text):
        return "neutral", 0.5
    
    text = str(text).strip()[:500]
    
    try:
        result = classifier(text)
        scores = {item['label'].lower(): item['score'] for item in result[0]}
        
        label_map = {'positive': 'positive', 'negative': 'negative', 'neutral': 'neutral'}
        best_label = max(scores.keys(), key=lambda x: scores[x])
        sentiment = label_map.get(best_label, 'neutral')
        confidence = scores[best_label]
        
        return sentiment, confidence
    except:
        return "neutral", 0.5

st.title("Customer Review Sentiment Analyzer")
st.write("Upload your CSV file to analyze review sentiments")

uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded {len(df)} reviews")
    
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    text_cols = [col for col in df.columns if df[col].dtype == 'object']
    
    if text_cols:
        selected_col = st.selectbox("Select review text column:", text_cols)
        
        if st.button("Analyze Sentiments", type="primary"):
            with st.spinner("Analyzing..."):
                model = load_model()
                
                results = []
                progress = st.progress(0)
                
                for i, row in df.iterrows():
                    progress.progress((i + 1) / len(df))
                    sentiment, confidence = analyze_text(row[selected_col], model)
                    
                    result_row = row.to_dict()
                    result_row['sentiment'] = sentiment
                    result_row['confidence'] = confidence
                    results.append(result_row)
                
                st.session_state.results = pd.DataFrame(results)
                progress.empty()
                st.success("Analysis complete!")

if st.session_state.results is not None:
    results_df = st.session_state.results
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reviews", len(results_df))
    with col2:
        positive = len(results_df[results_df['sentiment'] == 'positive'])
        st.metric("Positive", positive, f"{positive/len(results_df)*100:.1f}%")
    with col3:
        negative = len(results_df[results_df['sentiment'] == 'negative'])
        st.metric("Negative", negative, f"{negative/len(results_df)*100:.1f}%")
    with col4:
        neutral = len(results_df[results_df['sentiment'] == 'neutral'])
        st.metric("Neutral", neutral, f"{neutral/len(results_df)*100:.1f}%")
    
    tab1, tab2, tab3 = st.tabs(["Chart", "Results", "Download"])
    
    with tab1:
        sentiment_counts = results_df['sentiment'].value_counts()
        fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, 
                    title="Sentiment Distribution",
                    color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#f39c12'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        sentiment_filter = st.selectbox("Filter by sentiment:", ["All"] + list(results_df['sentiment'].unique()))
        
        if sentiment_filter != "All":
            filtered_df = results_df[results_df['sentiment'] == sentiment_filter]
        else:
            filtered_df = results_df
        
        st.dataframe(filtered_df, use_container_width=True)
    
    with tab3:
        csv_data = results_df.to_csv(index=False)
        st.download_button("Download Results", csv_data, "sentiment_results.csv", "text/csv")