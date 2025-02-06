import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from wordcloud import WordCloud  # We don't use built-in STOPWORDS anymore.
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords as nltk_stopwords
import plotly.express as px
import networkx as nx
import plotly.graph_objects as go
from io import BytesIO
import os
import pickle
from openai import OpenAI

# Download required NLTK resources
nltk.download('vader_lexicon')
nltk.download('stopwords')

# Set up OpenAI API key from environment variable or Streamlit secrets
api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
client = OpenAI(api_key=api_key)

# =============================================================================
# Data Loading, Validation, Mapping, and Cleaning Functions
# =============================================================================

@st.cache_data
def load_csv(file):
    """Lädt eine CSV-Datei und gibt ein DataFrame zurück."""
    try:
        return pd.read_csv(file)
    except Exception as e:
        st.error(f"Fehler beim Laden der CSV-Datei: {e}")
        return None

@st.cache_data(show_spinner=False)
def load_pickle(file_path):
    """Lädt einen Pickle-Datensatz von file_path."""
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Fehler beim Laden des Pickle-Datensatzes '{file_path}': {e}")
        return None

def validate_video_data(df):
    expected_columns = ['Video-ID', 'Titel', 'Upload-Datum', 'Views', 'Likes', 'Kommentare']
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        st.error("Die Video CSV fehlt die folgenden erforderlichen Spalten: " + ", ".join(missing))
        return False
    return True

def validate_comments_data(df):
    expected_columns = ['Video-ID', 'User-ID', 'Kommentar', 'Datum']
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        st.error("Die Kommentare CSV fehlt die folgenden erforderlichen Spalten: " + ", ".join(missing))
        return False
    return True

def data_summary(df):
    summary = pd.DataFrame({
        'Spalte': df.columns,
        'Datentyp': df.dtypes,
        'Fehlende Werte': df.isnull().sum()
    })
    return summary

def clean_video_data(df):
    df = df.drop_duplicates().copy()
    if 'Upload-Datum' in df.columns:
        df['Upload-Datum'] = pd.to_datetime(df['Upload-Datum'], errors='coerce')
    for col in ['Views', 'Likes', 'Kommentare']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def clean_comments_data(df):
    df = df.drop_duplicates().copy()
    if 'Datum' in df.columns:
        df['Datum'] = pd.to_datetime(df['Datum'], errors='coerce')
    return df

def map_video_columns(df):
    mapping = {
        "id": "Video-ID",
        "title": "Titel",
        "upload_date": "Upload-Datum",
        "view_count": "Views",
        "like_count": "Likes",
        "comment_count": "Kommentare"
    }
    available_mapping = {old: new for old, new in mapping.items() if old in df.columns}
    return df.rename(columns=available_mapping)

def map_comments_columns(df):
    mapping = {
        "video_id": "Video-ID",
        "author": "User-ID",
        "text": "Kommentar",
        "timestamp": "Datum",
        "likes": "Likes"
    }
    available_mapping = {old: new for old, new in mapping.items() if old in df.columns}
    return df.rename(columns=available_mapping)

def prepare_video_data(file):
    df = load_csv(file)
    if df is not None:
        df = map_video_columns(df)
        if validate_video_data(df):
            df = clean_video_data(df)
            return df
    return None

def prepare_comments_data(file):
    df = load_csv(file)
    if df is not None:
        df = map_comments_columns(df)
        if validate_comments_data(df):
            df = clean_comments_data(df)
            return df
    return None

# =============================================================================
# Analysis Functions and Customizable Components
# =============================================================================

def generate_wordcloud(text, custom_stopwords="", min_word_length=0):
    """
    Generates a WordCloud using German stopwords and additional custom stopwords.
    Returns the WordCloud object, the set of active stopwords, and the filtered text.
    """
    german_stopwords = set(nltk_stopwords.words("german"))
    additional_stopwords = set([w.strip().lower() for w in custom_stopwords.split(",") if w.strip()])
    active_stopwords = german_stopwords.union(additional_stopwords)
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    words = cleaned_text.split()
    filtered_words = [
        word for word in words
        if len(word) >= min_word_length and word.lower() not in active_stopwords
    ]
    filtered_text = " ".join(filtered_words)
    wc = WordCloud(width=800, height=400, background_color='white').generate(filtered_text)
    return wc, active_stopwords, filtered_text

def perform_sentiment_analysis(comments_df):
    sia = SentimentIntensityAnalyzer()
    if 'Kommentar' not in comments_df.columns:
        st.error("Spalte 'Kommentar' nicht gefunden in den Kommentardaten.")
        return comments_df
    comments_df['sentiment_score'] = comments_df['Kommentar'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
    def classify(score):
        if score >= 0.05:
            return 'Positiv'
        elif score <= -0.05:
            return 'Negativ'
        else:
            return 'Neutral'
    comments_df['sentiment'] = comments_df['sentiment_score'].apply(classify)
    return comments_df

def plot_engagement(df):
    fig = px.scatter(df, x='Views', y='Kommentare', size='Likes',
                     hover_data=['Titel'] if 'Titel' in df.columns else None,
                     title="Engagement: Views vs. Kommentare")
    return fig

def rank_top_videos(df, view_weight=1, like_weight=2, comment_weight=3):
    df = df.copy()
    df['engagement_score'] = df['Views'] * view_weight + df['Likes'] * like_weight + df['Kommentare'] * comment_weight
    top10 = df.sort_values('engagement_score', ascending=False).head(10)
    return top10

def perform_network_analysis(comments_df):
    G = nx.Graph()
    for vid, group in comments_df.groupby('Video-ID'):
        users = list(group['User-ID'].unique())
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                if G.has_edge(users[i], users[j]):
                    G[users[i]][users[j]]['weight'] += 1
                else:
                    G.add_edge(users[i], users[j], weight=1)
    return G

def download_figure(fig, filename="figure.png"):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    st.download_button(label="Download Bild", data=buf, file_name=filename, mime="image/png")

# ============================================================================
# Prompt Generation and Download Formatting Functions
# ============================================================================

def generate_llm_prompt(text, video_title="", analysis_mode="default", keywords=None, comment_count=None, max_length=30000):
    if len(text) > max_length:
        text = text[:max_length] + "..."
    keywords_str = ", ".join(keywords) if keywords else ""
    if analysis_mode == "selected":
        prompt = (
            f"Bitte führe eine detaillierte qualitative Analyse der folgenden Video-Kommentare durch. "
            f"Das Video heißt '{video_title}'. Es wurden {comment_count} Kommentare gefunden, die basierend auf den Schlüsselwörtern '{keywords_str}' gefiltert wurden. "
            f"Analysiere die wichtigsten Themen, Stimmungen und Erkenntnisse und fasse sie prägnant zusammen. Antworte nicht zu ausgeglichen, sondern mit eindeutigen und klar benannten Ergebnissen. "
            f"Hier sind die Kommentare:\n\n{text}\n\nZusammenfassung:"
        )
    elif analysis_mode == "global":
        prompt = (
            f"Bitte führe eine umfassende qualitative Analyse der folgenden Video-Kommentare aus allen Videos durch. "
            f"Es wurden {comment_count} Kommentare gefunden, die mit den Schlüsselwörtern '{keywords_str}' gefiltert wurden. "
            f"Analysiere die wichtigsten Themen, Stimmungen und Erkenntnisse und fasse sie prägnant zusammen. Antworte nicht zu ausgeglichen, sondern mit eindeutigen und klar benannten Ergebnissen. "
            f"Hier sind die Kommentare:\n\n{text}\n\nZusammenfassung:"
        )
    else:
        prompt = (
            f"Bitte führe eine qualitative Analyse der folgenden Video-Kommentare durch. "
            f"Das Video heißt '{video_title}'. Fasse die wichtigsten Themen, Stimmungen und Erkenntnisse zusammen. Antworte nicht zu ausgeglichen, sondern mit eindeutigen und klar benannten Ergebnissen. "
            f"Hier sind die Kommentare:\n\n{text}\n\nZusammenfassung:"
        )
    return prompt

def perform_llm_analysis(text, video_title="", analysis_mode="default", keywords=None, comment_count=None, max_length=30000, custom_prompt=None, max_tokens=1000):
    if custom_prompt is None:
        prompt = generate_llm_prompt(text, video_title, analysis_mode, keywords, comment_count, max_length)
    else:
        prompt = custom_prompt
    try:
        with st.spinner("GPT-4 Analyse läuft..."):
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Du bist ein hilfsbereiter Analyse-Assistent, der Video-Kommentare qualitativ auswertet. Du antwortest sehr konkret und eindeutig. Du belegst deine Ergebnisse mit Zitaten."},
                    {"role": "user", "content": prompt},
                ],
                model="gpt-4o",
                temperature=0.7,
                max_tokens=max_tokens,
            )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Fehler bei der LLM-Analyse: {e}")
        return "Fehler bei der LLM-Analyse"

def format_download_result(analysis_type, used_prompt, input_text, analysis_result):
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output = (
        f"Analysis Type: {analysis_type}\n"
        f"Date/Time: {timestamp}\n\n"
        f"Used Prompt:\n{used_prompt}\n\n"
        f"Input Text (Comments and Metrics):\n{input_text}\n\n"
        f"GPT-4 Analysis Result:\n{analysis_result}"
    )
    return output

# ============================================================================
# Functions for Saving and Loading Quantitative Analysis Results
# ============================================================================

def save_quantitative_results(results):
    return pickle.dumps(results)

def load_quantitative_results(file_obj):
    try:
        results = pickle.load(file_obj)
        return results
    except Exception as e:
        st.error("Fehler beim Laden der quantitativen Ergebnisse: " + str(e))
        return None

# =============================================================================
# Additional Filtering and Analysis Functions
# =============================================================================

def filter_comments(comments_df, username_filter="", start_date=None, end_date=None):
    df = comments_df.copy()
    if username_filter:
        df = df[df['User-ID'].fillna('').str.lower().str.contains(username_filter.lower())]
    if start_date:
        df = df[df['Datum'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['Datum'] <= pd.to_datetime(end_date)]
    return df

def analyze_comments_per_account(comments_df):
    count_df = comments_df.groupby('User-ID').size().reset_index(name='Kommentaranzahl')
    return count_df.sort_values('Kommentaranzahl', ascending=False)

def analyze_sentiments_per_account(comments_df):
    sentiment_df = comments_df.groupby(['User-ID', 'sentiment']).size().unstack(fill_value=0).reset_index()
    return sentiment_df

def compare_sentiments_to_views(video_df, comments_df):
    sentiment_video = comments_df.groupby(['Video-ID', 'sentiment']).size().unstack(fill_value=0).reset_index()
    merged_df = pd.merge(video_df, sentiment_video, on='Video-ID', how='left')
    for sentiment in ['Positiv', 'Negativ', 'Neutral']:
        if sentiment not in merged_df.columns:
            merged_df[sentiment] = 0
    merged_df['Positiv/Views'] = merged_df['Positiv'] / merged_df['Views'].replace({0: np.nan})
    merged_df['Negativ/Views'] = merged_df['Negativ'] / merged_df['Views'].replace({0: np.nan})
    return merged_df

# =============================================================================
# Streamlit App – Pages and User Interaction
# =============================================================================

st.title("Analyse des YouTube-Kanals 'unbubble'")

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if 'video_df' not in st.session_state:
    st.session_state.video_df = None
if 'comments_df' not in st.session_state:
    st.session_state.comments_df = None
if 'sentiment_analysis' not in st.session_state:
    st.session_state.sentiment_analysis = None
if 'quant_results' not in st.session_state:
    st.session_state.quant_results = {}
if 'wordcloud_title_fig' not in st.session_state:
    st.session_state.wordcloud_title_fig = None
if 'wordcloud_comments_fig' not in st.session_state:
    st.session_state.wordcloud_comments_fig = None

# -----------------------------------------------------------------------------
# Pre-calculate and load standard data files
# -----------------------------------------------------------------------------
default_video_file = "youtube_data/videos_detailed_with_comments.csv"
if st.session_state.video_df is None:
    if os.path.exists(default_video_file):
        video_df_default = prepare_video_data(default_video_file)
        if video_df_default is not None:
            st.session_state.video_df = video_df_default
            st.info("Standard Video-Metadaten aus 'videos_detailed_with_comments.csv' geladen.")
    else:
        st.error("Standard Video-Datensatz nicht gefunden.")

default_comments_file = "youtube_data/comments_with_sentiment.csv"
if st.session_state.comments_df is None:
    if os.path.exists(default_comments_file):
        comments_df_default = prepare_comments_data(default_comments_file)
        if comments_df_default is not None:
            st.session_state.comments_df = comments_df_default
            st.info("Standard Kommentardaten aus 'comments_with_sentiment.csv' geladen.")
    else:
        st.error("Standard Kommentardatensatz nicht gefunden.")

default_quant_file = "youtube_data/quant_results.pkl"
if not st.session_state.quant_results:
    if os.path.exists(default_quant_file):
        try:
            with open(default_quant_file, "rb") as f:
                quant_results_default = pickle.load(f)
            st.session_state.quant_results = quant_results_default
            st.info("Standard quantitative Analyseergebnisse aus 'quant_results.pkl' geladen.")
        except Exception as e:
            st.error(f"Fehler beim Laden der quantitativen Ergebnisse: {e}")
    else:
        st.info("Kein Standard quantitativer Ergebnis-Datensatz gefunden.")

if st.session_state.sentiment_analysis is None and st.session_state.quant_results:
    if "sentiment_df" in st.session_state.quant_results and isinstance(st.session_state.quant_results["sentiment_df"], pd.DataFrame):
        st.session_state.sentiment_analysis = st.session_state.quant_results["sentiment_df"]
        # st.info("Sentiment-Analyse aus quant_results geladen.")

# -----------------------------------------------------------------------------
# Sidebar Navigation – Two Pages Only
# -----------------------------------------------------------------------------
page = st.sidebar.selectbox("Navigation", 
    ["Dashboard & Metrics", "LLM Comment Analysis"]
)

# ----- Page: Dashboard & Metrics -----
if page == "Dashboard & Metrics":
    st.header("Dashboard & Metrics")
    if st.session_state.video_df is None or st.session_state.comments_df is None:
        st.warning("Standard-Daten wurden nicht geladen.")
    else:
        video_df = st.session_state.video_df
        comments_df = st.session_state.comments_df
        
        # Additional KPIs and Metrics
        total_videos = len(video_df)
        total_views = video_df['Views'].sum() if 'Views' in video_df.columns else 0
        total_likes = video_df['Likes'].sum() if 'Likes' in video_df.columns else 0
        total_comments = video_df['Kommentare'].sum() if 'Kommentare' in video_df.columns else 0
        avg_views = total_views / total_videos if total_videos > 0 else 0
        avg_likes = total_likes / total_videos if total_videos > 0 else 0
        avg_comments = total_comments / total_videos if total_videos > 0 else 0
        
        # Calculate engagement score (example: Views + 2*Likes + 3*Comments)
        video_df = video_df.copy()
        video_df['engagement_score'] = video_df['Views'] + 2 * video_df['Likes'] + 3 * video_df['Kommentare']
        avg_engagement = video_df['engagement_score'].mean()
        top_videos = rank_top_videos(video_df)
        
        # Ensure sentiment analysis is performed
        if st.session_state.sentiment_analysis is None or not isinstance(st.session_state.sentiment_analysis, pd.DataFrame):
            comments_df = perform_sentiment_analysis(comments_df)
            st.session_state.sentiment_analysis = comments_df.copy()
        else:
            comments_df = st.session_state.sentiment_analysis
        sentiment_counts = comments_df['sentiment'].value_counts()
        
        st.subheader("Key Performance Indicators (KPIs)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Videos", total_videos)
        col2.metric("Total Views", total_views)
        col3.metric("Total Likes", total_likes)
        col4.metric("Total Comments", total_comments)
        
        col5, col6, col7 = st.columns(3)
        col5.metric("Avg. Views", f"{avg_views:.0f}")
        col6.metric("Avg. Likes", f"{avg_likes:.0f}")
        col7.metric("Avg. Comments", f"{avg_comments:.0f}")
        
        st.metric("Avg. Engagement Score", f"{avg_engagement:.0f}")
        
        st.subheader("Sentiment Analysis")
        st.write("Sentiment Counts:")
        # Get the value counts and assign a name to the count values
        sentiment_counts = comments_df['sentiment'].value_counts()
        sentiment_counts.name = 'count'

        # Reset the index to turn it into a DataFrame and rename columns appropriately
        sentiment_counts_df = sentiment_counts.reset_index()
        sentiment_counts_df.rename(columns={'index': 'sentiment'}, inplace=True)

        st.write(sentiment_counts_df)

        # Use the new column names in the bar chart
        fig_sent = px.bar(
            sentiment_counts_df,
            x='sentiment',
            y='count',
            labels={'sentiment': 'Sentiment', 'count': 'Count'},
            title="Sentiment Distribution"
        )
        st.plotly_chart(fig_sent)
        
        st.subheader("Video Titles Wordcloud")
        custom_stopwords = st.text_input("Zusätzliche Stopwords (getrennt durch Komma)", 
                                         value="unbubble, shorts, 13 Fragen, Sollten, trifft, Sags, Sag's, sag, Fragen, unfiltered, live, 13", 
                                         key="wc_stopwords")
        min_word_length = st.number_input("Minimale Wortlänge", min_value=1, value=2, key="wc_min_length")
        if st.button("Generate Wordcloud for Video Titles", key="wc_generate"):
            title_text = " ".join(video_df['Titel'].dropna().astype(str))
            wc_title, active_stopwords_title, filtered_text = generate_wordcloud(title_text, custom_stopwords, min_word_length)
            fig_wc_title, ax_title = plt.subplots(figsize=(10, 5))
            ax_title.imshow(wc_title, interpolation="bilinear")
            ax_title.axis("off")
            st.pyplot(fig_wc_title)
            st.write("Active Stopwords:", sorted(active_stopwords_title))
        
        st.subheader("Engagement Analysis")
        fig_eng = plot_engagement(video_df)
        st.plotly_chart(fig_eng)
        
        st.subheader("Top 10 Videos by Engagement")
        st.dataframe(top_videos)
        csv_top10 = top_videos.to_csv(index=False).encode("utf-8")
        st.download_button("Download Top 10 Videos CSV", csv_top10, "top10_videos.csv", "text/csv")
        
        st.subheader("Daten Export")
        csv_video = video_df.to_csv(index=False).encode("utf-8")
        csv_comments = comments_df.to_csv(index=False).encode("utf-8")
        col_exp1, col_exp2 = st.columns(2)
        col_exp1.download_button("Download Video CSV", csv_video, "video_data.csv", "text/csv")
        col_exp2.download_button("Download Kommentare CSV", csv_comments, "comments_data.csv", "text/csv")

# ----- Page: LLM Comment Analysis -----
elif page == "LLM Comment Analysis":
    st.header("LLM Comment Analysis")
    if st.session_state.video_df is None or st.session_state.comments_df is None:
        st.warning("Standard-Daten wurden nicht geladen.")
    else:
        video_df = st.session_state.video_df
        comments_df = st.session_state.comments_df
        
        st.subheader("Analyse eines ausgewählten Videos")
        if "Titel" in video_df.columns and "Video-ID" in video_df.columns:
            selected_video_title = st.selectbox("Wähle ein Video zur Analyse", video_df["Titel"].tolist(), key="video_select")
            video_id = video_df.loc[video_df["Titel"] == selected_video_title, "Video-ID"].iloc[0]
        else:
            selected_video_title = st.selectbox("Wähle ein Video zur Analyse", video_df.index.tolist(), key="video_select")
            video_id = selected_video_title
        
        video_comments = comments_df[comments_df['Video-ID'] == video_id]
        all_comments_with_metrics = "\n".join(
            video_comments.apply(lambda row: f"User: {row['User-ID']} | Likes: {row['Likes']} | Comment: {row['Kommentar']}", axis=1)
        )
        project_info_all = (
            "Projekt: Qualitative Analyse von YouTube-Kommentaren.\n"
            "Aufgabe: Analysiere die Kommentare unter Berücksichtigung aller relevanten Metriken (Benutzername, Likes, Kommentartext)."
        )
        all_comments_input = project_info_all + "\n\n" + all_comments_with_metrics
        
        st.subheader("Analyse aller Kommentare des ausgewählten Videos")
        st.write(f"Gesamtanzahl Kommentare: {len(video_comments)}")
        edit_prompt_all = st.checkbox("Prompt bearbeiten (Gesamte Kommentare)", key="edit_prompt_all")
        if edit_prompt_all:
            default_prompt_all = generate_llm_prompt(
                all_comments_input,
                selected_video_title,
                "selected",
                [],
                comment_count=len(video_comments)
            )
            custom_prompt_all = st.text_area("Bearbeite den Prompt (Alle Kommentare)", value=default_prompt_all, height=200, key="custom_prompt_all")
        else:
            custom_prompt_all = None
        if st.button("Analyse ALLE Kommentare mit GPT-4", key="llm_all_comments"):
            used_prompt_all = custom_prompt_all if custom_prompt_all is not None else generate_llm_prompt(
                all_comments_input, selected_video_title, "selected", [], comment_count=len(video_comments)
            )
            result_all = perform_llm_analysis(
                all_comments_input,
                video_title=selected_video_title,
                analysis_mode="selected",
                keywords=[],
                comment_count=len(video_comments),
                custom_prompt=custom_prompt_all,
                max_tokens=2048
            )
            st.write(result_all)
            download_text = format_download_result("Analyse aller Kommentare", used_prompt_all, all_comments_input, result_all)
            st.download_button("Download vollständige Video GPT-4 Ergebnisse", download_text, "llm_video_all_analysis.txt", "text/plain")
        
        st.markdown("---")
        st.subheader("Filterung (optional)")
        filter_mode = st.radio("Filtermodus auswählen", options=["Keyword", "Username"], index=0)
        if filter_mode == "Keyword":
            keyword_groups = {
                "Akademiker": ["Akademiker", "Uni", "Studenten", "Akademisch", "Hochschule", "Professor", "Forschung"],
                "Gäste": ["Gast", "Gäste", "Auswahl der Gäste", "Interview", "Talkshow", "Diskussion"],
                "AFD": ["Alternative für Deutschland", "afd", "adf", "Alternative", "Rechtspartei", "Politik"],
                "Politisch": ["rechts", "links", "afd", "cdu", "spd", "grüne", "die grünen", "olaf scholz", "habeck", "merkel", "wahl", "regierung", "parlament"]
            }
            custom_keywords = st.text_input("Geben Sie benutzerdefinierte Schlüsselwörter ein (getrennt durch Komma)", key="custom_kw")
            if custom_keywords:
                keyword_groups["Benutzerdefiniert"] = [kw.strip() for kw in custom_keywords.split(",") if kw.strip()]
            selected_group = st.selectbox("Wähle eine Schlüsselwortgruppe", list(keyword_groups.keys()), key="keyword_group")
            keywords = keyword_groups[selected_group]
            st.write(f"Schlüsselwörter: {', '.join(keywords)}")
            keywords = [kw.strip().lower() for kw in keywords]
            filtered_comments = video_comments[video_comments['Kommentar'].str.lower().apply(lambda x: any(kw in x for kw in keywords))]
        else:
            username_filter_qual = st.text_input("Filter nach Benutzername (User-ID)", value="", key="username_qual")
            if username_filter_qual:
                filtered_comments = video_comments[video_comments['User-ID'].fillna('').str.lower().str.contains(username_filter_qual.lower())]
            else:
                filtered_comments = video_comments
        
        st.write(f"Anzahl der gefilterten Kommentare: {len(filtered_comments)}")
        st.subheader("Gefilterte Kommentare zur Analyse")
        if not filtered_comments.empty:
            st.dataframe(filtered_comments[['User-ID', 'Kommentar', 'Likes']].reset_index(drop=True))
        else:
            st.write("Keine passenden Kommentare gefunden.")
        
        edit_prompt_selected = st.checkbox("Prompt bearbeiten (Gefilterte Kommentare)", key="edit_prompt_selected")
        if edit_prompt_selected:
            filtered_comments_with_metrics = "\n".join(
                filtered_comments.apply(lambda row: f"User: {row['User-ID']} | Likes: {row['Likes']} | Comment: {row['Kommentar']}", axis=1)
            )
            project_info_filtered = (
                "Projekt: Qualitative Analyse von YouTube-Kommentaren (gefiltert).\n"
                "Aufgabe: Analysiere die ausgewählten Kommentare unter Berücksichtigung aller Metriken (Benutzername, Likes, Kommentartext)."
            )
            filtered_input = project_info_filtered + "\n\n" + filtered_comments_with_metrics
            default_prompt = generate_llm_prompt(
                filtered_input,
                selected_video_title,
                "selected",
                keywords if filter_mode == "Keyword" else [],
                comment_count=len(filtered_comments)
            )
            custom_prompt_selected = st.text_area("Bearbeite den Prompt", value=default_prompt, height=200, key="custom_prompt_selected")
        else:
            custom_prompt_selected = None
        
        if st.button("Analyse gefilterter Kommentare mit GPT-4", key="llm_selected"):
            if not filtered_comments.empty:
                filtered_comments_with_metrics = "\n".join(
                    filtered_comments.apply(lambda row: f"User: {row['User-ID']} | Likes: {row['Likes']} | Comment: {row['Kommentar']}", axis=1)
                )
                project_info_filtered = (
                    "Projekt: Qualitative Analyse von YouTube-Kommentaren (gefiltert).\n"
                    "Aufgabe: Analysiere die ausgewählten Kommentare unter Berücksichtigung aller Metriken (Benutzername, Likes, Kommentartext)."
                )
                filtered_input = project_info_filtered + "\n\n" + filtered_comments_with_metrics
                keywords_used = keywords if filter_mode == "Keyword" else []
                used_prompt_selected = custom_prompt_selected if custom_prompt_selected is not None else generate_llm_prompt(
                    filtered_input, selected_video_title, "selected", keywords_used, comment_count=len(filtered_comments)
                )
                llm_result = perform_llm_analysis(
                    filtered_input, 
                    video_title=selected_video_title, 
                    analysis_mode="selected", 
                    keywords=keywords_used, 
                    comment_count=len(filtered_comments),
                    custom_prompt=custom_prompt_selected,
                    max_tokens=2048
                )
                st.write(llm_result)
                download_text = format_download_result("Analyse gefilterter Kommentare", used_prompt_selected, filtered_input, llm_result)
                st.download_button("Download gefilterte Video GPT-4 Ergebnisse", download_text, "llm_video_analysis.txt", "text/plain")
            else:
                st.write("Keine passenden Kommentare gefunden.")
        
        st.markdown("---")
        st.subheader("Globale Suche in allen Video-Kommentaren")
        with st.expander("Globale Filteroptionen"):
            global_filter_mode = st.radio("Globaler Filtermodus auswählen", options=["Keyword", "Username"], index=0, key="global_filter_mode")
            if global_filter_mode == "Keyword":
                global_keyword_groups = {
                    "Akademiker": ["Akademiker", "Uni", "Studenten", "Akademisch", "Hochschule", "Professor", "Forschung"],
                    "Gäste": ["Gast", "Gäste", "Auswahl der Gäste", "Interview", "Talkshow", "Diskussion"],
                    "AFD": ["Alternative für Deutschland", "afd", "adf", "Alternative", "Rechtspartei", "Politik"],
                    "Politisch": ["rechts", "links", "afd", "cdu", "spd", "grüne", "die grünen", "olaf scholz", "habeck", "merkel", "wahl", "regierung", "parlament"]
                }
                custom_keywords_global = st.text_input("Global: Benutzerdefinierte Schlüsselwörter (getrennt durch Komma)", key="custom_kw_global")
                if custom_keywords_global:
                    global_keyword_groups["Benutzerdefiniert"] = [kw.strip() for kw in custom_keywords_global.split(",") if kw.strip()]
                selected_group_global = st.selectbox("Global: Wähle eine Schlüsselwortgruppe", list(global_keyword_groups.keys()), key="keyword_group_global")
                global_keywords = global_keyword_groups[selected_group_global]
                st.write(f"Global: Schlüsselwörter: {', '.join(global_keywords)}")
                global_keywords = [kw.strip().lower() for kw in global_keywords]
                all_filtered_comments = comments_df[comments_df['Kommentar'].str.lower().apply(lambda x: any(kw in x for kw in global_keywords))]
            else:
                global_username_filter = st.text_input("Global: Filter nach Benutzername (User-ID)", value="", key="username_global")
                if global_username_filter:
                    all_filtered_comments = comments_df[comments_df['User-ID'].fillna('').str.lower().str.contains(global_username_filter.lower())]
                else:
                    all_filtered_comments = comments_df
        
        st.write(f"Anzahl der passenden Kommentare in allen Videos: {len(all_filtered_comments)}")
        if not all_filtered_comments.empty:
            st.dataframe(all_filtered_comments[['Video-ID', 'User-ID', 'Kommentar', 'Likes']].reset_index(drop=True))
        else:
            st.write("Keine passenden Kommentare in allen Videos gefunden.")
        
        edit_prompt_global = st.checkbox("Globalen Prompt bearbeiten", key="edit_prompt_global")
        if edit_prompt_global:
            global_comments_with_metrics = "\n".join(
                all_filtered_comments.apply(lambda row: f"User: {row['User-ID']} | Likes: {row['Likes']} | Comment: {row['Kommentar']}", axis=1)
            )
            project_info_global = (
                "Projekt: Globale qualitative Analyse aller Video-Kommentare.\n"
                "Aufgabe: Analysiere die Kommentare unter Berücksichtigung aller Metriken (Benutzername, Likes, Kommentartext)."
            )
            global_input = project_info_global + "\n\n" + global_comments_with_metrics
            default_prompt_global = generate_llm_prompt(
                global_input,
                "Alle Videos", 
                "global",
                global_keywords if global_filter_mode == "Keyword" else [],
                comment_count=len(all_filtered_comments)
            )
            custom_prompt_global = st.text_area("Bearbeite den globalen Prompt", value=default_prompt_global, height=200, key="custom_prompt_global")
        else:
            custom_prompt_global = None
        
        if st.button("Analyse globale Kommentare mit GPT-4", key="llm_global"):
            if not all_filtered_comments.empty:
                global_comments_with_metrics = "\n".join(
                    all_filtered_comments.apply(lambda row: f"User: {row['User-ID']} | Likes: {row['Likes']} | Comment: {row['Kommentar']}", axis=1)
                )
                project_info_global = (
                    "Projekt: Globale qualitative Analyse aller Video-Kommentare.\n"
                    "Aufgabe: Analysiere die Kommentare unter Berücksichtigung aller Metriken (Benutzername, Likes, Kommentartext)."
                )
                global_input = project_info_global + "\n\n" + global_comments_with_metrics
                keywords_used_global = global_keywords if global_filter_mode == "Keyword" else []
                used_prompt_global = custom_prompt_global if custom_prompt_global is not None else generate_llm_prompt(
                    global_input, "Alle Videos", "global", keywords_used_global, comment_count=len(all_filtered_comments)
                )
                global_llm_result = perform_llm_analysis(
                    global_input, 
                    video_title="Alle Videos", 
                    analysis_mode="global", 
                    keywords=keywords_used_global, 
                    comment_count=len(all_filtered_comments),
                    custom_prompt=custom_prompt_global,
                    max_tokens=2048
                )
                st.write(global_llm_result)
                download_text = format_download_result("Globale Kommentare Analyse", used_prompt_global, global_input, global_llm_result)
                st.download_button("Download globale GPT-4 Ergebnisse", download_text, "llm_global_analysis.txt", "text/plain")
            else:
                st.write("Keine passenden Kommentare gefunden.")
