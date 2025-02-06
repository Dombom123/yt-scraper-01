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

# --- Missing load_pickle function definition ---
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
    # 1️⃣ Load German stopwords
    german_stopwords = set(nltk_stopwords.words("german"))

    # 2️⃣ Add custom stopwords (ensuring lowercase for case-insensitivity)
    additional_stopwords = set([w.strip().lower() for w in custom_stopwords.split(",") if w.strip()])
    active_stopwords = german_stopwords.union(additional_stopwords)

    # 3️⃣ Clean and tokenize text
    cleaned_text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = cleaned_text.split()  # Tokenize by whitespace

    # 4️⃣ Filter words: remove stopwords and apply minimum word length
    filtered_words = [
        word for word in words
        if len(word) >= min_word_length and word.lower() not in active_stopwords
    ]
    filtered_text = " ".join(filtered_words)

    # 5️⃣ Generate the WordCloud
    wc = WordCloud(width=800, height=400, background_color='white').generate(filtered_text)

    return wc, active_stopwords, filtered_text  # Returning active stopwords for debug display


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
            f"Analysiere die wichtigsten Themen, Stimmungen und Erkenntnisse und fasse sie prägnant zusammen. "
            f"Hier sind die Kommentare:\n\n{text}\n\nZusammenfassung:"
        )
    elif analysis_mode == "global":
        prompt = (
            f"Bitte führe eine umfassende qualitative Analyse der folgenden Video-Kommentare aus allen Videos durch. "
            f"Es wurden {comment_count} Kommentare gefunden, die mit den Schlüsselwörtern '{keywords_str}' gefiltert wurden. "
            f"Analysiere die wichtigsten Themen, Stimmungen und Erkenntnisse und fasse sie prägnant zusammen. "
            f"Hier sind die Kommentare:\n\n{text}\n\nZusammenfassung:"
        )
    else:
        prompt = (
            f"Bitte führe eine qualitative Analyse der folgenden Video-Kommentare durch. "
            f"Das Video heißt '{video_title}'. Fasse die wichtigsten Themen, Stimmungen und Erkenntnisse zusammen. "
            f"Hier sind die Kommentare:\n\n{text}\n\nZusammenfassung:"
        )
    return prompt

def perform_llm_analysis(text, video_title="", analysis_mode="default", keywords=None, comment_count=None, max_length=10000, custom_prompt=None, max_tokens=1000):
    if custom_prompt is None:
        prompt = generate_llm_prompt(text, video_title, analysis_mode, keywords, comment_count, max_length)
    else:
        prompt = custom_prompt
    try:
        with st.spinner("GPT-4 Analyse läuft..."):
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Du bist ein hilfsbereiter Analyse-Assistent, der Video-Kommentare qualitativ auswertet."},
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
# Initialize session state variables if not already set
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
# Default Data Loading Block
# -----------------------------------------------------------------------------
if st.session_state.video_df is None:
    default_video_file = "youtube_data/videos_detailed_with_comments.csv"
    if os.path.exists(default_video_file):
        video_df_default = prepare_video_data(default_video_file)
        if video_df_default is not None:
            st.session_state.video_df = video_df_default
            st.info("Standard Video-Metadaten aus 'videos_detailed_with_comments.csv' geladen.")
    else:
        st.info("Kein Standard-Video-Datensatz gefunden. Bitte lade eigene Daten hoch.")

if st.session_state.comments_df is None:
    default_comments_file = "youtube_data/comments_with_sentiment.csv"
    if os.path.exists(default_comments_file):
        comments_df_default = prepare_comments_data(default_comments_file)
        if comments_df_default is not None:
            st.session_state.comments_df = comments_df_default
            st.info("Standard Kommentardaten aus 'comments_with_sentiment.csv' geladen.")
    else:
        st.info("Kein Standard-Kommentardatensatz gefunden. Bitte lade eigene Daten hoch.")

# Load default quantitative results (pickle) if available.
if not st.session_state.quant_results:
    default_quant_file = "youtube_data/quant_results.pkl"
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

# Instead of loading sentiment_analysis separately from the same pickle,
# we now rely on quant_results (which should include sentiment analysis data, e.g., under key 'sentiment_df').
if st.session_state.sentiment_analysis is None and st.session_state.quant_results:
    if "sentiment_df" in st.session_state.quant_results and isinstance(st.session_state.quant_results["sentiment_df"], pd.DataFrame):
        st.session_state.sentiment_analysis = st.session_state.quant_results["sentiment_df"]
        st.info("Sentiment-Analyse aus quant_results geladen.")

# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------
page = st.sidebar.selectbox("Navigation", 
    ["Daten Upload & Vorbereitung", "Quantitative Analyse", "Qualitative Analyse", "Interaktives Dashboard"]
)

# ----- Seite 1: Daten Upload & Vorbereitung -----
if page == "Daten Upload & Vorbereitung":
    st.header("Daten Upload und Vorbereitung")
    st.write("Lade die CSV-Dateien mit den Video-Metadaten (z. B. `videos.csv`) und Kommentaren (z. B. `comments.csv`) hoch.")

    if st.session_state.video_df is not None:
        st.info("Standard Video-Daten wurden bereits geladen.")
        st.subheader("Vorschau Video-Metadaten")
        st.dataframe(st.session_state.video_df.head())
        st.subheader("Datenzusammenfassung")
        st.dataframe(data_summary(st.session_state.video_df))
    if st.session_state.comments_df is not None:
        st.info("Standard Kommentardaten wurden bereits geladen.")
        st.subheader("Vorschau Kommentardaten")
        st.dataframe(st.session_state.comments_df.head())
        st.subheader("Datenzusammenfassung")
        st.dataframe(data_summary(st.session_state.comments_df))
    
    video_file = st.file_uploader("Video-Metadaten CSV", type=["csv"], key="video")
    comments_file = st.file_uploader("Kommentare CSV", type=["csv"], key="comments")
    
    if video_file is not None:
        video_df = prepare_video_data(video_file)
        if video_df is not None:
            st.session_state.video_df = video_df
            st.subheader("Vorschau Video-Metadaten (Neue Upload)")
            st.dataframe(video_df.head())
            st.subheader("Datenzusammenfassung")
            st.dataframe(data_summary(video_df))
            st.success("Video-Daten erfolgreich geladen, validiert und bereinigt.")
            csv_video = video_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Video CSV", csv_video, "video_data.csv", "text/csv")
    
    if comments_file is not None:
        comments_df = prepare_comments_data(comments_file)
        if comments_df is not None:
            st.session_state.comments_df = comments_df
            st.subheader("Vorschau Kommentardaten (Neue Upload)")
            st.dataframe(comments_df.head())
            st.subheader("Datenzusammenfassung")
            st.dataframe(data_summary(comments_df))
            st.success("Kommentardaten erfolgreich geladen, validiert und bereinigt.")
            csv_comments = comments_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Kommentare CSV", csv_comments, "comments_data.csv", "text/csv")

# ----- Seite 2: Quantitative Analyse -----
elif page == "Quantitative Analyse":
    st.header("Quantitative Analyse")
    
    if st.session_state.video_df is None or st.session_state.comments_df is None:
        st.warning("Bitte zuerst im Bereich 'Daten Upload & Vorbereitung' die erforderlichen Daten hochladen.")
    else:
        video_df = st.session_state.video_df
        comments_df = st.session_state.comments_df
        
        st.subheader("Einstellungen")

        with st.expander("Wordcloud Einstellungen"):
            custom_stopwords = st.text_input("Zusätzliche Stopwords (getrennt durch Komma)", value="unbubble, shorts, 13 Fragen, Sollten, trifft, Sags, Sag's, sag, Fragen, unfiltered, live, 13")
            min_word_length = st.number_input("Minimale Wortlänge", min_value=1, value=2)

    

            # 1️⃣ Update WordCloud for Video Titles
            if st.button("🔄 Wordcloud für Video-Titel aktualisieren"):
                st.session_state.pop('wordcloud_title_fig', None)  # Clear previous figure
                if "Titel" in video_df.columns:
                    title_text = " ".join(video_df['Titel'].dropna().astype(str))
                    wc_title, active_stopwords_title, filtered_text = generate_wordcloud(title_text, custom_stopwords, min_word_length)
                    
                    fig_wc_title, ax_title = plt.subplots(figsize=(10, 5))
                    ax_title.imshow(wc_title, interpolation="bilinear")
                    ax_title.axis("off")
                    st.session_state['wordcloud_title_fig'] = fig_wc_title
                    
                    st.pyplot(fig_wc_title)  # Immediate render
                    st.write(f"🚩 Aktive Stopwords (Titel): {sorted(active_stopwords_title)}")
                    st.write(f"🚩 Filtered Text (Titel): {filtered_text}")
                    st.success("✅ Wordcloud für Video-Titel wurde aktualisiert!")



        
        if st.button("Starte Quantitative Analyse"):
            # Use preloaded sentiment analysis if available; otherwise, perform it.
            if (st.session_state.sentiment_analysis is
                isinstance(st.session_state.sentiment_analysis, pd.DataFrame) and 
                "sentiment" in st.session_state.sentiment_analysis.columns):
                comments_df_with_sentiment = st.session_state.sentiment_analysis
            else:
                comments_df_with_sentiment = perform_sentiment_analysis(comments_df)
                st.session_state.sentiment_analysis = comments_df_with_sentiment
            sentiment_counts = comments_df_with_sentiment['sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Anzahl']
            fig_eng = plot_engagement(video_df) if all(col in video_df.columns for col in ['Views', 'Likes', 'Kommentare']) else None
            top10_df = rank_top_videos(video_df) if all(col in video_df.columns for col in ['Views', 'Likes', 'Kommentare']) else pd.DataFrame()
            
            st.session_state.quant_results = {
                "sentiment_df": comments_df_with_sentiment.copy(),
                "sentiment_counts": sentiment_counts,
                "engagement_fig": fig_eng,
                "top10_df": top10_df,
                "wordcloud_title_fig": st.session_state.get("wordcloud_title_fig", None),
                "wordcloud_comments_fig": st.session_state.get("wordcloud_comments_fig", None)
            }
            st.success("Quantitative Analyse abgeschlossen.")
        
        if st.session_state.quant_results:
            st.subheader("Ergebnisse der Sentiment-Analyse")
            sentiment_counts = st.session_state.quant_results.get("sentiment_counts")
            if sentiment_counts is not None:
                fig_sent = px.bar(sentiment_counts, x='Sentiment', y='Anzahl', title="Verteilung der Sentiments")
                st.plotly_chart(fig_sent)
            
            st.subheader("Wordcloud der Video-Titel")
            fig_wc_title = st.session_state.quant_results.get("wordcloud_title_fig")
            if fig_wc_title is not None:
                st.pyplot(fig_wc_title)
                download_figure(fig_wc_title, "video_titles_wordcloud.png")
            

            st.subheader("Engagement Analyse")
            fig_eng = st.session_state.quant_results.get("engagement_fig")
            if fig_eng is not None:
                st.plotly_chart(fig_eng)
            
            st.subheader("Top 10 Videos nach Engagement")
            top10_df = st.session_state.quant_results.get("top10_df")
            if not top10_df.empty:
                st.dataframe(top10_df)
                csv_top10 = top10_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download Top 10 Videos CSV", csv_top10, "top10_videos.csv", "text/csv")
        
        st.markdown("---")
        st.subheader("Erweiterte Quantitative Analysen")
        with st.expander("Erweiterte Filteroptionen"):
            username_filter = st.text_input("Filter nach Benutzername (User-ID)", value="")
            if 'Datum' in comments_df.columns and not comments_df['Datum'].isnull().all():
                default_start = comments_df['Datum'].min().date()
                default_end = comments_df['Datum'].max().date()
                date_range = st.date_input("Filter nach Datum (Kommentare)", value=(default_start, default_end))
            else:
                date_range = None
            if date_range and isinstance(date_range, (tuple, list)):
                start_date, end_date = date_range
            else:
                start_date = end_date = None
            apply_filter = st.button("Filter anwenden", key="apply_filter_extended")
        
        if apply_filter:
            filtered_comments_extended = filter_comments(comments_df, username_filter, start_date, end_date)
            st.write(f"Nach Filterung verbleiben {len(filtered_comments_extended)} Kommentare.")
            
            st.subheader("Kommentare pro Account")
            comments_per_account = analyze_comments_per_account(filtered_comments_extended)
            st.dataframe(comments_per_account)
            fig_comments_account = px.bar(comments_per_account.head(10), x='User-ID', y='Kommentaranzahl', title="Top 10 Konten nach Kommentaranzahl")
            st.plotly_chart(fig_comments_account)
            
            st.subheader("Sentiment pro Account")
            sentiments_account = analyze_sentiments_per_account(filtered_comments_extended)
            st.dataframe(sentiments_account)
            if not sentiments_account.empty:
                sentiments_account['total'] = sentiments_account.get('Positiv', 0) + sentiments_account.get('Negativ', 0) + sentiments_account.get('Neutral', 0)
                top_accounts = sentiments_account.sort_values('total', ascending=False).head(10)
                fig_sentiments = px.bar(top_accounts, x='User-ID', y=['Positiv', 'Negativ', 'Neutral'],
                                        barmode='group', title="Sentiment Verteilung pro Account (Top 10)")
                st.plotly_chart(fig_sentiments)
            
            st.subheader("Vergleich von Sentiment und Views pro Video")
            video_sentiment_view = compare_sentiments_to_views(video_df, comments_df)
            st.dataframe(video_sentiment_view[['Video-ID', 'Titel', 'Views', 'Positiv', 'Negativ', 'Neutral', 
                                                 'Positiv/Views', 'Negativ/Views']])
            chart_option = st.selectbox("Darstellung wählen", ["Bar Chart", "Wordcloud"], key="chart_option")
            if chart_option == "Bar Chart":
                fig_video = px.bar(video_sentiment_view.head(10), x='Titel', y=['Positiv/Views', 'Negativ/Views'], 
                                   barmode='group', title="Sentiment zu Views Verhältnis (Top 10 Videos)")
                st.plotly_chart(fig_video)
            else:
                video_titles = " ".join(video_sentiment_view['Titel'].dropna().astype(str))
                wc_video = generate_wordcloud(video_titles, custom_stopwords="", min_word_length=2)
                fig_wc_video, ax_wc_video = plt.subplots(figsize=(10, 5))
                ax_wc_video.imshow(wc_video, interpolation="bilinear")
                ax_wc_video.axis("off")
                plt.close(fig_wc_video)
                st.pyplot(fig_wc_video)
                download_figure(fig_wc_video, "video_sentiment_wordcloud.png")
        
        st.markdown("---")
        st.subheader("Zusätzliche Analysen")
        with st.expander("Weitere Analyse Optionen"):
            extra_option = st.selectbox("Wählen Sie eine zusätzliche Analyse", 
                                         options=["Top 10 Commenters", "Kommentare pro Video", "Durchschnittlicher Sentiment Score pro Video", "Korrelationsmatrix (Heatmap)", "Verteilung der Likes pro Video"],
                                         key="extra_analysis")
            if extra_option == "Top 10 Commenters":
                top_commenters = analyze_comments_per_account(comments_df)
                st.dataframe(top_commenters.head(10))
                fig_top_commenters = px.bar(top_commenters.head(10), x='User-ID', y='Kommentaranzahl', title="Top 10 Commenters")
                st.plotly_chart(fig_top_commenters)
            elif extra_option == "Kommentare pro Video":
                comments_per_video = comments_df.groupby('Video-ID').size().reset_index(name='Anzahl Kommentare')
                if 'Titel' in video_df.columns:
                    comments_per_video = comments_per_video.merge(video_df[['Video-ID','Titel']], on='Video-ID', how='left')
                    fig_comments_video = px.bar(comments_per_video.sort_values("Anzahl Kommentare", ascending=False).head(10), 
                                                x='Titel', y='Anzahl Kommentare', 
                                                title="Top 10 Videos nach Kommentaranzahl")
                else:
                    fig_comments_video = px.bar(comments_per_video.sort_values("Anzahl Kommentare", ascending=False).head(10), 
                                                x='Video-ID', y='Anzahl Kommentare', 
                                                title="Top 10 Videos nach Kommentaranzahl")
                st.dataframe(comments_per_video.sort_values("Anzahl Kommentare", ascending=False).head(10))
                st.plotly_chart(fig_comments_video)
            elif extra_option == "Durchschnittlicher Sentiment Score pro Video":
                avg_sentiment = comments_df.groupby('Video-ID')['sentiment_score'].mean().reset_index(name='Durchschnittlicher Sentiment Score')
                if 'Titel' in video_df.columns:
                    avg_sentiment = avg_sentiment.merge(video_df[['Video-ID','Titel']], on='Video-ID', how='left')
                    fig_avg_sentiment = px.bar(avg_sentiment.sort_values("Durchschnittlicher Sentiment Score", ascending=False).head(10), 
                                               x='Titel', y='Durchschnittlicher Sentiment Score', 
                                               title="Top 10 Videos nach Durchschnittlichem Sentiment Score")
                else:
                    fig_avg_sentiment = px.bar(avg_sentiment.sort_values("Durchschnittlicher Sentiment Score", ascending=False).head(10), 
                                               x='Video-ID', y='Durchschnittlicher Sentiment Score', 
                                               title="Top 10 Videos nach Durchschnittlichem Sentiment Score")
                st.dataframe(avg_sentiment.sort_values("Durchschnittlicher Sentiment Score", ascending=False).head(10))
                st.plotly_chart(fig_avg_sentiment)
            elif extra_option == "Korrelationsmatrix (Heatmap)":
                numeric_cols = video_df.select_dtypes(include=[np.number])
                corr = numeric_cols.corr()
                fig_heatmap = px.imshow(corr, text_auto=True, title="Korrelationsmatrix der Video-Metriken")
                st.plotly_chart(fig_heatmap)
            elif extra_option == "Verteilung der Likes pro Video":
                fig_likes_hist = px.histogram(video_df, x='Likes', nbins=20, title="Verteilung der Likes pro Video")
                st.plotly_chart(fig_likes_hist)
        
        st.subheader("Ergebnisse speichern/laden")
        if st.session_state.quant_results:
            pickled_results = save_quantitative_results(st.session_state.quant_results)
            st.download_button("Download Quantitative Ergebnisse", data=pickled_results, file_name="quant_results.pkl", mime="application/octet-stream")
        uploaded_quant = st.file_uploader("Lade zuvor gespeicherte quantitative Ergebnisse", type=["pkl"], key="quant_load")
        if uploaded_quant is not None:
            loaded_results = load_quantitative_results(uploaded_quant)
            if loaded_results:
                st.session_state.quant_results = loaded_results
                st.success("Quantitative Ergebnisse erfolgreich geladen.")

# ----- Seite 3: Qualitative Analyse -----
elif page == "Qualitative Analyse":
    st.header("Qualitative Analyse mittels GPT-4")
    
    if st.session_state.video_df is None or st.session_state.comments_df is None:
        st.warning("Bitte zuerst im Bereich 'Daten Upload & Vorbereitung' die Daten hochladen.")
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
            if custom_prompt_all is not None:
                used_prompt_all = custom_prompt_all
            else:
                used_prompt_all = generate_llm_prompt(all_comments_input, selected_video_title, "selected", [], comment_count=len(video_comments))
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
                if custom_prompt_selected is not None:
                    used_prompt_selected = custom_prompt_selected
                else:
                    used_prompt_selected = generate_llm_prompt(filtered_input, selected_video_title, "selected", keywords_used, comment_count=len(filtered_comments))
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
                if custom_prompt_global is not None:
                    used_prompt_global = custom_prompt_global
                else:
                    used_prompt_global = generate_llm_prompt(global_input, "Alle Videos", "global", keywords_used_global, comment_count=len(all_filtered_comments))
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
                
# ----- Seite 4: Interaktives Dashboard -----
elif page == "Interaktives Dashboard":
    st.header("Interaktives Dashboard")
    
    if st.session_state.video_df is None or st.session_state.comments_df is None:
        st.warning("Bitte zuerst im Bereich 'Daten Upload & Vorbereitung' die Daten hochladen.")
    else:
        video_df = st.session_state.video_df
        comments_df = st.session_state.comments_df
        
        st.subheader("KPI Übersicht")
        col1, col2, col3 = st.columns(3)
        total_views = video_df['Views'].sum() if 'Views' in video_df.columns else 0
        total_likes = video_df['Likes'].sum() if 'Likes' in video_df.columns else 0
        total_comments = video_df['Kommentare'].sum() if 'Kommentare' in video_df.columns else 0
        col1.metric("Gesamt Views", total_views)
        col2.metric("Gesamt Likes", total_likes)
        col3.metric("Gesamt Kommentare", total_comments)
        
        st.subheader("Sentiment Verteilung")
        if 'sentiment' in comments_df.columns:
            sentiment_counts = comments_df['sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Anzahl']
            fig_pie = px.pie(sentiment_counts, names='Sentiment', values='Anzahl', title="Sentiment Anteil")
            st.plotly_chart(fig_pie)
        else:
            st.write("Die Sentiment-Analyse wurde noch nicht durchgeführt.")
        
        st.subheader("Daten Export")
        csv_video = video_df.to_csv(index=False).encode("utf-8")
        csv_comments = comments_df.to_csv(index=False).encode("utf-8")
        col_download1, col_download2 = st.columns(2)
        col_download1.download_button("Download Video CSV", csv_video, "video_data.csv", "text/csv")
        col_download2.download_button("Download Kommentare CSV", csv_comments, "comments_data.csv", "text/csv")
