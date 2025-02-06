import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from tabulate import tabulate
import os

# Download NLTK resources
nltk.download("vader_lexicon")
nltk.download("punkt")
nltk.download("stopwords")
nltk.download('punkt_tab')

# Load CSV
FILE_PATH = "youtube_data/comments.csv"

if not os.path.exists(FILE_PATH):
    print(f"❌ Error: File {FILE_PATH} not found.")
    exit()

df = pd.read_csv(FILE_PATH, encoding="utf-8")

# Ensure required columns exist
required_columns = {"video_id", "author", "text", "likes", "timestamp"}
if not required_columns.issubset(df.columns):
    print(f"❌ Error: CSV must contain these columns: {required_columns}")
    exit()

# Clean data
df = df.dropna(subset=["text"])
df["text"] = df["text"].astype(str).str.lower()
df = df.drop_duplicates(subset=["text"])

# Convert timestamp to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# Convert likes to numeric
df["likes"] = pd.to_numeric(df["likes"], errors="coerce").fillna(0).astype(int)

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

def show_stats():
    """Display basic stats about the dataset"""
    total_comments = len(df)
    unique_users = df["author"].nunique()
    avg_length = df["text"].apply(len).mean()
    avg_likes = df["likes"].mean()
    
    print("\n📊 Basic Stats:")
    print(tabulate([
        ["Total Comments", total_comments],
        ["Unique Users", unique_users],
        ["Average Comment Length", f"{avg_length:.2f} characters"],
        ["Average Likes per Comment", f"{avg_likes:.2f}"]
    ], headers=["Metric", "Value"], tablefmt="grid"))

def top_commenters():
    """Display top 10 commenters"""
    print("\n🏆 Top 10 Most Active Commenters:")
    top_users = df["author"].value_counts().head(10)
    print(tabulate(top_users.items(), headers=["User", "Comments"], tablefmt="grid"))

def sentiment_analysis():
    """Perform sentiment analysis"""
    df["sentiment"] = df["text"].apply(lambda x: sia.polarity_scores(x)["compound"])
    
    positive = df[df["sentiment"] > 0.2].sort_values(by="sentiment", ascending=False).head(5)
    negative = df[df["sentiment"] < -0.2].sort_values(by="sentiment", ascending=True).head(5)

    print("\n😊 Top 5 Positive Comments:")
    print(tabulate(positive[["text", "likes", "sentiment"]], headers=["Comment", "Likes", "Sentiment"], tablefmt="grid"))

    print("\n😠 Top 5 Negative Comments:")
    print(tabulate(negative[["text", "likes", "sentiment"]], headers=["Comment", "Likes", "Sentiment"], tablefmt="grid"))

def generate_wordcloud():
    """Generate a word cloud from the comments"""
    stop_words = set(stopwords.words("english"))
    text = " ".join(df["text"])
    words = " ".join([word for word in word_tokenize(text) if word.isalnum() and word not in stop_words])

    wordcloud = WordCloud(width=1000, height=500, background_color="black").generate(words)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

def most_liked_comments():
    """Display top 5 most liked comments"""
    top_liked = df.sort_values(by="likes", ascending=False).head(5)
    print("\n🔥 Top 5 Most Liked Comments:")
    print(tabulate(top_liked[["text", "author", "likes"]], headers=["Comment", "Author", "Likes"], tablefmt="grid"))

def comments_over_time():
    """Plot comments over time"""
    df_time = df.set_index("timestamp").resample("D").size()
    plt.figure(figsize=(10, 5))
    plt.plot(df_time, marker="o", linestyle="-")
    plt.xlabel("Date")
    plt.ylabel("Number of Comments")
    plt.title("📅 Comments Trend Over Time")
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()

def cli():
    """Simple CLI menu"""
    while True:
        print("\n📌 Comment Analyzer Menu:")
        print("1️⃣ Basic Stats")
        print("2️⃣ Top Commenters")
        print("3️⃣ Sentiment Analysis")
        print("4️⃣ Word Cloud")
        print("5️⃣ Most Liked Comments")
        print("6️⃣ Comment Trends Over Time")
        print("7️⃣ Exit")

        choice = input("Select an option (1-7): ")

        if choice == "1":
            show_stats()
        elif choice == "2":
            top_commenters()
        elif choice == "3":
            sentiment_analysis()
        elif choice == "4":
            generate_wordcloud()
        elif choice == "5":
            most_liked_comments()
        elif choice == "6":
            comments_over_time()
        elif choice == "7":
            print("👋 Exiting... Goodbye!")
            break
        else:
            print("❌ Invalid choice. Try again.")

if __name__ == "__main__":
    cli()
