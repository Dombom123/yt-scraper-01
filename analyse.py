import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from collections import Counter
import nltk

# Ensure you have the necessary NLTK data
nltk.download('vader_lexicon')
nltk.download('stopwords')

# Load the CSV files
videos_df = pd.read_csv('youtube_data/videos_detailed_with_comments.csv')
comments_df = pd.read_csv('youtube_data/comments.csv')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Sentiment Analysis on comments
comments_df['sentiment'] = comments_df['text'].apply(lambda x: sia.polarity_scores(x)['compound'])
comments_df['sentiment_label'] = comments_df['sentiment'].apply(lambda x: 'positive' if x > 0 else 'negative' if x < 0 else 'neutral')

# Save sentiment analysis results
comments_df.to_csv('youtube_data/comments_with_sentiment.csv', index=False)

# Count positive and negative comments per video
sentiment_counts = comments_df.groupby('video_id')['sentiment_label'].value_counts().unstack().fillna(0)
sentiment_counts.to_csv('youtube_data/sentiment_counts_per_video.csv')

# Wordcloud for comments
german_stopwords = set(stopwords.words('german'))
all_comments = ' '.join(comments_df['text'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=german_stopwords).generate(all_comments)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Most used words in comments
words = all_comments.split()
most_common_words = Counter(words).most_common(10)

# Videos with longest discussions
comments_df['comment_length'] = comments_df['text'].apply(lambda x: len(x.split()) if pd.notnull(x) else 0)
longest_discussions = comments_df.groupby('video_id')['comment_length'].sum().sort_values(ascending=False)
longest_discussions.to_csv('youtube_data/longest_discussions.csv')

# Most frequent commenters
most_comments_by_user = comments_df['author'].value_counts().head(10)
most_comments_by_user.to_csv('youtube_data/most_frequent_commenters.csv')

# Merge videos and comments data using the correct columns
merged_df = pd.merge(videos_df, comments_df, left_on='id', right_on='video_id', how='left')

# Top 10 successful videos by views
top_videos = videos_df.sort_values(by='view_count', ascending=False).head(10)
top_videos.to_csv('youtube_data/top_10_videos_by_views.csv', index=False)

# Outliers in comments/views ratio
videos_df['comments_to_views_ratio'] = videos_df['comment_count'] / videos_df['view_count']
outliers = videos_df[videos_df['comments_to_views_ratio'] > videos_df['comments_to_views_ratio'].mean() + 2 * videos_df['comments_to_views_ratio'].std()]
outliers.to_csv('youtube_data/outliers_in_comments_views_ratio.csv', index=False)

# Display results
print("Sentiment Counts per Video:\n", sentiment_counts)
print("\nMost Common Words in Comments:\n", most_common_words)
print("\nVideos with Longest Discussions:\n", longest_discussions.head(10))
print("\nMost Frequent Commenters:\n", most_comments_by_user)
print("\nTop 10 Successful Videos by Views:\n", top_videos[['id', 'title', 'view_count']])
print("\nOutliers in Comments/Views Ratio:\n", outliers[['id', 'title', 'comments_to_views_ratio']])

# Note: For LLM-based analysis, you would need to integrate with an LLM API or library to perform content analysis.