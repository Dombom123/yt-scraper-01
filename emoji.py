import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import string

# Title of the app
st.title('Emoji Word Cloud Generator')

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)
    
    # Combine all comments into a single string
    text = ' '.join(df['text'])

    # Define a regular expression to match words, ASCII art, and emojis
    normal_word = r"(?:\w[\w']+)"
    ascii_art = r"(?:[" + string.punctuation + r"][" + string.punctuation + r"]+)"
    emoji = r"(?:[^\s" + string.printable + r"])"
    regexp = r"{}|{}|{}".format(normal_word, ascii_art, emoji)

    # Generate the word cloud
    wordcloud = WordCloud(regexp=regexp, width=800, height=400, background_color='white').generate(text)

    # Display the word cloud
    st.image(wordcloud.to_array())

    # Option to download the word cloud
    st.download_button(
        label="Download Word Cloud",
        data=wordcloud.to_image().tobytes(),
        file_name="emoji_wordcloud.png",
        mime="image/png"
    )
