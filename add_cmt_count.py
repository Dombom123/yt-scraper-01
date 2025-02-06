import os
import json
import pandas as pd

# Paths
DATA_FOLDER = "youtube_data"
METADATA_FOLDER = os.path.join(DATA_FOLDER, "metadata")
CSV_PATH = os.path.join(DATA_FOLDER, "videos_detailed.csv")
NEW_CSV_PATH = os.path.join(DATA_FOLDER, "videos_detailed_with_comments.csv")

# Load the CSV file
df = pd.read_csv(CSV_PATH)

# Add a new column for comment count
df['comment_count'] = 0

# Iterate over each video ID in the CSV
for index, row in df.iterrows():
    video_id = row['id']
    comments_file = os.path.join(METADATA_FOLDER, video_id, "comments.json")
    
    # Check if the comments file exists
    if os.path.exists(comments_file):
        with open(comments_file, "r", encoding="utf-8") as f:
            try:
                comments_data = json.load(f)
                # Update the comment count in the DataFrame
                df.at[index, 'comment_count'] = len(comments_data)
            except json.JSONDecodeError:
                print(f"⚠️ Failed to decode comments for {video_id}")

# Save the updated DataFrame to a new CSV file
df.to_csv(NEW_CSV_PATH, index=False)