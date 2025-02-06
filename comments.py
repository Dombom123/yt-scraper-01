import os
import json
import subprocess
import pandas as pd
import time

# ---- Configuration ----
DATA_FOLDER = "youtube_data"
METADATA_FOLDER = os.path.join(DATA_FOLDER, "metadata")
VIDEOS_JSON_PATH = os.path.join(DATA_FOLDER, "videos.json")
COMMENTS_CSV_PATH = os.path.join(DATA_FOLDER, "comments.csv")

# ---- Helper Functions ----
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def fetch_metadata(video_id, video_url):
    """Fetches metadata including comments for a video."""
    print(f"📥 Fetching metadata for: {video_url}")
    
    video_folder = os.path.join(METADATA_FOLDER, video_id)
    create_folder(video_folder)
    metadata_file = os.path.join(video_folder, "info.json")
    comments_file = os.path.join(video_folder, "comments.json")
    
    cmd = [
        "yt-dlp", video_url,
        "--write-comments",
        "--write-thumbnail",
        "--skip-download",
        "--print-to-file", "after_filter:%(comments)j", comments_file,
        "--output", os.path.join(video_folder, "%(id)s")
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Metadata saved for {video_id}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to fetch metadata for {video_id}: {e}")
        return None
    
    return video_folder

def extract_comments():
    """Extracts comments from metadata and compiles into a CSV."""
    print("📊 Compiling comments into CSV...")
    all_comments = []
    
    for video_id in os.listdir(METADATA_FOLDER):
        comments_file = os.path.join(METADATA_FOLDER, video_id, "comments.json")
        
        if os.path.exists(comments_file):
            with open(comments_file, "r", encoding="utf-8") as f:
                try:
                    comments_data = json.load(f)
                    for comment in comments_data:
                        all_comments.append({
                            "video_id": video_id,
                            "author": comment.get("author"),
                            "author_id": comment.get("author_id"),
                            "author": comment.get("author"),
                            
                            "text": comment.get("text"),
                            "likes": comment.get("like_count"),
                            "timestamp": comment.get("timestamp"),
                        })
                except json.JSONDecodeError:
                    print(f"⚠️ Failed to decode comments for {video_id}")

    # "parent": "root",
    # "text": "Cannabis: Heilpflanze oder gef\u00e4hrliche Droge? Nanjing und Mathias haben dazu unterschiedliche Einstellungen. Aber wie seht Ihr das? Schreibt's uns in die Kommentare!",
    # "like_count": 26,
    # "author_id": "UCWKEY-aEu7gcv5ayIpgrnvg",
    # "author": "@unbubble",
    # "author_thumbnail": "https://yt3.ggpht.com/gHiWogNfGHQ6kbOajenQkOlGdTZj3FRP1W6q3zR8pCa-kZtAdbOoqhz78b4NWZZEN40QS1EFEbA=s88-c-k-c0x00ffffff-no-rj",
    # "author_is_uploader": true,
    # "author_is_verified": true,
    # "author_url": "https://www.youtube.com/@unbubble",
    # "is_favorited": false,
    # "_time_text": "2 years ago",
    # "timestamp": 1674000000,
    # "is_pinned": true
    
    df = pd.DataFrame(all_comments)
    df.to_csv(COMMENTS_CSV_PATH, index=False, encoding="utf-8")
    print(f"✅ Comments saved to {COMMENTS_CSV_PATH}")

# ---- Main Execution ----
if __name__ == "__main__":
    # create_folder(METADATA_FOLDER)
    
    # if not os.path.exists(VIDEOS_JSON_PATH):
    #     print("🚨 videos.json not found!")
    #     exit()
    
    # with open(VIDEOS_JSON_PATH, "r", encoding="utf-8") as f:
    #     videos = json.load(f)
    
    # for video in videos:
    #     video_id = video.get("id")
    #     video_url = video.get("url")
        
    #     if not video_id or not video_url:
    #         continue
        
    #     video_folder = os.path.join(METADATA_FOLDER, video_id)
    #     if os.path.exists(os.path.join(video_folder, "info.json")):
    #         print(f"⚡ Metadata already exists for {video_id}, skipping...")
    #         continue
        
    #     fetch_metadata(video_id, video_url)
    #     time.sleep(2)  # Avoid rate-limiting
    
    extract_comments()
