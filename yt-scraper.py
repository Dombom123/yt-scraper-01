import yt_dlp
import json
import os
import pandas as pd
import time

# ---- Configuration ----
CHANNEL_URL = "https://www.youtube.com/playlist?list=UUWKEY-aEu7gcv5ayIpgrnvg"
DATA_FOLDER = "youtube_data"
FAILED_VIDEOS_PATH = os.path.join(DATA_FOLDER, "failed_videos.json")
VIDEOS_JSON_PATH = os.path.join(DATA_FOLDER, "videos.json")
VIDEOS_DETAILED_JSON_PATH = os.path.join(DATA_FOLDER, "videos_detailed.json")

# ---- Helper Functions ----
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_json(data, file_path):
    """Saves JSON data to a file."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# ---- Step 1: Fetch Playlist Videos ----
def get_playlist_videos(playlist_url):
    print(f"📥 Fetching video list from: {playlist_url}")
    ydl_opts = {
        "quiet": False,
        "extract_flat": True,  # Only extract URLs
        "skip_download": True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(playlist_url, download=False)

    if not result:
        raise Exception("🚨 Failed to fetch playlist videos!")

    video_list = [
        {"id": video.get("id"), "title": video.get("title"), "url": f"https://www.youtube.com/watch?v={video.get('id')}"}
        for video in result.get("entries", [])
    ]
    print(f"✅ Fetched {len(video_list)} videos.")
    return video_list

# ---- Step 2: Fetch Detailed Video Metadata ----
def get_video_details(video_url):
    print(f"🔍 Fetching metadata for: {video_url}")
    ydl_opts = {
        "quiet": False,
        "skip_download": True
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(video_url, download=False)

        video_info = {
            "id": result.get("id"),
            "title": result.get("title"),
            "url": video_url,
            "duration": result.get("duration"),
            "upload_date": result.get("upload_date"),
            "view_count": result.get("view_count"),
            "like_count": result.get("like_count"),
        }

        metadata_folder = os.path.join(DATA_FOLDER, "metadata")
        create_folder(metadata_folder)
        metadata_path = os.path.join(metadata_folder, f"info_{video_info['id']}.json")

        # Avoid redundant API calls
        if os.path.exists(metadata_path):
            print(f"🔄 Skipping {video_info['title']}, metadata already exists.")
            return video_info

        save_json(result, metadata_path)
        print(f"✅ Metadata saved for: {video_info['title']}")
        return video_info
    except Exception as e:
        print(f"❌ Failed to fetch details for {video_url}: {e}")
        return None

# ---- Step 3: Fetch Comments ----
def get_comments(video_url, video_id):
    print(f"💬 Fetching comments for: {video_url}")
    ydl_opts = {
        "quiet": False,
        "writecomments": True,  # ✅ Corrected!
        "skip_download": True
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(video_url, download=False)

        comments = result.get("comments", [])
        comment_list = [{"author": c.get("author"), "text": c.get("text"), "likes": c.get("like_count"), "timestamp": c.get("timestamp")} for c in comments]

        comments_folder = os.path.join(DATA_FOLDER, "comments")
        create_folder(comments_folder)
        comments_path = os.path.join(comments_folder, f"comments_{video_id}.json")
        save_json(comment_list, comments_path)

        print(f"✅ {len(comment_list)} comments saved.")
        return comment_list

    except Exception as e:
        print(f"⚠️ Failed to fetch comments for {video_url}: {e}")
        return None

# ---- Main Execution ----
if __name__ == "__main__":
    create_folder(DATA_FOLDER)

    print("🔍 Fetching video list...")
    videos = get_playlist_videos(CHANNEL_URL)
    save_json(videos, VIDEOS_JSON_PATH)

    detailed_videos = []

    for idx, video in enumerate(videos):
        print(f"📹 Processing {idx+1}/{len(videos)}: {video['title']}")
        video_details = get_video_details(video["url"])
        if video_details:
            detailed_videos.append(video_details)
            save_json(detailed_videos, VIDEOS_DETAILED_JSON_PATH)

        # Fetch & save comments
        comments = get_comments(video["url"], video_details["id"]) if video_details else None

        # Wait to avoid rate-limiting
        time.sleep(2)

    # Save detailed videos as CSV
    pd.DataFrame(detailed_videos).to_csv(os.path.join(DATA_FOLDER, "videos_detailed.csv"), index=False, encoding="utf-8")

    print(f"✅ {len(detailed_videos)} videos saved.")
