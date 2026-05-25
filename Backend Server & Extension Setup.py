# ==============================================================================
# STEP 1: INSTALL ALL PACKAGES & DEPENDENCIES
# ==============================================================================
print("⏳ Step 1: Installing AI Models, Servers, and Data Science tools... Please wait.")
!pip install -q transformers easyocr torch pillow requests flask flask-cors pyngrok pandas matplotlib seaborn

# ==============================================================================
# STEP 2: IMPORT LIBRARIES & INITIALIZE CORE ENGINE (ML / AI / NLP)
# ==============================================================================
import io
import json
import torch
import requests
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok
import easyocr
from transformers import pipeline
from google.colab import files
from IPython.display import display, HTML

print("\n🤖 Step 2: Loading Multimodal ML & NLP Models into GPU/CPU Memory...")
device_id = 0 if torch.cuda.is_available() else -1

# AI Model A: EasyOCR Text Extraction (Computer Vision)
ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

# AI Model B: BART Zero-Shot Sentiment & Offensive Text Filter (NLP)
text_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device_id)

# AI Model C: CLIP Zero-Shot Image Theme Categorization (Computer Vision)
image_classifier = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32", device=device_id)
print("✅ Core Models Fully Loaded and Optimized!")

# ==============================================================================
# STEP 3: DATABASE DESIGN & CORE PREDICTIVE RECOMMENDATION ENGINE LOGIC
# ==============================================================================
columns = ['user_id', 'age_group', 'action', 'watch_time', 'genre', 'sentiment', 'is_offensive']
activity_db = pd.DataFrame(columns=columns)

# Injecting clean sample crowd-sourced data so your charts look amazing instantly
mock_data = [
    {"user_id": "U101", "age_group": "13-18", "action": "share", "watch_time": 8.5, "genre": "Gaming", "sentiment": "positive", "is_offensive": "safe"},
    {"user_id": "U102", "age_group": "19-25", "action": "like", "watch_time": 4.2, "genre": "Tech/Coding", "sentiment": "positive", "is_offensive": "safe"},
    {"user_id": "U103", "age_group": "26-35", "action": "view", "watch_time": 12.0, "genre": "Corporate/Work", "sentiment": "negative", "is_offensive": "safe"},
    {"user_id": "U104", "age_group": "19-25", "action": "view", "watch_time": 1.1, "genre": "Political", "sentiment": "negative", "is_offensive": "offensive"},
    {"user_id": "U105", "age_group": "36-50", "action": "share", "watch_time": 6.8, "genre": "Dad Jokes", "sentiment": "positive", "is_offensive": "safe"},
    {"user_id": "U106", "age_group": "19-25", "action": "like", "watch_time": 9.3, "genre": "Dark Humor", "sentiment": "negative", "is_offensive": "offensive"},
    {"user_id": "U107", "age_group": "13-18", "action": "view", "watch_time": 0.5, "genre": "Tech/Coding", "sentiment": "neutral", "is_offensive": "safe"},
]
activity_db = pd.concat([activity_db, pd.DataFrame(mock_data)], ignore_index=True)

user_profiles = {
    "Default_User": {"preferred_genres": {}, "age_group": "19-25"}
}

def get_recommendation(user_id):
    if user_id not in user_profiles:
        return "Trending General Memes"
    preferences = user_profiles[user_id]["preferred_genres"]
    if not preferences:
        return "Tech/Coding"
    return max(preferences, key=preferences.get)

# ==============================================================================
# STEP 4: FLASK ENDPOINT MAIN SERVICE PIPELINE (FAST FILTER OPTIMIZED)
# ==============================================================================
app = Flask(__name__)
CORS(app)

@app.route('/api/analyze-meme', methods=['POST'])
def process_meme_stream():
    global activity_db
    data = request.json
    image_url = data.get('image_url')
    duration = float(data.get('duration', 0))
    action = data.get('action', 'view')

    current_user = "Default_User"
    user_age = user_profiles[current_user]["age_group"]

    # SPEED FIX 1: If the user just scrolled past fast (less than 5 seconds) and didn't click like/share, skip completely!
    if action == 'view' and duration < 5.0:
        return jsonify({"status": "ignored", "message": "Fast scroll ignored to save processing time."}), 200

    print(f"\n📥 Processing Target Match -> Action: {action.upper()} | Watch Time: {duration}s")

    try:
        response = requests.get(image_url, timeout=5)
        image_bytes = response.content
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 1. Image Theme Categorization using OpenAI CLIP
        genre_labels = ["Tech/Coding", "Gaming", "Corporate/Work", "Political", "Dark Humor", "Dad Jokes"]
        img_res = image_classifier(img, candidate_labels=genre_labels)
        detected_genre = img_res[0]['label']

        # 2. Text Extraction via EasyOCR
        ocr_results = ocr_reader.readtext(image_bytes)
        extracted_text = " ".join([text[1] for text in ocr_results]).strip()

        # 3. Text NLP Sentiment & Offensive Trigger Filter via BART
        detected_sentiment = "neutral"
        safety_status = "safe"

        # SPEED FIX 2: Only spin up the NLP pipeline if text actually exists
        if extracted_text:
            sent_res = text_classifier(extracted_text, candidate_labels=["positive", "negative", "neutral"])
            detected_sentiment = sent_res['labels'][0]

            safety_res = text_classifier(extracted_text, candidate_labels=["safe clean humor", "offensive hate speech toxic"])
            if safety_res['labels'][0] == "offensive hate speech toxic" and safety_res['scores'][0] > 0.75:
                safety_status = "offensive"

        # 4. Behavioral Point Adjustments Matrix Logic
        weight = 1.0
        if action == "share": weight = 3.0
        elif action == "like": weight = 2.0
        elif action == "view" and duration >= 5.0: weight = 1.5

        if detected_genre not in user_profiles[current_user]["preferred_genres"]:
            user_profiles[current_user]["preferred_genres"][detected_genre] = 0
        user_profiles[current_user]["preferred_genres"][detected_genre] += weight

        # 5. Commit log telemetry into Database
        new_row = {
            "user_id": current_user, "age_group": user_age, "action": action,
            "watch_time": duration, "genre": detected_genre,
            "sentiment": detected_sentiment, "is_offensive": safety_status
        }
        activity_db = pd.concat([activity_db, pd.DataFrame([new_row])], ignore_index=True)
        next_suggested_genre = get_recommendation(current_user)

        print(f"🎯 ANALYSIS COMPLETE:")
        print(f"   ↳ Read Content : \"{extracted_text[:40]}...\"")
        print(f"   ↳ Classification: Genre={detected_genre} | Tone={detected_sentiment} | Status={safety_status.upper()}")
        print(f"   ↳ Next Up Recommendation for User: [ {next_suggested_genre.upper()} ]")

        return jsonify({
            "status": "success",
            "detected_genre": detected_genre,
            "safety_flag": safety_status,
            "next_recommendation": next_suggested_genre
        }), 200

    except Exception as e:
        print(f"   ❌ Pipeline Internal Process Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# ==============================================================================
# STEP 5: BACKEND DASHBOARD CHART GENERATOR
# ==============================================================================
@app.route('/generate-dashboard')
def build_charts():
    global activity_db
    if activity_db.empty:
        return "No data recorded yet."

    with app.app_context():
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        sns.set_theme(style="darkgrid")

        sns.countplot(ax=axes[0,0], data=activity_db, x='genre', palette='viridis', hue='genre', legend=False)
        axes[0,0].set_title("1. Meme Genre Popularity Analysis")
        axes[0,0].tick_params(axis='x', rotation=25)

        sns.barplot(ax=axes[0,1], data=activity_db, x='genre', y='watch_time', hue='action', errorbar=None, palette='magma')
        axes[0,1].set_title("2. User Engagement Dynamics by Action Style")
        axes[0,1].tick_params(axis='x', rotation=25)

        sns.countplot(ax=axes[1,0], data=activity_db, x='age_group', hue='genre', palette='tab10')
        axes[1,0].set_title("3. Audience Demographics Segmentation Matrix")

        sns.countplot(ax=axes[1,1], data=activity_db, x='genre', hue='is_offensive', palette='rocket')
        axes[1,1].set_title("4. Content Modality Safety Filter (Offensive Triggers)")
        axes[1,1].tick_params(axis='x', rotation=25)

        plt.tight_layout()
        plt.savefig('dashboard.png')
        plt.close()
    return "Dashboard updated successfully!"

# ==============================================================================
# STEP 6: COMPILING FRONTEND CHROME BROWSER EXTENSION PACKAGE FILES
# ==============================================================================
manifest_json = {
  "manifest_version": 3, "name": "Meme Advanced Sentiment Analytics", "version": "2.0",
  "description": "Enterprise Grade User Behavior Sentiment Tracker.",
  "permissions": ["activeTab"],
  "host_permissions": ["http://*.ngrok-free.app/*", "https://*.ngrok-free.app/*"],
  "content_scripts": [{"matches": ["https://*.instagram.com/*"], "js": ["content.js"]}]
}

content_js_code = """
let currentMemeUrl = null; let watchStartTime = null;
const BACKEND_URL = "TUNNEL_URL_PLACEHOLDER";

const screenWatcherOptions = { root: null, threshold: 0.6 };
const postWatcher = new IntersectionObserver(function(entries) {
    entries.forEach(function(entry) {
        let imageFound = entry.target.querySelector('img');
        if (imageFound === null) return;
        let imageLink = imageFound.src;

        if (entry.isIntersecting === true) {
            watchStartTime = Date.now(); currentMemeUrl = imageLink;
        } else {
            if (currentMemeUrl === imageLink && watchStartTime !== null) {
                let timeSpentInSeconds = (Date.now() - watchStartTime) / 1000;
                sendDataToDashboard("view", timeSpentInSeconds, currentMemeUrl);
                watchStartTime = null; currentMemeUrl = null;
            }
        }
    });
}, screenWatcherOptions);

document.addEventListener('click', function(event) {
    let clickedElement = event.target;
    let postBox = clickedElement.closest('article');
    if (postBox === null) return;
    let imageLink = postBox.querySelector('img') ? postBox.querySelector('img').src : "No URL";

    if (clickedElement.closest('svg[aria-label="Like"]') || clickedElement.closest('svg[aria-label="Unlike"]')) {
        sendDataToDashboard("like", 0, imageLink);
    }
    if (clickedElement.closest('svg[aria-label="Share Post"]')) {
        sendDataToDashboard("share", 0, imageLink);
    }
});

function sendDataToDashboard(actionName, duration, imageUrl) {
    fetch(BACKEND_URL + '/api/analyze-meme', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ action: actionName, duration: duration, image_url: imageUrl })
    })
    .then(res => res.json())
    .then(data => {
        console.log("🔥 ALGORITHM RESPONSE:", data);
    }).catch(err => console.error("❌ Network Pipeline Interrupted"));
}

function findPostsOnScreen() {
    let allPosts = document.querySelectorAll('article');
    allPosts.forEach(p => postWatcher.observe(p));
}
setInterval(findPostsOnScreen, 3000);
"""

# ==============================================================================
# STEP 7: SECURE NETWORK TUNNEL ACTIVATION & SERVER DEPLOYMENT
# ==============================================================================
# CHANGE THIS LINE BELOW WITH YOUR ACTUAL NGROK TOKEN 
YOUR_AUTHTOKEN = "PASTE_YOUR_NGROK_TOKEN_HERE"
ngrok.set_auth_token(YOUR_AUTHTOKEN)

public_url = ngrok.connect(5000).public_url
final_extension_js = content_js_code.replace("TUNNEL_URL_PLACEHOLDER", public_url)

with open('manifest.json', 'w') as f: json.dump(manifest_json, f, indent=2)
with open('content.js', 'w') as f: f.write(final_extension_js)
with zipfile.ZipFile('MemeSentimentExtension.zip', 'w') as zipf:
    zipf.write('manifest.json'); zipf.write('content.js')

print("\n" + "="*60)
print(f"🌍 LIVE PUBLIC TUNNEL ROUTE LINK: {public_url}")
print("="*60)

print("\n📦 Setup Complete! Download the updated 'MemeSentimentExtension.zip' from the folder icon panel.")
print("🔥 Server Active. Go scroll Instagram now!\n")
app.run(port=5000)