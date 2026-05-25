import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("🚀 Local Database Analytics Engine...")
print("-" * 50)

# 1. Check if the database exists and has data. If not, generate a beautiful mock dataset instantly!
# FIX: Checking 'activity_db' globally so it seamlessly reads from your Flask server cell
if 'activity_db' not in globals() or activity_db.empty:
    print("💡 Notice: Live database was empty. Generating presentation data matrix automatically...")

    # Complete, balanced sample data representing your project's data structures
    presentation_data = [
        {"user_id": "U101", "age_group": "19-25", "action": "share", "watch_time": 8.5, "genre": "Gaming", "sentiment": "positive", "is_offensive": "safe"},
        {"user_id": "U102", "age_group": "19-25", "action": "like", "watch_time": 6.2, "genre": "Tech/Coding", "sentiment": "positive", "is_offensive": "safe"},
        {"user_id": "U103", "age_group": "26-35", "action": "view", "watch_time": 12.0, "genre": "Corporate/Work", "sentiment": "negative", "is_offensive": "safe"},
        {"user_id": "U104", "age_group": "19-25", "action": "view", "watch_time": 1.1, "genre": "Political", "sentiment": "negative", "is_offensive": "offensive"},
        {"user_id": "U105", "age_group": "36-50", "action": "share", "watch_time": 7.8, "genre": "Dad Jokes", "sentiment": "positive", "is_offensive": "safe"},
        {"user_id": "U106", "age_group": "19-25", "action": "like", "watch_time": 9.3, "genre": "Dark Humor", "sentiment": "negative", "is_offensive": "offensive"},
        {"user_id": "U107", "age_group": "19-25", "action": "view", "watch_time": 5.5, "genre": "Tech/Coding", "sentiment": "neutral", "is_offensive": "safe"},
        {"user_id": "U108", "age_group": "26-35", "action": "like", "watch_time": 4.1, "genre": "Gaming", "sentiment": "positive", "is_offensive": "safe"},
        {"user_id": "U109", "age_group": "36-50", "action": "view", "watch_time": 3.2, "genre": "Corporate/Work", "sentiment": "neutral", "is_offensive": "safe"},
        {"user_id": "U110", "age_group": "19-25", "action": "share", "watch_time": 11.4, "genre": "Tech/Coding", "sentiment": "positive", "is_offensive": "safe"}
    ]
    current_db = pd.DataFrame(presentation_data)
else:
    current_db = activity_db
    print("🔥 Syncing live telemetry database directly from the API stream!")

# 2. Print out the data matrix summary stats
print("📊 PRESENTATION DATABASE SUMMARY STATS:")
print(f"   Total Interactions Recorded : {len(current_db)}")
print(f"   Most Frequent Meme Genre    : {current_db['genre'].mode()[0]}")
print(f"   Average Linger Time (Sec)   : {current_db['watch_time'].mean():.2f}s")
print(f"   Flagged Offensive Elements  : {len(current_db[current_db['is_offensive'] == 'offensive'])}")
print("-" * 50)

print("🎨 Rendering Visual Distribution Graphs...")

# 3. Build and display the 4-panel dashboard layout instantly
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.set_theme(style="darkgrid")

# Graph 1: Genre Popularity Analysis
sns.countplot(ax=axes[0,0], data=current_db, x='genre', palette='viridis', hue='genre', legend=False)
axes[0,0].set_title("1. Meme Genre Popularity Analysis")
axes[0,0].tick_params(axis='x', rotation=20)

# Graph 2: User Engagement Style
sns.barplot(ax=axes[0,1], data=current_db, x='genre', y='watch_time', hue='action', errorbar=None, palette='magma')
axes[0,1].set_title("2. User Engagement Dynamics by Action Style")
axes[0,1].tick_params(axis='x', rotation=20)

# Graph 3: Age Demographic Segmentation
sns.countplot(ax=axes[1,0], data=current_db, x='age_group', hue='genre', palette='tab10')
axes[1,0].set_title("3. Audience Demographics Segmentation Matrix")

# Graph 4: Offensive Filter Analysis
sns.countplot(ax=axes[1,1], data=current_db, x='genre', hue='is_offensive', palette='rocket')
axes[1,1].set_title("4. Content Modality Safety Filter (Offensive Triggers)")
axes[1,1].tick_params(axis='x', rotation=20)

plt.tight_layout()
plt.show()