# 📊 AI-Powered Meme Sentiment & Behavior Tracker

An enterprise-grade analytics pipeline that connects a Google Chrome/Instagram extension to a live machine learning backend server. It tracks user behavior metrics (hover duration, likes, shares) on Instagram memes, processes the content using advanced Computer Vision (CLIP, EasyOCR) and NLP models (BART), and maps real-time data onto an interactive dashboard.

---

## 🚀 How the System Architecture Works

* **The Extension (Frontend):** Injects a script into Instagram to track when memes are on screen and logs active engagement like clicks, shares, and watch telemetry.
* **The Server Core (`app.py`):** A Flask backend exposed via an Ngrok tunnel. It downloads image URLs, extracts embedded text using **EasyOCR**, and evaluates visual topics and tone via **OpenAI CLIP** and **BART Zero-Shot Classifiers**.
* **The Dashboard (`analytics_dashboard.py`):** Runs a localized data science script utilizing **Pandas**, **Matplotlib**, and **Seaborn** to plot user interaction distributions, genre popularity, and demographic behavior grids.

---

## 🛠️ How to Run the Project

### 1. Launch the Live AI Server
Open the `app.py` file, replace the placeholder token with your actual Ngrok authtoken, and execute the script:
```python
YOUR_AUTHTOKEN = "YOUR_REAL_NGROK_TOKEN"

### 💾 Step 2: Save the Changes

1. Scroll down to the bottom of the page to the **Commit changes** box.
2. Leave the default message (`Create README.md`) or type a custom note like `Add project architecture overview`.
3. Click the green **Commit changes** button.

---

### ✨ The Result

Once you click commit, go back to your main repository link. GitHub will now beautifully format that markdown file right underneath your file upload grid, creating a highly polished, professional landing page for your project repository!
