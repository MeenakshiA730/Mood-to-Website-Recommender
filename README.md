Mood-to-Website Recommender with User Journey Heatmaps

A full-stack web application that tracks how a user’s mood influences browsing patterns, generates personalized website recommendations, and visualizes on-page activity with heatmaps. The system also provides an interactive analytics dashboard with insights into engagement trends.

Features

Mood-based Website Recommendations

Users input their current mood (happy, bored, stressed, curious).

A simple ML model (k-NN / Decision Tree) suggests websites/content categories based on mood-to-preference data.

Real-Time User Journey Tracking

Logs pages visited, time spent, scroll depth, and click coordinates.

Stores activity in a structured database (SQLite / PostgreSQL).

Heatmap Generation

Visual heatmaps of click interactions on webpages using matplotlib/seaborn.

Helps identify which webpage sections attract the most engagement.

Engagement Analytics Dashboard (Streamlit / Plotly Dash)

Average session duration

Most-clicked elements

Scroll activity distribution

Mood vs. engagement correlation

Personalized Insights

Example: “When you’re bored, you spend more time on long-read articles.”

Example: “When you’re happy, you click more on videos and memes.”

Tech Stack

Frontend:

Streamlit (UI, dashboard)

Bootstrap (styling)

Backend:

Flask / FastAPI (API & data logging)

SQLite / PostgreSQL (database for analytics)

Data Processing & ML:

pandas, numpy

scikit-learn (ML recommendation engine: k-NN / Decision Tree)

Visualization:

matplotlib, seaborn, plotly

OpenCV (for click coordinate heatmaps)
