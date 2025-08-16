import os
import uuid
import sqlite3
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import plotly.graph_objs as go

from utils import generate_click_heatmap

# ----------------- Configuration -----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORE_DIR = os.path.join(BASE_DIR, "store")   # writable folder
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(STORE_DIR, "events.sqlite")
SEED_CSV = os.path.join(DATA_DIR, "mood_site_seed.csv")
HEATMAP_PATH = os.path.join(STORE_DIR, "heatmap_pageA.png")

os.makedirs(STORE_DIR, exist_ok=True)

# ----------------- DB helpers -----------------
def get_db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_conn()
    cur = conn.cursor()
    # original tables
    cur.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
      session_id TEXT PRIMARY KEY,
      start_ts TEXT,
      mood TEXT
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS events (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id TEXT,
      type TEXT,
      page_id TEXT,
      duration_sec REAL,
      max_scroll_depth REAL,
      ts TEXT
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS clicks (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id TEXT,
      page_id TEXT,
      x REAL,
      y REAL,
      ts TEXT
    );
    """)
    # new tables for impressions & recommendation clicks
    cur.execute("""
    CREATE TABLE IF NOT EXISTS impressions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id TEXT,
      site_url TEXT,
      site_category TEXT,
      page_id TEXT,
      ts TEXT
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS rec_clicks (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id TEXT,
      site_url TEXT,
      site_category TEXT,
      page_id TEXT,
      ts TEXT
    );
    """)
    conn.commit()
    conn.close()

init_db()

# ----------------- Recommender -----------------
class MoodRecommender:
    def __init__(self, pipe=None, df=None):
        self.pipe = pipe
        self.df = df

    @staticmethod
    def train(csv_path):
        if not os.path.exists(csv_path):
            # fallback dataset
            df = pd.DataFrame([
                {"mood":"happy","site_category":"memes","site_url":"https://9gag.com","engagement_score":0.9},
                {"mood":"happy","site_category":"video","site_url":"https://www.youtube.com","engagement_score":0.95},
                {"mood":"bored","site_category":"long-reads","site_url":"https://medium.com","engagement_score":0.85},
                {"mood":"stressed","site_category":"meditation","site_url":"https://www.headspace.com","engagement_score":0.9},
                {"mood":"curious","site_category":"learn","site_url":"https://www.khanacademy.org","engagement_score":0.95},
            ])
        else:
            df = pd.read_csv(csv_path)

        X = df[['mood']]
        y = df['site_category']
        ct = ColumnTransformer([("ohe", OneHotEncoder(handle_unknown='ignore'), ['mood'])], remainder='drop')
        knn = KNeighborsClassifier(n_neighbors=1, metric='cosine')
        pipe = Pipeline([("ct", ct), ("knn", knn)])
        pipe.fit(X, y)
        return MoodRecommender(pipe, df)

    def recommend(self, mood, topk=4):
        if self.pipe is None or self.df is None:
            return []
        try:
            pred_cat = self.pipe.predict(pd.DataFrame({'mood':[mood]}))[0]
        except Exception:
            pred_cat = None

        subset = self.df[self.df['mood'] == mood].copy()
        if subset.empty and pred_cat is not None:
            subset = self.df[self.df['site_category'] == pred_cat].copy()
        if subset.empty:
            subset = self.df.copy()

        subset['score'] = subset.get('engagement_score', 1.0) + np.random.rand(len(subset))*0.05
        subset = subset.sort_values('score', ascending=False)
        others = self.df[self.df['mood'] != mood].copy()
        if not others.empty:
            others['score'] = others.get('engagement_score', 0.5) * 0.8 + np.random.rand(len(others))*0.05

        recs = pd.concat([subset, others]).drop_duplicates(subset=['site_url']).head(topk)
        return recs[['site_category','site_url','engagement_score','score']].to_dict(orient='records')

# load recommender
def load_recommender():
    return MoodRecommender.train(SEED_CSV)

recommender = load_recommender()

# ----------------- Session helpers -----------------
def get_session_id():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        # create DB session row
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("INSERT OR IGNORE INTO sessions(session_id, start_ts, mood) VALUES (?,?,?)",
                    (st.session_state.session_id, datetime.utcnow().isoformat(), "unknown"))
        conn.commit()
        conn.close()
    return st.session_state.session_id

def set_mood_for_session(mood):
    sid = get_session_id()
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("UPDATE sessions SET mood=? WHERE session_id=?", (mood, sid))
    conn.commit()
    conn.close()

# ----------------- Logging helpers -----------------
def log_event(event_type, page_id=None, duration_sec=None, max_scroll=None):
    sid = get_session_id()
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO events(session_id,type,page_id,duration_sec,max_scroll_depth,ts) VALUES (?,?,?,?,?,?)",
                (sid, event_type, page_id, duration_sec, max_scroll, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def log_click(x, y, page_id="pageA"):
    sid = get_session_id()
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO clicks(session_id,page_id,x,y,ts) VALUES (?,?,?,?,?)",
                (sid, page_id, float(x), float(y), datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

# New: log when a recommendation is shown (an impression)
def log_impression(site_url, site_category, page_id="recommendations"):
    sid = get_session_id()
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO impressions(session_id,site_url,site_category,page_id,ts) VALUES (?,?,?,?,?)",
                (sid, site_url, site_category, page_id, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

# New: log a recommendation click (site_url & category)
def log_rec_click(site_url, site_category, page_id="recommendations"):
    sid = get_session_id()
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO rec_clicks(session_id,site_url,site_category,page_id,ts) VALUES (?,?,?,?,?)",
                (sid, site_url, site_category, page_id, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

# ----------------- UI Pages -----------------
st.set_page_config(page_title="Mood2Web (Streamlit)", layout="wide")
st.title("Mood-to-Website Recommender (Streamlit)")

menu = st.sidebar.selectbox("Navigation", ["Home", "Demo Content", "Heatmap", "Dashboard", "Reset Session"])

if menu == "Reset Session":
    st.session_state.clear()
    st.success("Session cleared. Reload the page.")
    st.stop()

get_session_id()  # ensure session exists

# ----------------- Home Page -----------------
if menu == "Home":
    st.header("Tell us your mood")
    mood = st.selectbox("How do you feel right now?", ["", "happy", "bored", "stressed", "curious"])
    if st.button("Get Recommendations"):
        if mood == "":
            st.warning("Please pick a mood.")
        else:
            set_mood_for_session(mood)
            recs = recommender.recommend(mood, topk=6)
            st.write(f"Recommendations for mood: **{mood}**")

            # show recommendations as rows with Visit buttons that log impressions & clicks
            for i, r in enumerate(recs):
                site_category = r['site_category']
                site_url = r['site_url']
                score = r.get('score', 0.0)

                # Log an impression for this recommendation
                try:
                    log_impression(site_url, site_category, page_id="home")
                except Exception:
                    # If DB write fails, continue showing the recommendation
                    pass

                cols = st.columns([8,2])
                with cols[0]:
                    st.markdown(f"**{site_category}** — {site_url} (score: {score:.2f})")
                with cols[1]:
                    btn_key = f"visit_{i}_{site_url}"
                    if st.button("Visit", key=btn_key):
                        # Log recommendation click
                        try:
                            log_rec_click(site_url, site_category, page_id="home")
                        except Exception:
                            pass
                        # show link for the user to open in new tab
                        st.markdown(f"[Open site]({site_url})")

            log_event("mood_selected", page_id="home")

    st.markdown("---")
    st.markdown("You can try the demo content page to record clicks and generate a heatmap.")

# ----------------- Demo Content -----------------
elif menu == "Demo Content":
    st.header("Demo Content Page (click the blocks)")
    st.write("Click the buttons below to simulate clicks on different page areas. Each click is logged to the DB.")

    blocks = [
        ("Featured Video", 200, 150),
        ("Long-read Article", 600, 420),
        ("Calm / Mindfulness Widget", 400, 820),
        ("Curiosity Corner", 300, 1220),
    ]

    for title, x_coord, y_coord in blocks:
        if st.button(f"Click: {title}"):
            log_click(x_coord + (np.random.rand()-0.5)*40, y_coord + (np.random.rand()-0.5)*40, page_id="pageA")
            st.success(f"Logged click on: {title}")

    if st.button("End Session (log page_end)"):
        duration = float(np.random.randint(10, 600))
        max_scroll = float(np.random.rand())
        log_event("page_end", page_id="pageA", duration_sec=duration, max_scroll=max_scroll)
        st.info(f"Logged end: duration {duration:.1f}s, scroll {max_scroll:.2f}")

# ----------------- Heatmap -----------------
elif menu == "Heatmap":
    st.header("Heatmap for demo pageA")
    conn = get_db_conn()
    df = pd.read_sql_query("SELECT x,y FROM clicks WHERE page_id=?", conn, params=("pageA",))
    conn.close()
    clicks = df.to_dict(orient='records')
    generate_click_heatmap(clicks, HEATMAP_PATH, canvas_width=1200, canvas_height=1600, bins=60)
    if os.path.exists(HEATMAP_PATH):
        st.image(HEATMAP_PATH, use_column_width=True)
    else:
        st.write("No heatmap available yet (no clicks recorded).")

# ----------------- Dashboard -----------------
elif menu == "Dashboard":
    st.header("Analytics Dashboard")
    conn = get_db_conn()
    ev = pd.read_sql_query("SELECT * FROM events", conn)
    sess = pd.read_sql_query("SELECT * FROM sessions", conn)
    clk = pd.read_sql_query("SELECT * FROM clicks", conn)
    impressions = pd.read_sql_query("SELECT * FROM impressions", conn)
    rec_clicks = pd.read_sql_query("SELECT * FROM rec_clicks", conn)
    conn.close()

    avg_duration = float(ev['duration_sec'].dropna().mean()) if 'duration_sec' in ev.columns and not ev['duration_sec'].dropna().empty else 0.0
    avg_scroll = float(ev['max_scroll_depth'].dropna().mean()) if 'max_scroll_depth' in ev.columns and not ev['max_scroll_depth'].dropna().empty else 0.0
    total_clicks = len(clk)
    sessions_cnt = len(sess)

    st.metric("Avg Session Duration (s)", f"{avg_duration:.1f}")
    st.metric("Avg Scroll Depth", f"{avg_scroll:.2f}")
    st.metric("Total Demo Clicks", f"{total_clicks}")
    st.metric("Sessions", f"{sessions_cnt}")

    st.markdown("---")
    st.subheader("Recommendation CTR (per site)")
    # compute CTR per site_url
    if not impressions.empty:
        imp_counts = impressions.groupby("site_url").size().reset_index(name="impressions")
    else:
        imp_counts = pd.DataFrame(columns=["site_url", "impressions"])

    if not rec_clicks.empty:
        click_counts = rec_clicks.groupby("site_url").size().reset_index(name="clicks")
    else:
        click_counts = pd.DataFrame(columns=["site_url", "clicks"])

    ctr_df = pd.merge(imp_counts, click_counts, on="site_url", how="left").fillna(0)
    if not ctr_df.empty:
        ctr_df['clicks'] = ctr_df['clicks'].astype(int)
        ctr_df['ctr'] = ctr_df.apply(lambda r: (r['clicks'] / r['impressions']) if r['impressions'] > 0 else 0.0, axis=1)
        st.dataframe(ctr_df.sort_values('ctr', ascending=False))
        overall_impressions = ctr_df['impressions'].sum()
        overall_clicks = ctr_df['clicks'].sum()
        overall_ctr = (overall_clicks / overall_impressions) if overall_impressions > 0 else 0.0
        st.metric("Overall CTR", f"{overall_ctr:.2%}")
    else:
        st.write("No recommendation impressions/clicks yet.")

    st.markdown("---")
    st.subheader("Mood vs Avg Duration")
    if not ev.empty and not sess.empty:
        ev_end = ev[ev['type'] == 'page_end'][['session_id','duration_sec']]
        mood_map = sess[['session_id','mood']]
        md = ev_end.merge(mood_map, on='session_id', how='left')
        mood_g = md.groupby('mood', dropna=True)['duration_sec'].mean().reset_index()
        trace = go.Bar(x=mood_g['mood'].tolist(), y=mood_g['duration_sec'].tolist())
        st.plotly_chart({"data":[trace], "layout":go.Layout(xaxis_title="Mood", yaxis_title="Avg Duration (s)")}, use_container_width=True)
    else:
        st.write("Not enough data to render mood-duration chart.")

    st.markdown("---")
    st.subheader("Clicks Over Time")
    if not clk.empty:
        clk['ts'] = pd.to_datetime(clk['ts'])
        time_series = clk.groupby(clk['ts'].dt.floor("T")).size().reset_index(name="clicks")
        st.line_chart(time_series.set_index("ts"))
    else:
        st.write("No click data yet.")

    st.markdown("---")
    st.subheader("Click Distribution by Mood")
    if not sess.empty and not clk.empty:
        mood_clicks = clk.merge(sess[['session_id','mood']], on="session_id", how="left")
        mood_clicks_g = mood_clicks.groupby("mood").size().reset_index(name="clicks")
        st.bar_chart(mood_clicks_g.set_index("mood"))
    else:
        st.write("No mood-click data yet.")

    st.markdown("---")
    st.subheader("Raw Tables")
    st.write("Events (latest 200):")
    if not ev.empty:
        st.dataframe(ev.sort_values('ts', ascending=False).head(200))
    else:
        st.write("No events yet.")
    st.write("Impressions (latest 200):")
    if not impressions.empty:
        st.dataframe(impressions.sort_values('ts', ascending=False).head(200))
    else:
        st.write("No impressions yet.")
    st.write("Recommendation Clicks (latest 200):")
    if not rec_clicks.empty:
        st.dataframe(rec_clicks.sort_values('ts', ascending=False).head(200))
    else:
        st.write("No recommendation clicks yet.")
    st.write("Raw Clicks (latest 200):")
    if not clk.empty:
        st.dataframe(clk.sort_values('ts', ascending=False).head(200))
    else:
        st.write("No clicks recorded yet.")
