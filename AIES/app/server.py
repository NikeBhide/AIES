"""
Local backend for the Hangout Finder website.

Run from the project's app/ folder:   python server.py
Then open:                            http://127.0.0.1:8080/

What it does: the web page sends each person's mood text to /api/scores and
gets back a similarity score for every activity, computed by the same
ActivityIndex used by demo.py (sentence embeddings by default). The page still
does the budget/group-size filtering and the Average / Least Misery / Borda
voting itself. If the embedding model can't load (e.g. no internet on first
run), it falls back to TF-IDF and says so in /api/health.
"""
import os
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from vectorizer import load_activities, ActivityIndex

WEB_DIR = Path(__file__).resolve().parent.parent / 'web'
METHOD = os.environ.get('HANGOUT_METHOD', 'embedding')

app = Flask(__name__, static_folder=None)

df = load_activities()
try:
    index = ActivityIndex(df, method=METHOD)
    active_method = METHOD
except Exception as exc:  # model missing / no internet on first run
    print(f"[warn] could not start '{METHOD}' scoring ({exc}); falling back to tfidf")
    index = ActivityIndex(df, method='tfidf')
    active_method = 'tfidf'
print(f"Scoring method: {active_method}  |  {len(df)} activities loaded")


@app.get('/api/health')
def health():
    return jsonify(ok=True, method=active_method, activities=len(df))


@app.post('/api/scores')
def scores():
    """Body: {"queries": ["relaxed social", ...]}
    Returns: {"method": "...", "scores": [{activity name: score, ...}, ...]}
    one score-dict per query, in the same order."""
    body = request.get_json(silent=True) or {}
    queries = body.get('queries')
    if not isinstance(queries, list) or not queries or len(queries) > 50:
        return jsonify(error="'queries' must be a non-empty list (max 50)"), 400
    names = df['name'].tolist()
    out = []
    for q in queries:
        s = index.score(str(q)[:500])
        out.append({n: round(float(v), 4) for n, v in zip(names, s)})
    return jsonify(method=active_method, scores=out)


@app.get('/')
def root():
    return send_from_directory(WEB_DIR, 'index.html')


@app.get('/<path:filename>')
def static_files(filename):
    return send_from_directory(WEB_DIR, filename)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=int(os.environ.get('PORT', 8080)), debug=False)
