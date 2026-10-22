# Group Hangout Recommender — Bangalore

An AI-based activity planner that recommends things to do based on mood,
budget, interest, and group size — with a focus on GROUP recommendations,
where members often disagree.

## What's actually in this project

- A real merged dataset: Bored API's generic activity taxonomy (196 entries)
  + 5,109 real Bangalore restaurants (Zomato/Kaggle) + 393 real Bangalore
  venues from OpenStreetMap (cinemas, parks, museums, gyms, malls, etc.)
- Collapsed into 219 generic, recommendable **activity types** (e.g. "Try
  Chinese food", "Watch a movie") each carrying up to 3 real sample venues,
  rather than exact-location matching
- Rule-based mood/energy tagging across 8 mood tags: relaxed, energetic,
  social, adventurous, creative, competitive, romantic, reflective
- TF-IDF-based semantic matching (mood/interest text -> ranked activities)
- Hard-constraint filtering (budget tier, group size, indoor/outdoor)
- Three group-aggregation strategies you can compare head-to-head:
  **Average**, **Least Misery**, **Borda Count**

## Live web app

`web/hangout_finder.html` is a complete, working, self-contained website —
just open it in any browser (double-click it, or drag it into a browser
window). No install, no server, no internet connection needed once opened.

- **Solo mode**: pick mood chips + interests + budget + group size -> ranked
  activity cards with real Bangalore venue examples
- **Group mode**: add members, each picks their own moods -> compare
  Average / Least Misery / Borda Count side by side, with a per-member score
  bar on every card

Note: this is a **JavaScript reimplementation** of the matching logic, not
the Python pipeline running in-browser (a static HTML page can't execute
Python). It uses a mood-tag/category overlap scorer instead of the Python
side's TF-IDF, but applies the same filtering (budget, group size) and the
same three group-aggregation strategies. For your report: frame the Python
code (`app/`) as the matching-engine prototype you built and validated, and
the website as the UI layer built on the same design. `web/activities_embedded_reference.json`
is the exact dataset embedded in the page, kept here for reference/reuse.

## Setup (Python pipeline / app/ folder)

### 1. Requirements
- Python 3.9 or newer (check with `python3 --version`)
- No GPU needed, no API keys needed — everything runs locally and offline

### 2. Install
```bash
cd AIES
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the demo
```bash
cd app
python3 demo.py
```
This runs a single-user example and a 3-person group example (with
deliberately mismatched moods/budgets) through all three aggregation
strategies, so you can see them disagree — that disagreement is the point.

## Project structure

```
AIES/
├── data/
│   ├── unified_activities.csv       # raw merged venue-level data (5,698 rows)
│   ├── activity_types.csv           # collapsed generic activity types (219 rows)
│   └── activity_types_tagged.csv    # + mood_tags, energy_level, recommendable flag
├── scripts/                         # one-time data-prep pipeline (run in this order)
│   ├── merge_pipeline.py            # Bored API + Zomato + OSM -> unified_activities.csv
│   ├── build_activity_types.py      # collapse venues -> generic activity types
│   └── tag_moods.py                 # rule-based mood_tags + energy_level
├── app/                             # the Python recommender engine
│   ├── vectorizer.py                # loads data, builds TF-IDF embeddings
│   ├── filters.py                   # hard constraints: budget, group size, weather
│   ├── single_user.py               # one person: mood -> filtered, ranked list
│   ├── group.py                     # group: Average / Least Misery / Borda compared
│   └── demo.py                      # runs both end to end — START HERE for the Python side
└── web/                             # the finished website — OPEN THIS FOR THE UI
    ├── hangout_finder.html          # double-click to open in any browser, no install
    └── activities_embedded_reference.json  # the data baked into the page, for reference
```

## How to use it in your own code

```python
from vectorizer import load_activities, ActivityIndex
from group import GroupMember, recommend_for_group

members = [
    GroupMember('Riya',  "broke this week, want something cheap and relaxing"),
    GroupMember('Arjun', "high energy, want to be outside and active"),
]

result_df, score_table = recommend_for_group(
    members,
    max_budget_tier='low',      # free / low / medium / high / premium
    total_group_size=2,
    weather_ok_for_outdoor=True,
    strategy='least_misery',    # or 'average' / 'borda'
    top_k=5,
)
print(result_df[['name', 'category', 'group_score']])
```

## Known limitations (be upfront about these in your report)

- **Food dominates the raw venue data** (5,109 of 5,698 rows are Zomato
  restaurants) — solved at the activity-type level by collapsing cuisines
  into 12 generic food types instead of 5,109 individual restaurants, but
  it's still your best-covered category.
- **Outdoor (2 types) and shopping (1 type) are thin** — limited by what
  OpenStreetMap has well-tagged for Bangalore. Worth hand-curating a few
  more if you have time.
- **Mood tagging is currently rule-based** (keyword + category matching),
  not LLM-generated — this is transparent and explainable (good for a
  report), but less nuanced than an LLM pass would be. Swapping in an LLM
  tagging step is a natural "future work" item.
- **TF-IDF is a lexical baseline, not true semantic embeddings** — it
  matches on word overlap, so e.g. "chill" and "relaxed" won't match each
  other unless both words appear in the activity descriptions. Upgrading to
  `sentence-transformers` (see the comment in `vectorizer.py`) is a
  drop-in swap that would meaningfully improve match quality — flagged as
  the most valuable single upgrade if you have time before submission.
- **No real distance/locality filtering** — sample venues are shown for
  context but aren't filtered by proximity to the user, by design decision
  (this was discussed explicitly — see project notes).
- **No itinerary/sequencing** — this recommends a ranked list of options,
  not a multi-stop planned day. Flagged as a stretch goal / future work.

## Suggested next steps
1. Build a small labeled evaluation set (30-50 group scenarios) to compute
   Precision@k / NDCG@k across the three strategies
2. Optionally build a Streamlit UI on top of `app/demo.py` for a live
   clickable interface
3. Run a small user study with real friend groups comparing strategy outputs
