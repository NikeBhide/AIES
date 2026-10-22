# Group Hangout Recommender (Bangalore)

An AI-based activity planner that recommends things to do based on mood, budget, interests and group size. It focuses on **group recommendations**, where members often want different things.

Built as a mini project for the AIES course.

## How it works

1. **Data**: Bored API activities, 5,109 Bangalore restaurants (Zomato/Kaggle) and 393 Bangalore venues (OpenStreetMap) are merged and collapsed into 219 generic activity types such as "Try Chinese food" or "Watch a movie". Each type carries up to 3 real sample venues.
2. **Mood tagging**: rule-based keyword and category matching assigns each activity one or more of 8 moods: relaxed, energetic, social, adventurous, creative, competitive, romantic, reflective.
3. **Filtering**: hard limits (budget tier, group size, indoor/outdoor) remove unsuitable activities before any ranking.
4. **Matching**: TF-IDF vectors and cosine similarity rank activities against a person's free-text mood description.
5. **Group aggregation**: each member's scores are combined using three strategies that you can compare side by side:
   - **Average**: maximises the group's mean happiness
   - **Least Misery**: maximises the unhappiest member's score
   - **Borda Count**: each member ranks the options and rank points are summed

## AI concepts used

- Content-based recommendation
- TF-IDF and cosine similarity (information retrieval)
- Constraint filtering followed by soft ranking
- Group recommender systems and social-choice aggregation
- Rule-based knowledge engineering (mood tagging)
- Multi-source data fusion

## Project structure

```
AIES/
├── data/        # unified_activities.csv, activity_types.csv, activity_types_tagged.csv
├── scripts/     # one-time data prep: merge_pipeline.py, build_activity_types.py, tag_moods.py
├── app/         # recommender engine: vectorizer.py, filters.py, single_user.py, group.py, demo.py
└── web/         # hangout_finder.html (standalone browser UI)
```

## Getting started

Requires Python 3.9+. No API keys needed, and everything runs offline.

```bash
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
cd app
python demo.py
```

The demo runs one single-user example and one 3-person group example with deliberately mismatched moods, so you can see the three strategies disagree.

### Use it in code

```python
from group import GroupMember, recommend_for_group

members = [
    GroupMember("Riya",  "broke this week, want something cheap and relaxing"),
    GroupMember("Arjun", "high energy, want to be outside and active"),
]
result_df, score_table = recommend_for_group(
    members, max_budget_tier="low", total_group_size=2, strategy="least_misery", top_k=5
)
print(result_df[["name", "category", "group_score"]])
```

### Web app

Open `web/hangout_finder.html` in any browser. Solo mode ranks activities for one person. Group mode compares all three strategies with a per-member score bar on each card. It is a JavaScript reimplementation that scores by mood-tag overlap rather than TF-IDF, so its rankings can differ from the Python version.

## Known limitations

- Food dominates the raw data (5,109 of 5,698 rows), and outdoor and shopping types are thin.
- TF-IDF only matches exact words, so "relaxing" does not match "relaxed". A member whose text shares no words with the activities scores 0 everywhere, which can make Least Misery return ties. Sentence embeddings (e.g. `all-MiniLM-L6-v2`) would fix this.
- Mood tags are rule-based, not learned.
- No distance filtering and no multi-stop itineraries.

## Future work

- Swap TF-IDF for sentence embeddings
- Build a labelled set of group scenarios and evaluate Precision@k / NDCG@k per strategy
- Add a Streamlit UI
- Run a small user study with real friend groups
