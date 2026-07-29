# Tech Stack Recommender — Project 3: AI Recommendation Logic

A content-based filtering engine (DecodeLabs Project 3) that maps a user's
raw skills to the closest job roles using **TF-IDF weighting + Cosine
Similarity** — no external ML libraries, pure algorithmic logic.

## Files

- `raw_skills.csv` — the "items" dataset: 16 job roles, each with a tag list
  of representative skills.
- `recommender.py` — the full IPO pipeline:
  1. **Ingestion** — collects at least 3 user skills.
  2. **Vector Mapping** — builds a shared vocabulary and TF-IDF vectors for
     both the user profile and every job role.
  3. **Scoring** — computes Cosine Similarity between the user vector and
     each item vector.
  4. **Sorting & Filtering** — ranks all roles and returns the Top-3.
  5. **Cold Start fallback** — if the user's skills share nothing with the
     vocabulary, a trending list is shown instead of an empty result.

## How to run

```bash
python3 recommender.py
```

You'll be prompted to enter at least 3 skills, comma-separated, e.g.:

```
Your skills: Python, Cloud Computing, Automation
```

Output:

```
--- Top 3 Recommended Career Paths ---
1. Site Reliability Engineer  (match: 37.3%)
   key skills: linux, kubernetes, automation, monitoring, ci/cd
2. Cloud Architect  (match: 24.9%)
   key skills: aws, azure, cloud computing, networking, security
3. Cybersecurity Analyst  (match: 24.6%)
   key skills: security, networking, linux, penetration testing, cryptography
```

## Extending it

- Add more roles/tags to `raw_skills.csv` — no code changes needed.
- Swap the CSV for a movies/products dataset to reuse the same engine for
  any content-based recommendation use case (the slides' Netflix/Amazon
  example).
- Try an "onboarding survey" style input instead of free text to avoid
  vocabulary mismatches (e.g. "Web Design" vs "Frontend Development").
  
