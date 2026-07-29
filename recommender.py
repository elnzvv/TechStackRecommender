import csv
import math
import os
from collections import Counter



def load_items(csv_path):
    """Load job roles ('items') and their tag lists from raw_skills.csv."""
    items = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            role = row["job_role"].strip()
            tags = [t.strip().lower() for t in row["skills"].split("|") if t.strip()]
            items.append({"role": role, "tags": tags})
    return items


def get_user_skills(min_inputs=3):
    """
    Capture the user state (Step 1 of the IPO model).
    Requires a minimum of `min_inputs` skills, per the project spec.
    """
    print("=== DecodeLabs Tech Stack Recommender ===")
    print(f"Enter at least {min_inputs} skills or interests (comma-separated).")
    print("Example: Python, Cloud Computing, Automation\n")

    raw = input("Your skills: ").strip()
    skills = [s.strip().lower() for s in raw.split(",") if s.strip()]

    while len(skills) < min_inputs:
        print(f"Please enter at least {min_inputs} skills.")
        raw = input("Your skills: ").strip()
        skills = [s.strip().lower() for s in raw.split(",") if s.strip()]

    return skills



def build_vocabulary(items, user_skills):
    """All items' tags plus the user's skills must share one vocabulary space."""
    vocab = set()
    for item in items:
        vocab.update(item["tags"])
    vocab.update(user_skills)
    return sorted(vocab)


def compute_idf(items, vocab):
    """
    IDF = log(Total Documents / Documents containing term t)
    'Documents' here = job role tag-sets (the items).
    """
    n_docs = len(items)
    idf = {}
    for term in vocab:
        doc_count = sum(1 for item in items if term in item["tags"])
        # +1 smoothing avoids division by zero for terms unseen in any item
        idf[term] = math.log(n_docs / doc_count) if doc_count > 0 else 0.0
    return idf


def compute_tf(tag_list, vocab):
    """TF = (count of term in doc) / (total terms in doc)."""
    counts = Counter(tag_list)
    total = len(tag_list) if tag_list else 1
    return {term: counts.get(term, 0) / total for term in vocab}


def build_tfidf_vector(tag_list, vocab, idf):
    tf = compute_tf(tag_list, vocab)
    return [tf[term] * idf[term] for term in vocab]




def cosine_similarity(vec_a, vec_b):
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0  # Cold Start: a zero vector can't be scored
    return dot / (mag_a * mag_b)


def score_items(items, item_vectors, user_vector):
    scored = []
    for item, vec in zip(items, item_vectors):
        score = cosine_similarity(user_vector, vec)
        scored.append((item["role"], score, item["tags"]))
    return scored



def rank_top_n(scored_items, n=3):
    ranked = sorted(scored_items, key=lambda x: x[1], reverse=True)
    return ranked[:n]




def trending_fallback(items, n=3):
    """If the user vector matches nothing (all-zero similarity), fall back
    to a simple 'trending' list (here: the first N roles) instead of
    returning an empty result."""
    return [(item["role"], 0.0, item["tags"]) for item in items[:n]]




def recommend(user_skills, items, top_n=3):
    vocab = build_vocabulary(items, user_skills)
    idf = compute_idf(items, vocab)

    item_vectors = [build_tfidf_vector(item["tags"], vocab, idf) for item in items]
    user_vector = build_tfidf_vector(user_skills, vocab, idf)

    scored = score_items(items, item_vectors, user_vector)
    top_matches = rank_top_n(scored, n=top_n)

    if all(score == 0 for _, score, _ in top_matches):
        print("\n[Cold Start detected: no overlapping skills found]")
        print("Showing trending roles instead.\n")
        top_matches = trending_fallback(items, n=top_n)

    return top_matches


def print_results(user_skills, results):
    print("\n--- Your Input ---")
    print(", ".join(user_skills))

    print(f"\n--- Top {len(results)} Recommended Career Paths ---")
    for rank, (role, score, tags) in enumerate(results, start=1):
        pct = round(score * 100, 1)
        print(f"{rank}. {role}  (match: {pct}%)")
        print(f"   key skills: {', '.join(tags[:5])}")
    print()


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "raw_skills.csv")

    items = load_items(csv_path)
    user_skills = get_user_skills(min_inputs=3)

    results = recommend(user_skills, items, top_n=3)
    print_results(user_skills, results)


if __name__ == "__main__":
    main()
