import json

def load_recent_feedback(path="data/feedback.jsonl", limit=10):
    feedback = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                feedback.append(json.loads(line))
    except FileNotFoundError:
        return []

    return feedback[-limit:]

def summarize_feedback(feedback_items):
    summary = {
        "positive": [],
        "negative": []
    }

    for item in feedback_items:
        entry = f'- "{item["suggestion"]}" (score={item["context"]["similarity_score"]})'
        if item["rating"] == "positive":
            summary["positive"].append(entry)
        else:
            summary["negative"].append(entry)

    return summary

