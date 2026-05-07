
import json
from datetime import datetime

def save_feedback(suggestion, rating, context):
    feedback = {
        "suggestion": suggestion,
        "rating": rating,
        "context": context,
        "timestamp": datetime.utcnow().isoformat()
    }

    with open("data/feedback.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(feedback) + "\n")
        print("Feedback saved")
