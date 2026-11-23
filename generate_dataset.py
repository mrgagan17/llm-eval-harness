import json
from pathlib import Path

BASE_PROMPTS = [
    ("coding", "Write a Python one-liner to reverse a list `nums`.", "nums[::-1]"),
    ("coding", "What is the average time complexity of a hash table lookup?", "O(1) constant time."),
    ("coding", "In Python, what is the difference between a tuple and a list?", "Lists are mutable; tuples are immutable."),
    ("reasoning", "If 3 machines make 3 widgets in 3 minutes, how long do 100 machines take to make 100 widgets?", "3 minutes."),
    ("reasoning", "A bat and a ball cost $1.10 total. The bat costs $1.00 more than the ball. How much does the ball cost?", "$0.05 (5 cents)."),
    ("extraction", "Extract the email and date as JSON: 'Contact Priya at priya@acme.io before 2026-12-01.'", '{"email": "priya@acme.io", "date": "2026-12-01"}'),
    ("extraction", "Extract the total price: 'Subtotal $40, Tax $4, Grand Total $44.00.'", "$44.00"),
    ("sql", "Write a SQL query to count rows in a table named `orders`.", "SELECT COUNT(*) FROM orders;"),
    ("sql", "What SQL clause filters grouped aggregate results?", "HAVING"),
    ("summarization", "Summarize in 10 words or fewer: 'HTTP 429 means the client sent too many requests in a given time.'", "HTTP 429 indicates the client exceeded the rate limit."),
]

def build_50_dataset() -> None:
    Path("data").mkdir(exist_ok=True)
    dataset = []
    for i in range(50):
        cat, prompt, ref = BASE_PROMPTS[i % len(BASE_PROMPTS)]
        suffix = "" if i < len(BASE_PROMPTS) else f" (Variant #{i // len(BASE_PROMPTS) + 1})"
        dataset.append(
            {
                "id": i + 1,
                "category": cat,
                "prompt": f"{prompt}{suffix}",
                "reference_answer": ref,
            }
        )
    Path("data/benchmark_50.json").write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    print("Generated data/benchmark_50.json with 50 prompts.")

if __name__ == "__main__":
    build_50_dataset()
