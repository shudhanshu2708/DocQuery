import json
import requests
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

API_URL = "http://127.0.0.1:8000/ask"

judge = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)


with open("evaluation/questions.json", "r", encoding="utf-8") as file:
    questions = json.load(file)


def get_answer(item):
    try:
        response = requests.post(
            API_URL,
            json={"question": item["question"]},
            timeout=120
        )

        if response.status_code != 200:
            print(
                f"Q{item['id']} failed - "
                f"HTTP {response.status_code}: {response.text[:200]}"
            )

            return {
                "id": item["id"],
                "question": item["question"],
                "expected": item["expected"],
                "actual": None
            }

        data = response.json()

        return {
            "id": item["id"],
            "question": item["question"],
            "expected": item["expected"],
            "actual": data["answer"]
        }

    except Exception as e:

        print(f"Q{item['id']} failed: {e}")

        return {
            "id": item["id"],
            "question": item["question"],
            "expected": item["expected"],
            "actual": None
        }


def judge_answer(item):

    prompt = f"""
You are evaluating a document question-answering system.

Question:
{item["question"]}

Expected answer:
{item["expected"]}

Generated answer:
{item["actual"]}

Decide whether the generated answer is semantically correct.

Rules:

1. CORRECT if the generated answer has the same meaning as the expected answer.

2. CORRECT if the generated answer contains additional relevant
information that does not contradict the expected answer.

3. INCORRECT if the generated answer contradicts the expected answer.

4. INCORRECT if the answer is unrelated.

5. If the expected answer says:
"I don't know based on the document."
then the generated answer must also correctly indicate that
the information is not available in the document.

Reply with ONLY:

CORRECT

or

INCORRECT
"""

    result = judge.invoke(prompt)

    return result.content.strip().upper() == "CORRECT"


# --------------------------------------------------
# Generate answers
# --------------------------------------------------

results = []

print("Running DocQuery evaluation...\n")

for item in questions:

    result = get_answer(item)

    results.append(result)

    if result["actual"] is not None:
        print(f"Q{item['id']} completed")
    else:
        print(f"Q{item['id']} failed")


# --------------------------------------------------
# Judge answers
# --------------------------------------------------

correct = 0
incorrect = 0
failed = []

print("\nJudging answers...\n")

for item in results:

    if item["actual"] is None:

        print(f"Q{item['id']}: API FAILED")

        failed.append(item)

        continue

    is_correct = judge_answer(item)

    if is_correct:

        correct += 1

        print(f"Q{item['id']}: CORRECT")

    else:

        incorrect += 1

        print(f"Q{item['id']}: INCORRECT")

        failed.append(item)


# --------------------------------------------------
# Final results
# --------------------------------------------------

total = len(results)
evaluated = correct + incorrect

if evaluated > 0:
    accuracy = (correct / evaluated) * 100
else:
    accuracy = 0


print("\n==============================")
print("       DocQuery Evaluation")
print("==============================")

print(f"Total Questions : {total}")
print(f"Evaluated       : {evaluated}")
print(f"Correct         : {correct}")
print(f"Incorrect       : {incorrect}")
print(f"API Failures    : {total - evaluated}")
print(f"Accuracy        : {accuracy:.2f}%")

print("==============================")


if failed:

    print("\nFailed Questions:")

    for item in failed:

        print(f"\nQ{item['id']}: {item['question']}")

        print(f"Expected: {item['expected']}")

        print(f"Actual:   {item['actual']}")