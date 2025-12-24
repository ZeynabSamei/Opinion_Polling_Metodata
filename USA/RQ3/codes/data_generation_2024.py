import pandas as pd
import json
from pathlib import Path
import random
import sys
from tqdm import tqdm

# ==========================================
# Paths
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset_test"
OUTPUT_DIR = BASE_DIR / "dataset_test"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
file_path = DATA_DIR / "anes_timeseries_2024_csv_20220210.csv"

# ==========================================
# 1. Load Data
# ==========================================
df = pd.read_csv(file_path)

# ==========================================
# 2. Map Columns & Text Values
# ==========================================
COLS_MAP = {
    "V242096x": "vote_choice",
    "V241550": "gender",
    "V241501x": "race",
    "V241458x": "age",
    "V241177": "ideology",
    "V242400": "pol_interest",
    "V241439": "church_attendance",
}

df = df[list(COLS_MAP.keys())].rename(columns=COLS_MAP)
df = df[(df[list(COLS_MAP.values())] >= 0).all(axis=1)]

# 3-class vote coding
df["vote_choice"] = df["vote_choice"].replace({3: 3, 4: 3, 5: 3, 6: 3})
df = df[df["vote_choice"].isin([1, 2, 3])]

# ==========================================
# 3. Interview Questions
# ==========================================
QUESTIONS = [
    {"col": "gender", "question": "Interviewer: What is your gender?", "vals": {1: "man", 2: "woman"}},
    {"col": "race", "question": "Interviewer: Which racial group do you identify with?", "vals": {
        1: "white", 2: "black", 3: "hispanic", 4: "asian", 5: "native American", 6: "mixed race"}},
    {"col": "age", "question": "Interviewer: What is your age in years?", "vals": None},
    {"col": "ideology", "question": "Interviewer: How would you describe your political ideology?", "vals": {
        1: "extremely liberal", 2: "liberal", 3: "slightly liberal", 4: "moderate",
        5: "slightly conservative", 6: "conservative", 7: "extremely conservative"}},
    {"col": "church_attendance", "question": "Interviewer: Do you attend religious services?", "vals": {1: "yes", 2: "no"}},
    {"col": "pol_interest", "question": "Interviewer: How interested are you in politics?", "vals": {
        1: "very interested", 2: "somewhat interested", 3: "not very interested", 4: "not at all interested"}},
    {"col": "vote_choice", "question": "Interviewer: Who will you vote for in 2024?", "vals": {1: "Kamala Harris", 2: "Donald Trump", 3: "others"}}
]

# ==========================================
# 4. System Prompt
# ==========================================
SYSTEM_PROMPT = "You are an expert political analyst. Answer the interview questions truthfully based on the information provided."

# ==========================================
# 5. Omitted Variable (LLM should predict)
# ==========================================
if len(sys.argv) < 2:
    print("Usage: python generate_interview.py <variable_to_predict>")
    sys.exit(1)

HR_OMIT = sys.argv[1]  # e.g., "vote_choice", "church_attendance"
omit_question = next(q for q in QUESTIONS if q["col"] == HR_OMIT)

# ==========================================
# 6. Build Interview Dialogs
# ==========================================
chat_data = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    try:
        interview_text = ""
        for q in QUESTIONS:
            col = q["col"]
            if col == HR_OMIT:
                continue  # omit this variable
            if col not in row or pd.isna(row[col]):
                continue

            interview_text += q["question"] + "\n"
            if q["vals"] is None:
                interview_text += f"Me: {int(row[col])}\n\n"
            else:
                val = q["vals"].get(int(row[col]))
                if val is None:
                    continue
                interview_text += f"Me: {val}\n\n"

        # The LLM answer placeholder
        assistant_text = omit_question["vals"].get(row[HR_OMIT], "unknown") if omit_question["vals"] else str(row[HR_OMIT])

        chat_data.append({
            "features_raw": row.to_dict(),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": interview_text.strip()},
                {"role": "assistant", "content": assistant_text}
            ]
        })
    except Exception:
        continue

# ==========================================
# 7. Shuffle & Save
# ==========================================
SEED = 42
random.seed(SEED)
random.shuffle(chat_data)

output_file = OUTPUT_DIR / f"anes_2024_interview_{HR_OMIT}.json"
with open(output_file, "w") as f:
    json.dump(chat_data, f, indent=2)

print(f"Saved dataset predicting '{HR_OMIT}': {output_file}")
