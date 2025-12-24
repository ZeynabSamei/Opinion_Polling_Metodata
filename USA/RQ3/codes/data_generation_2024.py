import pandas as pd
import json
from pathlib import Path
import random
from tqdm import tqdm

# ==========================================
# Paths
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset_test"
OUTPUT_DIR = BASE_DIR / "dataset_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

file_path = DATA_DIR / "anes_timeseries_2024_csv_20250808.csv"
output_file = OUTPUT_DIR / "anes_2024_interview_all_features.json"

# ==========================================
# 1. Load Data
# ==========================================
df = pd.read_csv(file_path, low_memory=False)

# ==========================================
# 2. Map Columns & Clean
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
# 3. Interview Questions & Survey Text
# ==========================================
QUESTIONS = [
    {"col": "gender",
     "question": "What is your gender? Are you 'male' or 'female'?",
     "vals": {1: "male", 2: "female"}},
    
    {"col": "race",
     "question": "I am going to read you a list of race categories. What race do you consider yourself to be? 'White', 'Black', 'Asian', or 'Hispanic'?",
     "vals": {1: "White", 2: "Black", 3: "Hispanic", 4: "Asian", 5: "Native American", 6: "Mixed race"}},
    
    {"col": "age",
     "question": "What is your age in years?",
     "vals": None},
    
    {"col": "church_attendance",
     "question": "Lots of things keep people from attending religious services. Do you ever attend religious services? Please respond with 'yes' or 'no'.",
     "vals": {1: "yes", 2: "no"}},
    
    {"col": "pol_interest",
     "question": "How interested would you say you are in politics? Are you 'very interested', 'somewhat interested', 'not very interested', or 'not at all interested'?",
     "vals": {1: "very interested", 2: "somewhat interested", 3: "not very interested", 4: "not at all interested"}},
    
    {"col": "vote_choice",
     "question": "Which presidential candidate did you vote for in the 2024 election? 'Kamala Harris', 'Donald Trump', or 'someone else'? Note: Only displayed if respondent voted.",
     "vals": {1: "Kamala Harris", 2: "Donald Trump", 3: "someone else"}},
    
    {"col": "ideology",
     "question": "When asked about your political ideology, would you say you are 'extremely liberal', 'liberal', 'slightly liberal', 'moderate', 'slightly conservative', 'conservative', or 'extremely conservative'?",
     "vals": {1: "extremely liberal", 2: "liberal", 3: "slightly liberal", 4: "moderate",
              5: "slightly conservative", 6: "conservative", 7: "extremely conservative"}}
]

SYSTEM_PROMPT = "You are an expert political analyst. Answer the interview questions truthfully based on the information provided."

# ==========================================
# 4. Build Interviews
# ==========================================
chat_data = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    try:
        for HR_OMIT in [q["col"] for q in QUESTIONS]:
            interview_text = ""

            # Add all questions except omitted feature
            for q in QUESTIONS:
                col = q["col"]
                if col == HR_OMIT:
                    continue
                if col not in row or pd.isna(row[col]):
                    continue

                interview_text += f"- {q['question']}\n"
                if q["vals"] is None:
                    interview_text += f"- Respondent: {int(row[col])}\n\n"
                else:
                    val = q["vals"].get(int(row[col]))
                    if val is None:
                        continue
                    interview_text += f"- Respondent: {val}\n\n"

            # Add the omitted feature as the last question
            omit_question = next(q for q in QUESTIONS if q["col"] == HR_OMIT)
            interview_text += f"- {omit_question['question']}\n"

            if omit_question["vals"] is None:
                assistant_text = str(row[HR_OMIT])
            else:
                assistant_text = omit_question["vals"].get(row[HR_OMIT], "unknown")

            chat_data.append({
                "features_raw": row.to_dict(),
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": interview_text.strip()},
                    {"role": "assistant", "content": assistant_text}
                ],
                "omitted_feature": HR_OMIT
            })
    except Exception:
        continue

# ==========================================
# 5. Shuffle & Save
# ==========================================
SEED = 42
random.seed(SEED)
random.shuffle(chat_data)

with open(output_file, "w") as f:
    json.dump(chat_data, f, indent=2)

print(f"Saved dataset predicting all features: {output_file}")
