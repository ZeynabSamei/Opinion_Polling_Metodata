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

file_path = DATA_DIR / "anes_timeseries_2020_csv_20220210.csv"
output_file = OUTPUT_DIR / "anes_2020_interview_all_features.json"

# ==========================================
# 1. Load Data
# ==========================================
df = pd.read_csv(file_path, low_memory=False)

COLS_MAP = {
    "V202110x": "vote_choice",
    "V201600": "gender",
    "V201549x": "race",
    "V201507x": "age",
    "V201200": "ideology",
    "V201231x": "party_id",
    "V202406": "pol_interest",
    "V201452": "church_attendance",
    "V202022": "discuss_politics"
}

df = df[list(COLS_MAP.keys())].rename(columns=COLS_MAP)
df = df[(df[list(COLS_MAP.values())] >= 0).all(axis=1)]

# 3-class vote coding
df["vote_choice"] = df["vote_choice"].replace({3: 3, 4: 3, 5: 3})
df = df[df["vote_choice"].isin([1, 2, 3])]

# ==========================================
# 2. Text Mappings
# ==========================================
RACE_MAP = {1: "white", 2: "black", 3: "hispanic", 4: "asian", 5: "native American", 6: "mixed race"}
GENDER_MAP = {1: "man", 2: "woman"}
IDEOLOGY_MAP = {1: "extremely liberal", 2: "liberal", 3: "slightly liberal", 4: "moderate",
                5: "slightly conservative", 6: "conservative", 7: "extremely conservative"}
PARTY_MAP = {1: "a strong Democrat", 2: "not a very strong Democrat", 3: "an independent Democrat",
             4: "an independent", 5: "an independent Republican", 6: "not a very strong Republican",
             7: "a strong Republican"}
INTEREST_MAP = {1: "very", 2: "somewhat", 3: "not very", 4: "not at all"}
DISCUSS_MAP = {1: "I like to discuss politics with my family and friends.",
               2: "I never discuss politics with my family or friends."}
CHURCH_MAP = {1: "attend church", 2: "do not attend church"}

# ==========================================
# 3. System Prompt
# ==========================================
SYSTEM_PROMPT = "You are an expert political analyst. Answer the interview questions truthfully based on the information provided."

# ==========================================
# 4. Questions Setup
# ==========================================
QUESTIONS = [
    {"col": "gender", "question": "What is your gender? Are you 'man' or 'woman'?", "vals": GENDER_MAP},
    {"col": "race", "question": "I am going to read you a list of race categories. What race do you consider yourself to be? 'White', 'Black', 'Asian', 'Hispanic', 'Native American', or 'Mixed race'?", "vals": RACE_MAP},
    {"col": "age", "question": "What is your age in years?", "vals": None},
    {"col": "ideology", "question": "When asked about your political ideology, would you say you are 'extremely liberal', 'liberal', 'slightly liberal', 'moderate', 'slightly conservative', 'conservative', or 'extremely conservative'?", "vals": IDEOLOGY_MAP},
    {"col": "party_id", "question": "How would you describe your political affiliation?", "vals": PARTY_MAP},
    {"col": "church_attendance", "question": "Do you attend religious services? Please respond with 'attend church' or 'do not attend church'.", "vals": CHURCH_MAP},
    {"col": "pol_interest", "question": "How interested would you say you are in politics? Are you 'very', 'somewhat', 'not very', or 'not at all' interested?", "vals": INTEREST_MAP},
    {"col": "discuss_politics", "question": "Do you discuss politics with your family and friends?", "vals": DISCUSS_MAP},
    {"col": "vote_choice", "question": "Which presidential candidate did you vote for in 2020? 'Joe Biden', 'Donald Trump', or 'others'?", "vals": {1: "Joe Biden", 2: "Donald Trump", 3: "others"}}
]

# ==========================================
# 5. Build Interview-style Dataset for all features
# ==========================================
chat_data = []

for omit_feature in [q["col"] for q in QUESTIONS]:
    omit_question = next(q for q in QUESTIONS if q["col"] == omit_feature)

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Omitting {omit_feature}"):
        try:
            interview_text = ""
            for q in QUESTIONS:
                col = q["col"]
                if col == omit_feature:
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

            # Omitted feature is always last
            interview_text += f"- {omit_question['question']}"

            # Assistant answer placeholder (ground truth)
            assistant_text = omit_question["vals"].get(row[omit_feature], "unknown") if omit_question["vals"] else str(row[omit_feature])

            chat_data.append({
                "omit_variable": omit_feature,
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
# 6. Shuffle & Save
# ==========================================
SEED = 42
random.seed(SEED)
random.shuffle(chat_data)

with open(output_file, "w") as f:
    json.dump(chat_data, f, indent=2)

print(f"Saved ANES 2020 dataset with all features predicted in turn: {output_file}")
