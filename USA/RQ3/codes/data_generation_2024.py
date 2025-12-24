import pandas as pd
import json
from pathlib import Path
import random

# ==========================================
# Paths
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset_test"
OUTPUT_DIR = BASE_DIR / "dataset_test"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

file_path = DATA_DIR / "anes_timeseries_2024_csv_20220210.csv"
json_filename = OUTPUT_DIR / "anes_2024_interview_dataset.json"

# ==========================================
# 1. Load & Clean Data
# ==========================================
df = pd.read_csv(file_path)

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

FEATURE_COLS = list(df.columns)
df = df[(df[FEATURE_COLS] >= 0).all(axis=1)]

# 3-class vote coding
df["vote_choice"] = df["vote_choice"].replace({3: 3, 4: 3, 5: 3, 6: 3})
df = df[df["vote_choice"].isin([1, 2, 3])]

print("Final dataset size:", df.shape)

# ==========================================
# 2. Interview Question Definitions
# ==========================================
QUESTIONS = [
    {
        "col": "gender",
        "question": 'Interviewer: What is your gender?',
        "vals": {1: "man", 2: "woman"}
    },
    {
        "col": "race",
        "question": 'Interviewer: Which racial group do you identify with?',
        "vals": {
            1: "white",
            2: "black",
            3: "hispanic",
            4: "asian",
            5: "native American",
            6: "mixed race"
        }
    },
    {
        "col": "age",
        "question": 'Interviewer: What is your age in years?',
        "vals": None  # numeric
    },
    {
        "col": "ideology",
        "question": 'Interviewer: How would you describe your political ideology?',
        "vals": {
            1: "extremely liberal",
            2: "liberal",
            3: "slightly liberal",
            4: "moderate",
            5: "slightly conservative",
            6: "conservative",
            7: "extremely conservative"
        }
    },
    {
        "col": "church_attendance",
        "question": 'Interviewer: Do you attend religious services?',
        "vals": {1: "yes", 2: "no"}
    },
    {
        "col": "pol_interest",
        "question": 'Interviewer: How interested are you in politics?',
        "vals": {
            1: "very interested",
            2: "somewhat interested",
            3: "not very interested",
            4: "not at all interested"
        }
    }
]

# ==========================================
# 3. System Prompt
# ==========================================
SYSTEM_PROMPT = (
    "You are an expert political analyst. Based on the interview below, "
    "predict the respondent’s vote choice in the 2024 US Presidential Election. "
    "Respond with exactly one of the following:\n"
    "- Donald Trump\n"
    "- Kamala Harris\n"
    "- others"
)

# ==========================================
# 4. Build Interview Dialogs
# ==========================================
chat_data = []

for _, row in df.iterrows():
    try:
        interview_text = ""

        for q in QUESTIONS:
            col = q["col"]
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

        # Ground truth vote
        if row["vote_choice"] == 1:
            assistant_text = "Kamala Harris"
        elif row["vote_choice"] == 2:
            assistant_text = "Donald Trump"
        else:
            assistant_text = "others"

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
# 5. Shuffle (Reproducible)
# ==========================================
SEED = 42
random.seed(SEED)
random.shuffle(chat_data)

# ==========================================
# 6. Save Output
# ==========================================
with open(json_filename, "w") as f:
    json.dump(chat_data, f, indent=2)

print("Saved interview-style dataset:")
print(" -", json_filename)
