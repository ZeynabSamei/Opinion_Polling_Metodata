import pandas as pd
import json
from pathlib import Path
import random

# ==========================================
# Paths (VS Code & Git friendly)
# ==========================================

# Directory where THIS script lives
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "dataset_test"
OUTPUT_DIR = BASE_DIR / "dataset_test"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Input dataset
file_path = DATA_DIR / "anes_timeseries_2024_csv_20250808.csv"

# Output files
json_filename  = OUTPUT_DIR / "anes_2024_test_dataset.json"


# ==========================================
# 1. Load and Clean Data
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

FEATURE_COLS = [c for c in df.columns]

df = df[(df[FEATURE_COLS] >= 0).all(axis=1)]

# df = df[df["vote_choice"].isin([1, 2])].copy()
df = df.copy()
df["vote_choice"] = df["vote_choice"].replace({3: 3,4: 3,5: 3,6: 3})
df = df[df["vote_choice"].isin([1, 2, 3])]

print("Final dataset size:", df.shape)

# ==========================================
# 2. Text Mappings
# ==========================================

RACE_MAP = {
    1: "white", 2: "black", 3: "hispanic",
    4: "asian", 5: "native American", 6: "mixed race"
}

GENDER_MAP = {1: "man", 2: "woman"}

IDEOLOGY_MAP = {
    1: "extremely liberal", 2: "liberal", 3: "slightly liberal",
    4: "moderate", 5: "slightly conservative",
    6: "conservative", 7: "extremely conservative"
}


INTEREST_MAP = {
    1: "very", 2: "somewhat",
    3: "not very", 4: "not at all"
}


CHURCH_MAP = {
    2: "do not attend church",
    1: "attend church"
}

# ==========================================
# 3. Build Dataset
# ==========================================

SYSTEM_PROMPT = (
    "You are an expert political analyst specializing in US elections and voting behavior. "
    "Your task is to analyze the demographic profile provided in the text and predict the vote "
    "choice in the 2024 Election. Output strictly one name: 'Donald Trump', 'Kamala Harris' or 'Other'."
    "Select 'Other' if the vote is unknown or if it is likely for someone other than these two candidates."
    
)

chat_data = []
csv_rows = []

for _, row in df.iterrows():
    try:
        # Raw features
        raw_features = row.to_dict()

        # Text-mapped features
        text_features = {
            "race": RACE_MAP[row["race"]],
            "gender": GENDER_MAP[row["gender"]],
            "age": int(row["age"]),
            "ideology": IDEOLOGY_MAP[row["ideology"]],
            "church_attendance": CHURCH_MAP[row["church_attendance"]],
            "pol_interest": INTEREST_MAP[row["pol_interest"]],
        }

        user_text = (
            f"Racially, I am {text_features['race']}. "
            f"I am a {text_features['gender']}. "
            f"I am {text_features['age']} years old. "
            f"Ideologically, I am {text_features['ideology']}. "
            f"I {text_features['church_attendance']}. "
            f"I am {text_features['pol_interest']} interested in politics. "

        )

        if row["vote_choice"] == 1:
                    assistant_text = "Kamala Harris"
                elif row["vote_choice"] == 2:
                    assistant_text = "Donald Trump"
                else:
                    assistant_text = "Other"
        
                chat_data.append({
                    "features_raw": raw_features,
                    "features_text": text_features,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": assistant_text}
                    ]
                })

        csv_rows.append({
            **raw_features,
            **{f"{k}_text": v for k, v in text_features.items()},
            "user_content": user_text,
            "assistant_content": assistant_text
        })

    except KeyError:
        continue

# ==========================================
# 4. Shuffle (Reproducible)
# ==========================================

SEED = 42
random.seed(SEED)

combined = list(zip(chat_data, csv_rows))
random.shuffle(combined)
chat_data, csv_rows = zip(*combined)

chat_data = list(chat_data)
csv_rows = list(csv_rows)

# ==========================================
# 5. Save Outputs
# ==========================================


# JSON
with open(json_filename, "w") as f:
    json.dump(chat_data, f, indent=2)


print("Saved files:")
print(" -", json_filename)
