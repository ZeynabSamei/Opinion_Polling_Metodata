import pandas as pd
import json
from pathlib import Path
import random

# ==========================================
# Paths (VS Code & Git friendly)
# ==========================================

# Directory where THIS script lives
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "result"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Input dataset
file_path = DATA_DIR / "anes_timeseries_2020_csv_20220210.csv"

# Output files
jsonl_filename = OUTPUT_DIR / "anes_2020_finetune.jsonl"
json_filename  = OUTPUT_DIR / "anes_2020_chat_finetune.json"
csv_filename   = OUTPUT_DIR / "anes_2020_chat_finetune.csv"



# ==========================================
# 1. Load and Clean Data
# ==========================================

df = pd.read_csv(file_path)

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

FEATURE_COLS = [c for c in df.columns if c != "vote_choice"]

df = df[(df[FEATURE_COLS] >= 0).all(axis=1)]
df = df[df["vote_choice"].isin([1, 2])].copy()

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

PARTY_MAP = {
    1: "a strong Democrat",
    2: "not a very strong Democrat",
    3: "an independent Democrat",
    4: "an independent",
    5: "an independent Republican",
    6: "not a very strong Republican",
    7: "a strong Republican"
}

INTEREST_MAP = {
    1: "very", 2: "somewhat",
    3: "not very", 4: "not at all"
}

DISCUSS_MAP = {
    1: "I like to discuss politics with my family and friends.",
    2: "I never discuss politics with my family or friends."
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
    "choice in the 2020 Election. Output strictly one name: 'Donald Trump' or 'Joe Biden'."
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
            "party_id": PARTY_MAP[row["party_id"]],
            "church_attendance": CHURCH_MAP[row["church_attendance"]],
            "pol_interest": INTEREST_MAP[row["pol_interest"]],
            "discuss_politics": DISCUSS_MAP[row["discuss_politics"]],
        }

        user_text = (
            f"Racially, I am {text_features['race']}. "
            f"I am a {text_features['gender']}. "
            f"I am {text_features['age']} years old. "
            f"Ideologically, I am {text_features['ideology']}. "
            f"Politically, I am {text_features['party_id']}. "
            f"I {text_features['church_attendance']}. "
            f"I am {text_features['pol_interest']} interested in politics. "
            f"{text_features['discuss_politics']} "
            "In the 2020 presidential election, I voted for "
        )

        assistant_text = (
            "Joe Biden" if row["vote_choice"] == 1 else "Donald Trump"
        )

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

# JSONL
with open(jsonl_filename, "w") as f:
    for entry in chat_data:
        json.dump(entry, f)
        f.write("\n")

# JSON
with open(json_filename, "w") as f:
    json.dump(chat_data, f, indent=2)

# CSV
df_out=pd.DataFrame(csv_rows)
pd.DataFrame(csv_rows).to_csv(csv_filename, index=False)

print("Saved files:")
print(" -", jsonl_filename)
print(" -", json_filename)
print(" -", csv_filename)
