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

# Output files
json_filename  = OUTPUT_DIR /"test_canada_immigration_2024.json"


# ==========================================
# 1. Load and Clean Data
# ==========================================

# df = pd.read_csv(file_path)
df = pd.read_stata(DATA_DIR/ "Democracy Checkup 2024 v1.0.dta",convert_categoricals=False)
print(df.shape)

def merge_vismin(row):
    values = ''
    for i, label in VISMIN_MAP.items():
        col = f"dc24_vismin_{i}"
        if row[col] != -99:
            values=label
    return values if values else np.nan



VISMIN_MAP = { 1: "White",  2: "Indigenous", 3: "South Asian", 4: "Chinese", 5: "Black", 6: "Filipino",
               7: "Latin American", 8: "Arab", 9: "Southeast Asian",10: "West Asian",11:"Korean",12:"Japanese",
               13:"Other"}
vismin_cols = [f"dc24_vismin_{i}" for i in VISMIN_MAP]


PARTY_LABELS = {1: "Liberal Party",  2: "Conservative Party", 3: "New Democratic Party", 4: "Bloc Québécois",
                5: "Green Party", 6: "People's Party of Canada", 7:"Other", 8:"Other", 9:"Other"}


df["vismin"] = df.apply(merge_vismin, axis=1)


COLS_MAP = {
    "dc24_imm_level": "immigration", 
    "dc24_vote_choice":"party_choice",
    "dc24_genderid": "gender", 
    "vismin": "race", 
    "dc24_age_in_years": "age", 
    "dc24_pol_interest_1": "pol_interest",
    "dc24_religion": "religion", 
    "dc24_disc_pol_freq":"discuss_pol"
}

df = df[list(COLS_MAP.keys())].rename(columns=COLS_MAP)

FEATURE_COLS = [c for c in df.columns]

df = df[df["party_choice"]!=-99]
df = df[df["immigration"]!=-99]
df = df[df["discuss_pol"]!=-99]
df = df[df["religion"]!=-99]


print("Final dataset size:", df.shape)

# ==========================================
# 2. Text Mappings
# ==========================================


RELIGION_MAP = { 1: "None/Atheist",  2: "Agnostic", 3: "Buddhist",4: "Hindu", 5: "Jewish", 6: "Muslim",
                 7: "Sikh", 8: "Anglican", 9: "Baptist", 10: "Catholic", 11: "Orthodox",
                 12: "Jehovah's Witness", 13: "Lutheran", 14: "Mormon",
                 15: "Pentecostal / Fundamentalist/ Evangelical", 16: "Presbyterian",
                 17: "Protestant", 18: "United Church of Canada", 19: "Christian Reformed",
                 20: "Salvation Army", 21: "Mennonite", 22: "Other"}



IMIG_MAP = {
    1: "More immigrants", 2: " Fewer immigrants", 3: "About the same number of immigrants as now"}

GENDER_MAP = {1: "man", 2: "woman", 3:" Non-binary", 4:"Another gender"}

PARTY_MAP = {1: "Liberal Party",  2: "Conservative Party", 3: "New Democratic Party", 4: "Bloc Québécois",
             5: "Green Party", 6: "People's Party of Canada", 7:"Other", 8:"Other", 9:"Other"}


DISCUSS_MAP = {
    1: "daily", 2: " more than once a week",3: "weekly", 4: "more than once a month ",
    5: "monthly", 6: "rarely" }

# df = df[df["discuss_pol"]<5]

# ==========================================
# 3. Build Dataset
# ==========================================

SYSTEM_PROMPT = (
    "You are a political analyst with expertise in Canadian public opinion and social issues. "
    "Using the demographic information provided, predict the respondent's stance on immigration. "
    "Respond with exactly one option from the following: More immigrants, Fewer immigrants, About the same number of immigrants as now."
)

chat_data = []
csv_rows = []

for _, row in df.iterrows():
    try:
        # Raw features
        raw_features = row.to_dict()

        # Text-mapped features
        text_features = {
            "race": row["race"],
            "gender": GENDER_MAP[row["gender"]],
            "age": int(row["age"]),
#             "ideology": PARTY_MAP[row["ideology"]],
            "religion": RELIGION_MAP[row["religion"]],
            "pol_interest": int(row["pol_interest"]),
            "discuss_pol": DISCUSS_MAP[row["discuss_pol"]], 
            "immigration": IMIG_MAP[row["immigration"]],            
            
        }

        user_text = (
            f"Racially, I am {text_features['race']}. "
            f"I am a {text_features['gender']}. "
            f"I am {text_features['age']} years old. "
#             f"Ideologically, I support {text_features['ideology']} party. "
            f"My religion is {text_features['religion']}. "
            f"I am {text_features['pol_interest']} out of 10 interested in politics. "
            f"I {text_features['discuss_pol']} discuss about politics. "
            
            "I think Canada should admit "

        )


        assistant_text = text_features["immigration"]



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

with open(json_filename, "w") as f:
    json.dump(chat_data, f, indent=2)


print("Saved files:")
print(" -", json_filename)
