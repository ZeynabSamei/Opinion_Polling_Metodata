import os
import pandas as pd
import numpy as np

# -----------------------------
# Helper: safe tetrachoric correlation
# -----------------------------
def tetrachoric_corr_safe(vec1, vec2):
    """Compute tetrachoric correlation; return NaN if degenerate."""
    A = np.sum((vec1 == 0) & (vec2 == 0))
    B = np.sum((vec1 == 0) & (vec2 == 1))
    C = np.sum((vec1 == 1) & (vec2 == 0))
    D = np.sum((vec1 == 1) & (vec2 == 1))
    if (A+B)==0 or (C+D)==0 or (A+C)==0 or (B+D)==0:
        return np.nan
    try:
        return np.cos(np.pi / (1 + np.sqrt((A*D)/(B*C))))
    except:
        return np.nan

# -----------------------------
# Main function
# -----------------------------
def create_summary_table(file_path):
    # Load dataset
    df = pd.read_pickle(file_path)

    # Prepare binary votes
    candidates = list(df['probs'][0].keys())
    df['llm_binary'] = df['probs'].apply(lambda p: max(p, key=p.get))
    df['anes_binary'] = df['ground_truth']
    df['llm_num'] = df['llm_binary'].apply(lambda x: 0 if x == candidates[1] else 1)
    df['anes_num'] = df['anes_binary'].apply(lambda x: 0 if x == candidates[1] else 1)

    # Features and age bins
    feature_cols = ['vote_choice','gender','race','age','ideology','party_id',
                    'pol_interest','church_attendance','discuss_politics']
    age_bins = [0, 30, 45, 60, 120]
    age_labels = ['18–30yearsold','31–45yearsold','46–60yearsold','Over60']
    df['age_bin'] = pd.cut(df['features_text'].apply(lambda x: x.get('age') if isinstance(x, dict) else np.nan),
                           bins=age_bins, labels=age_labels)

    # -----------------------------
    # Build summary
    # -----------------------------
    summary_rows = []

    # Whole sample
    n_total = len(df)
    tetra_total = tetrachoric_corr_safe(df['llm_num'].values, df['anes_num'].values)
    prop_total = np.mean(df['llm_binary'] == df['anes_binary'])

    row = {
        'Variable': 'Wholesample',
        'n_samples': n_total,
        'Tetra': tetra_total,
        'Prop.Agree': prop_total
    }

    for c in candidates:
        real_pct = np.mean(df['anes_binary'] == c)
        llm_pct = np.mean([p[c] for p in df['probs']])
        row[f'RealPct_{c}'] = real_pct
        row[f'LLMPct_{c}'] = llm_pct
        row[f'Bias_{c}'] = llm_pct - real_pct

    summary_rows.append(row)

    # Subgroups
    for feature in feature_cols:
        if feature == 'age':
            values = df['age_bin'].dropna().unique()
        else:
            values = df['features_text'].apply(lambda x: x.get(feature) if isinstance(x, dict) else None).dropna().unique()
        
        for val in values:
            sub = df[df['age_bin']==val] if feature=='age' else df[
                df['features_text'].apply(lambda x: x.get(feature) if isinstance(x, dict) else None)==val
            ]
            n = len(sub)
            tetra = tetrachoric_corr_safe(sub['llm_num'].values, sub['anes_num'].values) if n>=2 else np.nan
            prop = np.mean(sub['llm_binary'] == sub['anes_binary'])
            
            row = {
                'Variable': val,
                'n_samples': n,
                'Tetra': tetra,
                'Prop.Agree': prop
            }
            
            for c in candidates:
                real_pct = np.mean(sub['anes_binary'] == c)
                llm_pct = np.mean([p[c] for p in sub['probs']])
                row[f'RealPct_{c}'] = real_pct
                row[f'LLMPct_{c}'] = llm_pct
                row[f'Bias_{c}'] = llm_pct - real_pct
            
            summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)

    # Save CSV in the same folder
    output_path = os.path.join(os.path.dirname(file_path),
                               os.path.basename(file_path).replace('.pkl','_summary.csv'))
    df_summary.to_csv(output_path, index=False)
    print(f"Summary table saved to: {output_path}")
    return df_summary
