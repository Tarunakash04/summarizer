def detect_columns(df_list):
    sets = [set(df.columns) for df in df_list if not df.empty]
    if not sets:
        return [], []
    common = set.intersection(*sets)
    all_cols = set.union(*sets)
    uncommon = all_cols - common
    return sorted(common), sorted(uncommon)

def construct_summary_prompt(df, target, features):
    sample_df = df[[*features, target]].dropna().head(10)
    csv = sample_df.to_csv(index=False)
    return f"""
You are a data analyst. Analyze this table of testing log data.

The column '{target}' is the metric of interest. The columns {', '.join(features)} are configurations or environment attributes.

Based on the sample below, find patterns and summarize how features affect the target.

TABLE:
{csv}

Give a concise analytical summary.
"""
