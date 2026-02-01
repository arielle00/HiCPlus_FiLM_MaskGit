"""
School count analysis function.

Processes a DataFrame containing school information and counts subjects by state.
"""

import pandas as pd


def schoolCount(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process school data to count subjects by state.
    
    Steps:
    1. Drop schools offering < 3 subjects
    2. Clean state_code (keep only alphanumeric)
    3. Create subject indicator columns (english, maths, physics, chemistry)
    4. Group by state_code and sum subject counts
    
    Args:
        df: DataFrame with columns 'subjects' and 'state_code'
            'subjects' should be a space-separated string of subject names
    
    Returns:
        DataFrame with columns: state_code, english, maths, physics, chemistry
        Each subject column contains the count of schools offering that subject
        in each state.
    """
    # ----------------------------------
    # 1. Drop schools offering < 3 subjects
    # ----------------------------------
    df = df[df["subjects"].str.split().str.len() >= 3].copy()

    # ----------------------------------
    # 2. Clean state_code (keep only alphanumeric)
    # ----------------------------------
    df["state_code"] = df["state_code"].str.replace(r"[^a-zA-Z0-9]", "", regex=True)

    # ----------------------------------
    # 3. Create subject indicator columns
    # ----------------------------------
    subjects = ["english", "maths", "physics", "chemistry"]
    for sub in subjects:
        df[sub] = df["subjects"].str.contains(fr"\b{sub}\b", case=False, regex=True).astype(int)

    # ----------------------------------
    # 4. Group by state_code (preserve order)
    # ----------------------------------
    result = (
        df.groupby("state_code", sort=False)[subjects]
          .sum()
          .reset_index()
    )

    return result


# Example usage
if __name__ == "__main__":
    # Example data
    sample_data = pd.DataFrame({
        "subjects": [
            "english maths physics",
            "english chemistry",
            "maths physics chemistry",
            "english maths physics chemistry",
            "english maths",
        ],
        "state_code": ["CA-01", "NY-02", "TX-03", "CA-01", "FL-04"]
    })
    
    print("Input data:")
    print(sample_data)
    print("\n" + "="*50 + "\n")
    
    result = schoolCount(sample_data)
    print("Output (school counts by state):")
    print(result)
