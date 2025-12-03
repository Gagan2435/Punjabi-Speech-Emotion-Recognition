import pandas as pd

df = pd.read_csv("feature_ext_output.csv")

# Strip leading/trailing spaces
df['emotion'] = df['emotion'].str.strip()

# Standardize labels
df['emotion'] = df['emotion'].replace({
    '4. Angry': '4. Anger',
    '5.  Fear': '5. Fear',
})

# Save cleaned CSV
df.to_csv("feature_ext_output_cleaned.csv", index=False)

print("✅ Cleaned emotion labels and saved to feature_ext_output_cleaned.csv")
print(df['emotion'].value_counts())
