import pandas as pd

# Read the CSV file
input_file = "Amazon M2 Archive/products_train.csv"
output_file = "Amazon M2 Archive/products_train_uk.csv"

print(f"Reading {input_file}...")
df = pd.read_csv(input_file)

print(f"Total rows: {len(df)}")
print(f"Locale distribution:\n{df['locale'].value_counts()}")

# Filter for UK locale only
uk_df = df[df['locale'] == 'UK']

print(f"\nUK locale rows: {len(uk_df)}")

# Save to new CSV file
uk_df.to_csv(output_file, index=False)

print(f"\nSaved UK locale data to: {output_file}")

