import pandas as pd

# Load the original CSV file
df = pd.read_csv('datasets/clean/clean_data_egypt.csv')

# Count occurrences of each region
region_counts = df['region'].value_counts()

# Filter regions that appear more than 30 times
regions_to_keep = region_counts[region_counts > 30].index
filtered_df = df[df['region'].isin(regions_to_keep)]

# Save the filtered DataFrame to a new CSV file
filtered_df.to_csv('datasets/clean/clean_data.csv', index=False) 