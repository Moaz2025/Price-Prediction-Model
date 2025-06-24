import pandas as pd

# Load the CSV file
df = pd.read_csv('datasets/clean/clean_data_egypt.csv')

# Print unique values in the 'region' column, sorted lexicographically
unique_regions = sorted(df['region'].unique())
# print(unique_regions)
print(len(unique_regions))

# Print the count of each unique value in the 'region' column, sorted lexicographically
region_counts = df['region'].value_counts().sort_index()
i = 0
total = 0
for location, count in region_counts.items():
    if count > 30:
        i += 1
        total += count
        print((location, count))
print(i)
print(total)
