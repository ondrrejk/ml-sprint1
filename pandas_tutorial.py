import pandas as pd

df = pd.DataFrame(
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["A", "B", "C"], index=["x", "y", "z"]
)

print(df.head())  # first 5 rows (default)
print(df.head(1))  # first row
print(df.tail(2))  # last 2 rows
print(df.columns)  # just the headers (columns)
print(df.index.to_list())  # just the indexes (rows)
print(df.info())  # information about the dataframe
print(df.describe())  # count, mean, std, min, percentages
print(df.nunique())  # how many unique numbers are in each column
print(df["A"].unique())  # list of unique numbers in column "A"
print(df.shape)  # (x, y) shape of the dataframe
