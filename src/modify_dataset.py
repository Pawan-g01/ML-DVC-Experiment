import pandas as pd

file_path = "../data/iris.csv"

df = pd.read_csv(file_path)

df["dataset_version"] = "v2"

df.to_csv(file_path, index=False)

print("Dataset modified successfully!")
print(df.head())