import glob
import pandas as pd

file_list = glob.glob("all_output*.csv")

all_data = []

for file in file_list:
    data = pd.read_csv(file)
    print(data)
    all_data.append(data)

merged_data = pd.concat(all_data, ignore_index=True)

merged_data.to_csv("merged_output.csv", index=False)

