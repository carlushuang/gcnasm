import pandas as pd

# data = pd.read_csv("all_output.csv")
data = pd.read_csv("merged_output.csv")

top10_gflops = data.nlargest(10, "GB/s")

print(top10_gflops[["BS", "GS", "UNROLL", "PIXELS_MB", "INNER", "ROW_PER_THREAD", "PADDING", "GB/s"]])

