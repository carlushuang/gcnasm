import pandas as pd

data = pd.read_csv("all_output.csv")

top10_gflops = data.nlargest(10, "GFlops")

print(top10_gflops[["BS", "GS", "UNROLL", "PIXELS_MB", "INNER", "GFlops"]])

