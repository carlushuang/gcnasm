import pandas as pd

BS_values = []
GS_values = []
UNROLL_values = []
PIXELS_MB_values = []

BASE_SIZE = 437.5
PIXELS_MB = BASE_SIZE

# for BS in range(64, 1025, 64):
for BS in range(256, 1025, 128):
#for BS in [1024]:
    for GS in range(80, 3201, 80):
        for UNROLL in [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]:
        # for UNROLL in [1, 2, 4, 8, 16, 32, 64]:
            i = 1
            PIXELS_MB =  ((BS * GS * 16 / 4 / 1024 / 1024) * UNROLL) * i
            while PIXELS_MB < BASE_SIZE:
                PIXELS_MB =  ((BS * GS * 16 / 4 / 1024 / 1024) * UNROLL) * i
                i = i + 1

            print("BS = ", BS)
            print("GS = ", GS)
            print("UNROLL = ", UNROLL)
            print("PIXELS_MB = ", PIXELS_MB)
            print("--------------------------")
            
            BS_values.append(BS)
            GS_values.append(GS)
            UNROLL_values.append(UNROLL)
            PIXELS_MB_values.append(PIXELS_MB)

data = {
    "BS": BS_values,
    "GS": GS_values,
    "UNROLL": UNROLL_values,
    "PIXELS_MB": PIXELS_MB_values
}

df = pd.DataFrame(data)

df.to_csv("all_input.csv", index=False)
