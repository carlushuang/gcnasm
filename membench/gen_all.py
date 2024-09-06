import sys
import os
import pandas as pd
import subprocess

MB_size = int(sys.argv[1])
BASE_SIZE = MB_size / 4
PIXELS_MB = BASE_SIZE

BS_values = []
GS_values = []
UNROLL_values = []
INNER_values = []
PIXELS_MB_values = []


compile_command = "hipcc -o query_cu query_cu.cpp"
subprocess.run(compile_command, shell=True, check=True)

execute_command = "./query_cu"
output = subprocess.check_output(execute_command, shell=True)
os.system("rm query_cu")

# Convert the output to integer and store in base_CU
base_CU = int(output.strip())



# for BS in range(64, 1025, 64):
for BS in range(128, 1025, 128):
#for BS in [1024]:
    for GS in range(base_CU, 1601, base_CU):
        for UNROLL in [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]:
            for INNER in [1]:
                i = 1
                PIXELS_MB =  ((BS * GS * 16 / 4 / 1024 / 1024) * UNROLL * INNER) * i
                while PIXELS_MB < BASE_SIZE:
                    PIXELS_MB =  ((BS * GS * 16 / 4 / 1024 / 1024) * UNROLL * INNER) * i
                    i = i + 1

                print("BS = ", BS)
                print("GS = ", GS)
                print("UNROLL = ", UNROLL)
                print("INNER = ", INNER)
                print("PIXELS_MB = ", PIXELS_MB)
                print("--------------------------")

                BS_values.append(BS)
                GS_values.append(GS)
                UNROLL_values.append(UNROLL)
                INNER_values.append(INNER)
                PIXELS_MB_values.append(PIXELS_MB)

data = {
    "BS": BS_values,
    "GS": GS_values,
    "UNROLL": UNROLL_values,
    "INNER": INNER_values,
    "PIXELS_MB": PIXELS_MB_values
}

df = pd.DataFrame(data)

df.to_csv("all_input.csv", index=False)

