import pandas as pd
import os
import sys
import subprocess
import re
import json
import csv

def update_config(config, section_name, input_row):
    section = config.get(section_name, {})
    for key, value in section.items():
        for input_key, input_value in input_row.items():
            if input_key in key:
                section[key] = input_value
    config[section_name] = section
    return config

def enable_section(config, target_section):
    for section_name, section in config.items():
        first_key = list(section.keys())[0]
        if section_name == target_section:
            config[section_name][first_key] = "1"
        else:
            config[section_name][first_key] = "0"
    return config

def write_config_to_file(config, file_name):
    with open(file_name, "w") as file:
        json.dump(config, file, indent=4)

def extract_float(output, pattern):
    regex = re.compile(pattern + r'\s*-->>\s*([\d\.]+)')
    match = regex.search(output)
    if match:
        return float(match.group(1))
    return None

def main():
    input_file = sys.argv[1]
    section_name = sys.argv[2]

    os.system("rm all_output.csv")
    with open("config.json", "r") as file:
        config = json.load(file)

    config = enable_section(config, section_name)

    with open(input_file, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            BS = row["BS"]
            GS = row["GS"]
            UNROLL = row["UNROLL"]
            PIXELS_MB = row["PIXELS_MB"]
            
            updated_config = update_config(config, section_name, row)
            write_config_to_file(updated_config, "config.json")
            # print(json.dumps(updated_config, indent=4))
            
            os.system("bash build.sh")
            output = subprocess.check_output("./memcpy_async").decode("utf-8")
            print(output)
            result_float = extract_float(output, section_name)
            print("GFlops = ", result_float)

            if not os.path.exists("all_output.csv"):
                with open("all_output.csv", "w") as outfile:
                    outfile.write("BS,GS,UNROLL,PIXELS_MB,GFlops\n")

            with open("all_output.csv", "a") as outfile:
                outfile.write(f"{BS},{GS},{UNROLL},{PIXELS_MB},{result_float}\n")
            
if __name__ == "__main__":
    main()


