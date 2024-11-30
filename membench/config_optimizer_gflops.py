import pandas as pd
import os
import sys
import subprocess
import re
import json
import csv
import threading
import queue
from itertools import cycle
from tqdm import tqdm


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

def process_gpu_tasks(gpu_id, task_queue, task_progress):
    while not task_queue.empty():
        try:
            row = task_queue.get(block=False)
        except queue.Empty:
            break

        BS = row["BS"]
        GS = row["GS"]
        UNROLL = row["UNROLL"]
        INNER = row["INNER"]
        ROW_PER_THREAD = row["ROW_PER_THREAD"]
        PADDING = row["PADDING"]
        PIXELS_MB = row["PIXELS_MB"]

        with open("config.json", "r") as file:
            config = json.load(file)

        section_name = sys.argv[2]
        arch = sys.argv[3]
        config = enable_section(config, section_name)
        updated_config = update_config(config, section_name, row)
        write_config_to_file(updated_config, f"config{gpu_id}.json")
        cmd = f"ARCH={arch} EXEC_NAME=membench{gpu_id} CONFIG_FILE=config{gpu_id}.json bash build.sh; HIP_VISIBLE_DEVICES={gpu_id} ./membench{gpu_id}"
        print("Execute: ", cmd)

        output_file = f"all_output{gpu_id}.csv"
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        output, _ = process.communicate()
        output = output.decode("utf-8")
        result_float = extract_float(output, section_name)
        # print(output)
        if not os.path.exists(output_file):
            with open(output_file, "w") as outfile:
                outfile.write("BS,GS,UNROLL,PIXELS_MB,INNER,ROW_PER_THREAD,PADDING,GB/s\n")

        with open(output_file, "a") as outfile:
            outfile.write(f"{BS},{GS},{UNROLL},{PIXELS_MB},{INNER},{ROW_PER_THREAD},{PADDING},{result_float}\n")

        #print("---------------------------------------------------------------------------------------", file=sys.stderr)
        task_progress.update(1)
        #print("\n")


def main():
    input_file = sys.argv[1]
    gpu_num = int(sys.argv[4])

    os.system("rm all_output*.csv")

    task_queue = queue.Queue()
    with open(input_file, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        # task_progress = tqdm(desc="Tasks", total=sum(1 for _ in reader), ncols=80)
        task_progress = tqdm(desc="\033[32mTasks\033[0m", total=sum(1 for _ in reader), ncols=80)
        csvfile.seek(0)
        reader = csv.DictReader(csvfile)

        for row in reader:
            task_queue.put(row)

        threads = []
        for i in range(gpu_num):
            thread = threading.Thread(target=process_gpu_tasks, args=(i, task_queue, task_progress))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

    task_progress.close()
###########################################

            
if __name__ == "__main__":
    main()


