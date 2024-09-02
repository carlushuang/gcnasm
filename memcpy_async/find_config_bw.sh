#!/bin/sh

CONFIG_FILE="config.json"
SECTION_NAME="$1"

if [ -z "$SECTION_NAME" ]; then
    echo "Please provide a section name as an argument."
    echo "Available section names are:"
    cat "$CONFIG_FILE" | python -c "import sys, json; config = json.load(sys.stdin); print('\n'.join(config.keys()));"
    exit 1
fi

python gen_all.py
python config_optimizer_gflops.py all_input.csv "$SECTION_NAME"
python findtop10.py
