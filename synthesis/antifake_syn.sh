#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <input_directory> <output_directory>"
    exit 1
fi

INPUT_DIR=$1
OUTPUT_DIR=$2


mkdir -p "$OUTPUT_DIR"


find "$INPUT_DIR" -type f -name "*.wav" | while read -r source_file; do
    if [ -f "$source_file" ]; then

        base_name=$(basename "$source_file")


        if [[ "$base_name" =~ ^.+_[0-9]+\.wav$ ]]; then
            echo "[INFO] Processing speaker file: $base_name"

            converted_file="$OUTPUT_DIR/${base_name%.*}_ffmpeg.wav"
            ffmpeg -i "$source_file" -acodec pcm_s16le -ac 1 -ar 16000 -ab 256k "$converted_file"

            processed_file="$OUTPUT_DIR/${base_name%.*}_antifake.wav"
            python3 run.py "$converted_file" "$processed_file"

            echo "[INFO] Processed: $source_file -> $processed_file"
        else
            echo "[SKIP] Skipping derived file: $base_name"
        fi
    else
        echo "[WARNING] Skipping: $source_file (not a valid file)"
    fi
done

echo "[INFO] Batch processing complete!"
