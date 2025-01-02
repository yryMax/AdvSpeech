#!/bin/bash

# 检查参数
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
            echo "[INFO] Found speaker file: $base_name"


            converted_file="$OUTPUT_DIR/${base_name%.*}_ffmpeg.wav"
            processed_file="$OUTPUT_DIR/${base_name%.*}_antifake.wav"


            if [ -f "$processed_file" ]; then
                echo "[SKIP] Already processed: $processed_file"
                continue
            fi


            echo "[INFO] Converting: $source_file -> $converted_file"
            ffmpeg -i "$source_file" -acodec pcm_s16le -ac 1 -ar 16000 -ab 256k "$converted_file"


            echo "[INFO] Processing with antifake: $converted_file -> $processed_file"
            python3 run.py "$converted_file" "$processed_file"

            echo "[INFO] Completed: $source_file -> $processed_file"
        else
            echo "[SKIP] Skipping derived file: $base_name (does not match pattern)"
        fi
    else
        echo "[WARNING] Skipping: $source_file (not a valid file)"
    fi
done

echo "[INFO] Batch processing complete!"
