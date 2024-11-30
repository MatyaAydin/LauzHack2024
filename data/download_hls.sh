#!/bin/bash

# Updated URL of the HLS stream
HLS_URL="https://6162417352ffd.streamlock.net/hls/platjeoost.stream/playlist.m3u8?wowzatokenendtime=1732980763&wowzatokenhash=Z0fix7mDSvBHY0SqHmnHAYwtbZZD_mm0MgQtnP0cCn8=&wowzatokenstarttime=1732978903"

# Directory to save videos
OUTPUT_DIR="./videos"

# Duration for each video segment (in seconds)
SEGMENT_DURATION=300  # 5 minutes

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Function to download a single segment
download_segment() {
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    OUTPUT_FILE="$OUTPUT_DIR/segment_$TIMESTAMP.mp4"

    echo "Downloading 5-minute segment to $OUTPUT_FILE..."
    ffmpeg -y -i "$HLS_URL" -t "$SEGMENT_DURATION" -c copy "$OUTPUT_FILE"

    if [ $? -eq 0 ]; then
        echo "Segment saved: $OUTPUT_FILE"
    else
        echo "Error downloading segment."
    fi
}

# Infinite loop to download segments every 5 minutes
while true; do
    download_segment
    echo "Waiting for 5 minutes..."
    sleep "$SEGMENT_DURATION"
done
