#!/bin/bash
set -e

SOURCE_DIR="/home/jakub-pytka/HACKNATION/Kopalnia/kopalnia"

# 1. Extract 1st minute
echo "Extracting 1st minute..."
ffmpeg -i "$SOURCE_DIR/nagranie1.mkv" -t 60 -c:v libx264 nagranie1_1min.mp4 -y
ffmpeg -i "$SOURCE_DIR/nagranie2.mp4" -t 60 -c:v libx264 nagranie2_1min.mp4 -y
ffmpeg -i "$SOURCE_DIR/nagranie3.mp4" -t 60 -c:v libx264 nagranie3_1min.mp4 -y

# 2. Inference
echo "Running inference..."
python3 inference_dual.py nagranie1_1min.mp4 --output output/wynik_nagranie1.mp4
python3 inference_dual.py nagranie2_1min.mp4 --output output/wynik_nagranie2.mp4
python3 inference_dual.py nagranie3_1min.mp4 --output output/wynik_nagranie3.mp4

# 3. Web Conversion
echo "Converting to web..."
ffmpeg -i output/wynik_nagranie1.mp4 -c:v libx264 -preset fast -crf 23 output/wynik_nagranie1_web.mp4 -y
ffmpeg -i output/wynik_nagranie2.mp4 -c:v libx264 -preset fast -crf 23 output/wynik_nagranie2_web.mp4 -y
ffmpeg -i output/wynik_nagranie3.mp4 -c:v libx264 -preset fast -crf 23 output/wynik_nagranie3_web.mp4 -y

echo "Done!"
