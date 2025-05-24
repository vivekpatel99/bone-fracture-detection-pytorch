#!/bin/bash

# --- Configuration ---
FPS=15         # Frames per second for the GIF
WIDTH=720      # Width of the GIF in pixels (-1 preserves aspect ratio)
OUTPUT_DIR="readme-assets" # Directory to save the output GIF (default: current directory)
# --- End Configuration ---

# --- Input Validation ---
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg is not installed. Please install it first."
    echo "On Debian/Ubuntu: sudo apt update && sudo apt install ffmpeg"
    echo "On macOS (using Homebrew): brew install ffmpeg"
    exit 1
fi

if [ -z "$1" ]; then
    echo "Usage: $0 <input_video_file> [output_gif_name]"
    echo "Example: $0 my_cool_video.mp4"
    echo "Example: $0 my_cool_video.mp4 awesome_demo.gif"
    exit 1
fi

INPUT_VIDEO="$1"

if [ ! -f "$INPUT_VIDEO" ]; then
    echo "Error: Input video file not found: $INPUT_VIDEO"
    exit 1
fi
# --- End Input Validation ---

# --- Determine Output Filename ---
# Get the base name of the video without extension
BASENAME=$(basename "$INPUT_VIDEO" | sed 's/\.[^.]*$//')

# Use provided output name or generate one
if [ -n "$2" ]; then
    OUTPUT_GIF="${OUTPUT_DIR}/$2"
    # Ensure output has .gif extension if provided without one
    if [[ "$OUTPUT_GIF" != *.gif ]]; then
        OUTPUT_GIF="${OUTPUT_GIF}.gif"
    fi
else
    OUTPUT_GIF="${OUTPUT_DIR}/${BASENAME}.gif"
fi

PALETTE_FILE="/tmp/palette_$(date +%s).png" # Temporary palette file
# --- End Determine Output Filename ---

# --- Ensure Output Directory Exists ---
OUTPUT_PATH_DIR=$(dirname "$OUTPUT_GIF")
mkdir -p "$OUTPUT_PATH_DIR"
# --- End Ensure Output Directory Exists ---

# --- Conversion Process ---
echo "Starting GIF conversion for: $INPUT_VIDEO"
echo "Settings: FPS=$FPS, Width=$WIDTH"
echo "Outputting to: $OUTPUT_GIF"

# 1. Generate the color palette
echo "Step 1: Generating color palette..."
ffmpeg -v warning -i "$INPUT_VIDEO" -vf "fps=${FPS},scale=${WIDTH}:-1:flags=lanczos,palettegen" -y "$PALETTE_FILE"

if [ $? -ne 0 ]; then
    echo "Error: Failed to generate palette."
    rm -f "$PALETTE_FILE" # Clean up temp file
    exit 1
fi

# 2. Create the GIF using the palette
echo "Step 2: Creating GIF using the palette..."
ffmpeg -v warning -i "$INPUT_VIDEO" -i "$PALETTE_FILE" -filter_complex "fps=${FPS},scale=${WIDTH}:-1:flags=lanczos[x];[x][1:v]paletteuse" -y "$OUTPUT_GIF"

if [ $? -ne 0 ]; then
    echo "Error: Failed to create GIF."
    rm -f "$PALETTE_FILE" # Clean up temp file
    exit 1
fi

# --- Cleanup ---
rm -f "$PALETTE_FILE"
echo "-------------------------------------"
echo "Success! GIF created: $OUTPUT_GIF"
echo "-------------------------------------"
# --- End Conversion Process ---

exit 0
