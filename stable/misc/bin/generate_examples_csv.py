'''
Generate a CSV file with a list of the images in a directory structure

From: ChatGPT
'''

import os
import csv
from pathlib import Path

# Define image file extensions (add more if needed)
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

def find_image_files(directory):
    directory = Path(directory)
    return [str(p.relative_to(directory)) for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]

def save_to_csv(file_paths, output_csv):
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['File', 'Class'])  # header
        for path in file_paths:
            writer.writerow([path, 0])

# Example usage
if __name__ == "__main__":
    root_dir = "/home/fagg/datasets/core50/core50_128x128"
    output_csv = "images.csv"
    
    image_files = find_image_files(root_dir)
    save_to_csv(image_files, output_csv)
    print(f"Found {len(image_files)} images. CSV written to {output_csv}")
