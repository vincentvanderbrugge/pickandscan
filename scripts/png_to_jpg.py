import os
import argparse
from PIL import Image


def convert_png_to_jpg(png_directory, jpg_directory):
    # Ensure the output directory exists
    if not os.path.exists(jpg_directory):
        os.makedirs(jpg_directory)

    # List all PNG files in the input directory
    png_files = [f for f in os.listdir(png_directory) if f.lower().endswith('.png')]

    # Convert each PNG file to JPG and save in the output directory
    for png_file in png_files:
        png_path = os.path.join(png_directory, png_file)
        jpg_file = os.path.splitext(png_file)[0] + '.jpg'
        jpg_path = os.path.join(jpg_directory, jpg_file)

        try:
            with Image.open(png_path) as img:
                img.convert('RGB').save(jpg_path, 'JPEG')
            print(f"Converted: {png_path} -> {jpg_path}")
        except Exception as e:
            print(f"Error converting {png_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PNG images to JPG format.")
    parser.add_argument("-i", "--input", help="Input directory containing PNG images", required=True)
    parser.add_argument("-o", "--output", help="Output directory for converted JPG images", required=True)
    args = parser.parse_args()

    convert_png_to_jpg(args.input, args.output)