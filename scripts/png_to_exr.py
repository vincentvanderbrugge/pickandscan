import os
import argparse
from PIL import Image
import numpy as np
import imageio
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"


def convert_png_to_exr(png_directory, exr_directory):
    # Ensure the output directory exists
    if not os.path.exists(exr_directory):
        os.makedirs(exr_directory)

    # List all PNG files in the input directory
    png_files = [f for f in os.listdir(png_directory) if f.lower().endswith('.png')]

    # Convert each PNG file to exr and save in the output directory
    for png_file in tqdm(png_files):
        png_path = os.path.join(png_directory, png_file)
        exr_file = os.path.splitext(png_file)[0] + '.exr'
        exr_path = os.path.join(exr_directory, exr_file)

        try:
            img = Image.open(png_path)
            imga = np.array(img)
            imga = imga / 1000
            imga = imga.astype("float32")
            imageio.imwrite(exr_path, imga)
            # with Image.open(png_path) as img:
            #     img.convert('RGB').save(exr_path, 'EXR')
            # print(f"Converted: {png_path} -> {exr_path}")
        except Exception as e:
            print(f"Error converting {png_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PNG depth images to EXR format.")
    parser.add_argument("-i", "--input", help="Input directory containing PNG images", required=True)
    parser.add_argument("-o", "--output", help="Output directory for converted exr images", required=True)
    args = parser.parse_args()

    convert_png_to_exr(args.input, args.output)