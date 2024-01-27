from PIL import Image
import numpy as np
import sys

def read_png_depth(png_file):
    # Read the PNG file
    depth_image = Image.open(png_file)

    # Convert the image to a numpy array
    depth_array = np.array(depth_image)

    return depth_array

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python read_png_depth.py input.png")
        sys.exit(1)

    input_png = sys.argv[1]

    depth_array = read_png_depth(input_png)

    # Now you can use the 'depth_array' in your further processing or analysis
    print(f"Depth image loaded from {input_png}. Shape: {depth_array.shape}")
