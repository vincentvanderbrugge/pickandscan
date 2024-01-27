# import OpenEXR
# import Imath
import cv2
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
from PIL import Image
import imageio
import numpy as np
import sys
import matplotlib.pyplot as plt


def exr_to_png(exr_file, png_file, depth_png):
    # Read the EXR file

    img = cv2.imread(exr_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    imga = img.astype("float32")
    imageio.imwrite('float_img.exr', arr)
    exr_image = OpenEXR.InputFile(exr_file)
    dw = exr_image.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # Read the depth channel
    depth_channel = exr_image.channel('R', Imath.PixelType(Imath.PixelType.FLOAT))
    depth_array = np.frombuffer(depth_channel, dtype=np.float32)
    depth_array.shape = (size[1], size[0])

    # Normalize the depth values to the range [0, 1]
    normalized_depth = (depth_array - np.min(depth_array)) / (np.max(depth_array) - np.min(depth_array))

    # Convert the normalized depth array to a uint16 array (16-bit PNG)
    depth_uint16 = (normalized_depth * 65535).astype(np.uint16)

    # Create a Pillow image from the uint16 array
    depth_image = Image.fromarray(depth_uint16, mode='I')

    # Save the PNG file
    depth_image.save(png_file)
    print(f"Depth image saved to {png_file}")


if __name__ == "__main__":
    # if len(sys.argv) != 3:
    #     print("Usage: python exr_to_png.py input.exr output.png")
    #     sys.exit(1)

    input_exr = sys.argv[1]
    output_png = sys.argv[2]
    depth_png = sys.argv[3]

    exr_to_png(input_exr, output_png, depth_png)