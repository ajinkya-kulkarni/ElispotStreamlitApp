#!/usr/bin/env python3
# encoding: utf-8
#
# Copyright (C) 2022 Max Planck Institute for Multidisclplinary Sciences
# Copyright (C) 2022 University Medical Center Goettingen
# Copyright (C) 2022 Ajinkya Kulkarni <ajinkya.kulkarni@mpinat.mpg.de>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

######################################################################################

# This file contains all the modules/functions necessary for running the streamlit application or the example notebooks.

######################################################################################

from PIL import Image
from skimage.segmentation import relabel_sequential
from skimage.measure import label, regionprops
from matplotlib.colors import LinearSegmentedColormap

######################################################################################

def subset_colormap(cmap, min_val=0.0, max_val=1.0, n=100):
    """
    Generate a subset of a colormap.

    Args:
    cmap (Colormap): The original colormap.
    min_val (float): The start point of the subset, between 0 and 1.
    max_val (float): The end point of the subset, between 0 and 1.
    n (int): The number of colors in the subset.

    Returns:
    LinearSegmentedColormap: A new colormap that is a subset of the original.
    """
    colors = cmap(np.linspace(min_val, max_val, n))
    return LinearSegmentedColormap.from_list('subset', colors, n)

def normalize_image_percentile(image, small_diff_threshold=1e-4):
    # Convert image to float32 for precise division
    image = image.astype(np.float32)

    # Calculate the 1st and 99th percentiles
    p1 = np.percentile(image, 1)
    p99 = np.percentile(image, 99)

    # Check if the difference is too small or zero
    if p99 - p1 < small_diff_threshold:
        normalized_image = np.full(image.shape, 0)
    else:
        # Normalize the image
        normalized_image = (image - p1) / (p99 - p1)
        # Clip values to [0, 1]
        normalized_image = np.clip(normalized_image, 0, 1)

    return normalized_image

def relabel_instance_segmentation(instance_segmented_image):
    """
    Relabels the instance segmented image sequentially.

    Parameters:
        instance_segmented_image : numpy.ndarray
            The instance segmented image to be relabeled.

    Returns:
        relabeled_image : numpy.ndarray
            The relabeled image.
    """
    relabeled_image, _, _ = relabel_sequential(instance_segmented_image)
    return relabeled_image

def threshold_mask(mask, threshold_value):
    """Thresholds a mask based on a given value."""
    thresholded_mask = mask.copy()
    thresholded_mask[thresholded_mask < threshold_value] = 0
    thresholded_mask[thresholded_mask >= threshold_value] = 1
    return thresholded_mask

def create_centroid_mask_from_regions(mask_array, threshold_value = 0.5):
    """Creates a mask where the centroids of regions in the input mask are marked."""
    labeled_mask = label(mask_array.astype(np.uint16))
    centroid_mask = np.zeros_like(mask_array)
    for region in regionprops(labeled_mask):
        centroid = np.round(region.centroid).astype(np.uint16)
        centroid_mask[centroid[0], centroid[1]] = region.label

    thresholded_centroid_mask = threshold_mask(centroid_mask, threshold_value)
    return thresholded_centroid_mask.astype(np.uint16)

def mean_intensity_around_point(image_array, point, neighborhood_size):
    
    x, y = point
    half_size = neighborhood_size // 2
    
    # Ensure the neighborhood doesn't go out of image bounds
    x_min = max(x - half_size, 0)
    x_max = min(x + half_size + 1, image_array.shape[1])
    y_min = max(y - half_size, 0)
    y_max = min(y + half_size + 1, image_array.shape[0])
    
    neighborhood = image_array[y_min:y_max, x_min:x_max]
    
    return np.mean(neighborhood)

# Function to convert mask arrays to DataFrames
def get_points_df(mask_array):
    points = np.argwhere(mask_array)
    return pd.DataFrame({'x_coord': points[:, 1], 'y_coord': points[:, 0]})

######################################################################################

import streamlit.components.v1 as components
import base64
import io
from typing import Union, Tuple
import requests
from PIL import Image
import numpy as np

def read_image_and_convert_to_base64(image: Union[Image.Image, str, np.ndarray]) -> Tuple[str, int, int]:
	"""
	Reads an image in PIL Image, file path, or numpy array format and returns a base64-encoded string of the image
	in JPEG format, along with its width and height.

	Args:
		image: An image in PIL Image, file path, or numpy array format.

	Returns:
		A tuple containing:
		- base64_src (str): A base64-encoded string of the image in JPEG format.
		- width (int): The width of the image in pixels.
		- height (int): The height of the image in pixels.

	Raises:
		TypeError: If the input image is not of a recognized type.

	Assumes:
		This function assumes that the input image is a valid image in PIL Image, file path, or numpy array format.
		It also assumes that the necessary libraries such as Pillow and scikit-image are installed.

	"""
	# Set the maximum image size to None to allow reading of large images
	Image.MAX_IMAGE_PIXELS = None

	# If input image is PIL Image, convert it to RGB format
	if isinstance(image, Image.Image):
		image_pil = image.convert('RGB')

	# If input image is a file path, open it using requests library if it's a URL, otherwise use PIL Image's open function
	elif isinstance(image, str):
		try:
			image_pil = Image.open(
				requests.get(image, stream=True).raw if str(image).startswith("http") else image
			).convert("RGB")
		except:
			# If opening image using requests library fails, try to use scikit-image library to read the image
			try:
				import skimage.io
			except ImportError:
				raise ImportError("Please run 'pip install -U scikit-image imagecodecs' for large image handling.")

			# Read the image using scikit-image and convert it to a PIL Image
			image_sk = skimage.io.imread(image).astype(np.uint8)
			if len(image_sk.shape) == 2:
				image_pil = Image.fromarray(image_sk, mode="1").convert("RGB")
			elif image_sk.shape[2] == 4:
				image_pil = Image.fromarray(image_sk, mode="RGBA").convert("RGB")
			elif image_sk.shape[2] == 3:
				image_pil = Image.fromarray(image_sk, mode="RGB")
			else:
				raise TypeError(f"image with shape: {image_sk.shape[3]} is not supported.")

	# If input image is a numpy array, create a PIL Image from it
	elif isinstance(image, np.ndarray):
		if image.shape[0] < 5:
			image = image[:, :, ::-1]
		image_pil = Image.fromarray(image).convert("RGB")

	# If input image is not of a recognized type, raise a TypeError
	else:
		raise TypeError("read image with 'pillow' using 'Image.open()'")

	# Get the width and height of the image
	width, height = image_pil.size

	# Save the PIL Image as a JPEG image with maximum quality (100) and no subsampling
	in_mem_file = io.BytesIO()
	image_pil.save(in_mem_file, format="JPEG", subsampling=0, quality=100)

	# Encode the bytes of the JPEG image in base64 format
	img_bytes = in_mem_file.getvalue()
	image_str = base64.b64encode(img_bytes).decode("utf-8")

	# Create a base64-encoded string of the image in JPEG format
	base64_src = f"data:image/jpg;base64,{image_str}"

	# Return the base64-encoded string along with the width and height of the image
	return base64_src, width, height

######################################################

def image_comparison(
	img1: str,
	img2: str,
	label1: str,
	label2: str,
	width_value = 674,
	show_labels: bool=True,
	starting_position: int=50,
) -> components.html:
	"""
	Creates an HTML block containing an image comparison slider of two images.

	Args:
		img1 (str): A string representing the path or URL of the first image to be compared.
		img2 (str): A string representing the path or URL of the second image to be compared.
		label1 (str): A label to be displayed above the first image in the slider.
		label2 (str): A label to be displayed above the second image in the slider.
		width_value (int, optional): The maximum width of the slider in pixels. Defaults to 500.
		show_labels (bool, optional): Whether to show the labels above the images in the slider. Defaults to True.
		starting_position (int, optional): The starting position of the slider. Defaults to 50.

	Returns:
		A Dash HTML component that displays an image comparison slider.

	"""
		# Convert the input images to base64 format
	img1_base64, img1_width, img1_height = read_image_and_convert_to_base64(img1)
	img2_base64, img2_width, img2_height = read_image_and_convert_to_base64(img2)

	# Get the maximum width and height of the input images
	img_width = int(max(img1_width, img2_width))
	img_height = int(max(img1_height, img2_height))

	# Calculate the aspect ratio of the images
	h_to_w = img_height / img_width

	# Determine the height of the slider based on the width and aspect ratio
	if img_width < width_value:
		width = img_width
	else:
		width = width_value
	height = int(width * h_to_w)

	# Load CSS and JS for the slider
	cdn_path = "https://cdn.knightlab.com/libs/juxtapose/latest"
	css_block = f'<link rel="stylesheet" href="{cdn_path}/css/juxtapose.css">'
	js_block = f'<script src="{cdn_path}/js/juxtapose.min.js"></script>'

	# Create the HTML code for the slider
	htmlcode = f"""
		<style>body {{ margin: unset; }}</style>
		{css_block}
		{js_block}
		<div id="foo" style="height: {height}; width: {width};"></div>
		<script>
		slider = new juxtapose.JXSlider('#foo',
			[
				{{
					src: '{img1_base64}',
					label: '{label1}',
				}},
				{{
					src: '{img2_base64}',
					label: '{label2}',
				}}
			],
			{{
				animate: true,
				showLabels: {str(show_labels).lower()},
				showCredits: true,
				startingPosition: "{starting_position}%",
				makeResponsive: true,
			}});
		</script>
		"""

	# Create a Dash HTML component from the HTML code
	static_component = components.html(htmlcode, height=height, width=width)

	return static_component

##########################################################################
