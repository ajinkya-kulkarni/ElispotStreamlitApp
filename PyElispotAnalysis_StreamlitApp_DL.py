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

##########################################################################

import streamlit as st
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt

from spotiflow.model import Spotiflow
spotiflow_model = Spotiflow.from_folder("Pre_Trained_Model")

import sys
# Don't generate the __pycache__ folder locally
sys.dont_write_bytecode = True 
# Print exception without the buit-in python warning
sys.tracebacklimit = 0 

##########################################################################

from modules import *

allowed_image_size = 1000 # Only images with sizes less than 1000x1000 allowed

##########################################################################

# Open the logo file in binary mode and read its contents into memory
with open("logo.jpg", "rb") as f:
	image_data = f.read()

# Create a BytesIO object from the image data
image_bytes = BytesIO(image_data)

# Configure the page settings using the "set_page_config" method of Streamlit
st.set_page_config(
	page_title='PyElispotAnalysis',
	page_icon=image_bytes,  # Use the logo image as the page icon
	layout="centered",
	initial_sidebar_state="expanded",
	menu_items={
		'Get help': 'mailto:ajinkya.kulkarni@mpinat.mpg.de',
		'Report a bug': 'mailto:ajinkya.kulkarni@mpinat.mpg.de',
		'About': 'This is an application for demonstrating the PyElispotAnalysis package. Developed, tested, and maintained by Ajinkya Kulkarni: https://github.com/ajinkya-kulkarni at the MPI-NAT, Goettingen.'
	}
)

##########################################################################

# Set the title of the web app
st.title(':blue[Spot detection for Elispot assay images]')

st.caption('Application screenshots and source code available [here](https://github.com/ajinkya-kulkarni/ElispotStreamlitApp). Sample image to test this application is available [here](https://github.com/ajinkya-kulkarni/ElispotStreamlitApp/blob/main/image.tif).', unsafe_allow_html = False)

st.divider()

##########################################################################

# # Create a form using the "form" method of Streamlit
# with st.form(key = 'form1', clear_on_submit = False):

# Add some text explaining what the user should do next
st.markdown(':blue[Upload the image to be analyzed.]')

# Add a file uploader to allow the user to upload an image file
uploaded_file = st.file_uploader("Upload a file", type = ["tif", "tiff", "png", "jpg", "jpeg"], accept_multiple_files = False, label_visibility = 'collapsed')

######################################################################

# If no file was uploaded, stop processing and exit early
if uploaded_file is None:
	st.stop()

st.divider()

##############################################################

original_image = Image.open(uploaded_file).convert('L')
original_image_array = np.array(original_image)

if original_image_array.shape[0] > allowed_image_size or original_image_array.shape[1] > allowed_image_size:
	st.error('Uploaded image exceeds the allowed image size. Please reduce the image size to 1000x1000.')
	st.stop()

##########################################################################

normalized_original_image_array = normalize_image_percentile(original_image_array)

inverted_image_array = 1 - normalized_original_image_array

##############################################################

points, details = spotiflow_model.predict(inverted_image_array, exclude_border=False, verbose=False, device="cpu", subpix=True)

##############################################################

x_coords = points[:, 1].astype(int)
y_coords = points[:, 0].astype(int)

# Pair x and y coordinates
points = list(zip(x_coords, y_coords))

##############################################################

# Define the neighborhood size (e.g., 3 for 3x3 neighborhood)
neighborhood_size = 3

# Calculate mean intensities
mean_intensities = []
for point in points:
	mean_intensity = mean_intensity_around_point(original_image_array, point, neighborhood_size)
	mean_intensities.append(int(mean_intensity))

mean_intensities = np.array(mean_intensities)

min_val = mean_intensities.min()
max_val = mean_intensities.max()

normalized_mean_intensities = (mean_intensities - min_val) / (max_val - min_val)

##############################################################

display_image = original_image_array
intensity_image = normalized_mean_intensities

##############################################################

# Define the colormap
custom_cmap = subset_colormap(plt.cm.coolwarm_r, 0, 1)

##############################################################

# Define the desired output image size in pixels
output_image_size = (674, 674)

# Define the DPI for the figure to match the desired output size
# Since the figure size is in inches, and we want the final image size in pixels,
# we can set the DPI such that when multiplied by the figure size in inches,
# it gives the desired size in pixels.
dpi = 200
fig_size = (output_image_size[0] / dpi, output_image_size[1] / dpi)

# Create the figure with the specified size and DPI
fig = plt.figure(figsize=fig_size, dpi=dpi)

# Plot the image and scatter plot
plt.imshow(display_image, cmap="gray")
scatter = plt.scatter(x_coords, y_coords, c=intensity_image, s=20, cmap=custom_cmap,
linewidth=0.5, edgecolors='black', alpha=0.6)
plt.axis("off")

# Adjust layout
plt.tight_layout()

# Save the figure to a bytes buffer
buf = BytesIO()
plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
buf.seek(0)

# Load the image from the buffer
pil_img = Image.open(buf)

##############################################################

image_comparison(img1=display_image, img2=pil_img, label1="", label2="")

##############################################################

# buf = BytesIO()
# pil_img.save(buf, format="TIFF")
# byte_im = buf.getvalue()

# btn = st.download_button(label="Download Image", data=byte_im, file_name="Result.tif")

# Close the buffer
buf.close()

##############################################################

st.divider()

##########################################################################

# Setup figure and axes for a 1x4 grid
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# First subplot: Just the original image
axs[0].imshow(display_image, cmap="gray")
axs[0].axis("off")
axs[0].set_title('Image', pad = 15)

# Third subplot: Original image with scatter points colored by intensity
axs[1].imshow(display_image, cmap="gray")
scatter = axs[1].scatter(x_coords, y_coords, c=intensity_image, s=20, cmap=custom_cmap, 
linewidth=0.5, edgecolors='black', alpha=0.6)
axs[1].axis("off")
axs[1].set_title(f'Prediction, {len(y_coords)} Spots', pad = 15)

# Fourth subplot: Normalized histogram of spot intensities
n, bins, patches = axs[2].hist(intensity_image, bins='auto', color='tab:blue', density=True, 
alpha=0.7, rwidth=0.7)
axs[2].set_title('Normalized Spot Intensities', pad = 15)
axs[2].set_xlabel('Normalized Intensity')
axs[2].set_ylabel('Frequency')
axs[2].grid(which='both', axis='y', alpha=0.1)
axs[2].grid(which='both', axis='x', alpha=0.1)
axs[2].set_ylim(0, np.ceil(max(n) * 10) / 10 + 0.5)
axs[2].set_xlim(-0.02, 1.02)

# Adjust layout
plt.tight_layout()

st.pyplot(fig)

##########################################################################

st.divider()

st.stop()

##########################################################################
