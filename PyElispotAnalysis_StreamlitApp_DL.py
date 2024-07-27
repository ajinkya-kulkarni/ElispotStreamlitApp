import streamlit as st
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd

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
	layout="wide",
	initial_sidebar_state="expanded",
	menu_items={
		'Get help': 'mailto:kulkajinkya@gmail.com',
		'Report a bug': 'mailto:kulkajinkya@gmail.com',
		'About': 'This is an application for demonstrating the PyElispotAnalysis package. Developed, tested, and maintained by Ajinkya Kulkarni: https://github.com/ajinkya-kulkarni.'
	}
)

##########################################################################

# Set the title of the web app
st.title(':blue[Deep learning based spot detection]')

st.caption('Sample image to test this application is available [here](https://github.com/ajinkya-kulkarni/ElispotStreamlitApp/blob/main/image.tif).', unsafe_allow_html = False)

##########################################################################

# Add some text explaining what the user should do next
st.markdown(':blue[Upload the image to be analyzed.]')

# Add a file uploader to allow the user to upload an image file
uploaded_file = st.file_uploader("Upload a file", type = ["tif", "tiff", "png", "jpg", "jpeg"], accept_multiple_files = False, label_visibility = 'collapsed')

######################################################################

# If no file was uploaded, stop processing and exit early
if uploaded_file is None:
	st.stop()

##############################################################

original_image_pil = Image.open(uploaded_file)
original_image_grayscale_pil = original_image_pil.convert('L')

original_image_np = np.array(original_image_pil)
original_image_grayscale_np = np.array(original_image_grayscale_pil)

if original_image_grayscale_np.shape[0] > allowed_image_size or original_image_grayscale_np.shape[1] > allowed_image_size:
	st.error('Uploaded image exceeds the allowed image size. Please reduce the image size to 1000x1000.')
	st.stop()

normalized_original_image_grayscale_np = normalize_image_percentile(original_image_grayscale_np)

inverted_image_array_np = 1 - normalized_original_image_grayscale_np

points, details = spotiflow_model.predict(inverted_image_array_np, exclude_border=False, verbose=False, device="cpu", subpix=True)

##############################################################

x_coords = points[:, 1].astype(int)
y_coords = points[:, 0].astype(int)

assert len(x_coords) == len(y_coords), "The lengths of x_coords and y_coords are not the same."

# Combine x, y coordinates, and intensities directly
spot_points = np.column_stack((x_coords, y_coords, details.intens.flatten()))

# No need to convert back to NumPy arrays
intensities = spot_points[:, 2]

##############################################################

# Min-max normalization
min_value = np.min(intensities)
max_value = np.max(intensities)

normalized_intensities = (intensities - min_value) / (max_value - min_value)

##############################################################

display_image = original_image_np

##############################################################

# Create the figure with the specified size and DPI
fig = plt.figure(figsize=(5, 5), dpi = 200)

# Plot the image and scatter plot
plt.imshow(display_image, cmap="gray")
scatter = plt.scatter(x_coords, y_coords, s=20, linewidth=1, edgecolors='black', facecolors = "yellow", alpha=0.5)
plt.title(f'{len(x_coords)} spots detected')
plt.axis("off")

# Adjust layout
plt.tight_layout()

col1, col2, col3 = st.columns(3)

with col1:
    st.empty()
	
with col2:
    st.pyplot(fig)

with col3:
    st.empty()

##########################################################################

# Save the figure to a bytes buffer
buf = BytesIO()
plt.savefig(buf, format='png', bbox_inches='tight', dpi = 200, pad_inches=0)
buf.seek(0)
# Load the image from the buffer
pil_img = Image.open(buf)

buf = BytesIO()
pil_img.save(buf, format="TIFF")
byte_im = buf.getvalue()

btn = st.download_button(label="Download Result Image", data=byte_im, file_name="Result1.tif")

# Close the buffer
buf.close()

##########################################################################

st.divider()

##########################################################################

# Setup figure and axes for a 1x4 grid
fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=200)

# First subplot: Just the original image
axs[0].imshow(display_image, cmap="gray")
axs[0].axis("off")
axs[0].set_title('Image')

# Third subplot: Original image with scatter points colored by intensity
axs[1].imshow(display_image, cmap="gray")
scatter = axs[1].scatter(x_coords, y_coords, s=20, linewidth=0.5, edgecolors='black', c=normalized_intensities, cmap='coolwarm', alpha=0.5)
axs[1].axis("off")
axs[1].set_title('Spot intensities')

# # Add colorbar to the scatter plot
# divider = make_axes_locatable(axs[1])
# cax = divider.append_axes("right", size="3%", pad=0.07)
# cbar = fig.colorbar(scatter, cax=cax)
# cbar.set_label('Normalized Spot Intensity')

# Fourth subplot: Normalized histogram of spot intensities
n, bins, patches = axs[2].hist(normalized_intensities, bins=10, color='tab:blue', density=True, alpha=0.5, rwidth=0.5)
axs[2].set_title('Normalized Spot Intensities')
axs[2].set_xlabel('Normalized Intensity')
axs[2].set_ylabel('Frequency')
axs[2].grid(which='both', axis='y', alpha=0.1)
axs[2].grid(which='both', axis='x', alpha=0.1)
axs[2].set_ylim(0, np.ceil(max(n) * 10) / 10 + 0.5)
axs[2].set_xlim(-0.01, 1.01)

# Adjust layout
plt.tight_layout()

st.pyplot(fig)

##########################################################################

# Save the figure to a bytes buffer
buf = BytesIO()
plt.savefig(buf, format='png', bbox_inches='tight', dpi = 200, pad_inches=0)
buf.seek(0)
# Load the image from the buffer
pil_img = Image.open(buf)

buf = BytesIO()
pil_img.save(buf, format="TIFF")
byte_im = buf.getvalue()

btn = st.download_button(label="Download Intensity Image", data=byte_im, file_name="Result2.tif")

# Close the buffer
buf.close()

##########################################################################

st.divider()

##########################################################################

def create_intensity_stats(normalized_intensities):
    stats = {
        "Spots": len(normalized_intensities),
        "Intensity Min": np.min(normalized_intensities),
        "Intensity Max": np.max(normalized_intensities),
        "Intensity Mean": np.mean(normalized_intensities),
        "Intensity Median": np.median(normalized_intensities),
        "Intensity Std Dev": np.std(normalized_intensities),
        "Intensity Variance": np.var(normalized_intensities),
        "Intensity Mode": pd.Series(normalized_intensities).mode().values[0] if not pd.Series(normalized_intensities).mode().empty else np.nan
    }
    
    return pd.DataFrame([stats])

##############################################################

dataframe = create_intensity_stats(normalized_intensities)

st.dataframe(dataframe.style.format("{:.2f}"), use_container_width = True)

##############################################################

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

csv = convert_df(dataframe)

##############################################################

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='dataframe.csv',
    mime='text/csv',
)

st.stop()

##########################################################################
