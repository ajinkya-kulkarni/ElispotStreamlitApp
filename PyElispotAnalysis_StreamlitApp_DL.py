import streamlit as st
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from PIL import Image
from spotiflow.model import Spotiflow

# Initialize the Spotiflow model
spotiflow_model = Spotiflow.from_folder("Pre_Trained_Model")

# Maximum allowed image size
allowed_image_size = 1000  # Only images with sizes less than 1000x1000 allowed

# Load and set the page icon
with open("logo.jpg", "rb") as f:
    image_data = f.read()
image_bytes = BytesIO(image_data)

# Configure the Streamlit page
st.set_page_config(
    page_title='PyElispotAnalysis',
    page_icon=image_bytes,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get help': 'mailto:kulkajinkya@gmail.com',
        'Report a bug': 'mailto:kulkajinkya@gmail.com',
        'About': 'Developed, tested, and maintained by Ajinkya Kulkarni: https://github.com/ajinkya-kulkarni'
    }
)

# Set the title and add a caption with a link to a sample image
st.title(':blue[Deep learning based spot detection]')
st.caption('Sample image to test this application is available [here](https://github.com/ajinkya-kulkarni/ElispotStreamlitApp/blob/main/image.tif).', 
           unsafe_allow_html=False)

# Add instructions and file uploader
st.markdown(':blue[Upload the image to be analyzed.]')
uploaded_file = st.file_uploader("Upload a file", type=["tif", "tiff", "png", "jpg", "jpeg"], 
                                 accept_multiple_files=False, label_visibility='collapsed')

# Stop execution if no file is uploaded
if uploaded_file is None:
    st.stop()

# Process the uploaded image
original_image_pil = Image.open(uploaded_file)
original_image_grayscale_pil = original_image_pil.convert('L')
original_image_np = np.array(original_image_pil)
original_image_grayscale_np = np.array(original_image_grayscale_pil)

# Check if the image size is within the allowed limit
if original_image_grayscale_np.shape[0] > allowed_image_size or original_image_grayscale_np.shape[1] > allowed_image_size:
    st.error('Uploaded image exceeds the allowed image size. Please reduce the image size to 1000x1000.')
    st.stop()

def normalize_image_percentile(image, small_diff_threshold=1e-4):
    """
    Normalize the image using percentile-based contrast stretching.
    
    Args:
        image (numpy.ndarray): Input image array.
        small_diff_threshold (float): Threshold for minimum intensity difference.
    
    Returns:
        numpy.ndarray: Normalized image array.
    """
    image = image.astype(np.float32)
    p1, p99 = np.percentile(image, (1, 99))
    
    if p99 - p1 < small_diff_threshold:
        return np.zeros_like(image)
    
    normalized_image = np.clip((image - p1) / (p99 - p1), 0, 1)
    return normalized_image

normalized_original_image_grayscale_np = normalize_image_percentile(original_image_grayscale_np)
inverted_image_array_np = 1 - normalized_original_image_grayscale_np

# Predict spots using the Spotiflow model
points, details = spotiflow_model.predict(inverted_image_array_np, exclude_border=False, 
                                          verbose=False, device="auto", subpix=True)

# Extract coordinates and intensities
x_coords = points[:, 1].astype(int)
y_coords = points[:, 0].astype(int)
spot_points = np.column_stack((x_coords, y_coords, details.intens.flatten()))
intensities = spot_points[:, 2]

def min_max_normalize_intensities(intensities):
    """
    Normalize intensities to the range [0, 1] using min-max normalization.
    
    Args:
        intensities (numpy.ndarray): Array of intensity values.
    
    Returns:
        numpy.ndarray: Normalized intensity values.
    """
    if intensities.size == 0:
        return np.array([])
    
    min_value, max_value = np.min(intensities), np.max(intensities)
    
    if max_value == min_value:
        return np.zeros_like(intensities)
    
    return (intensities - min_value) / (max_value - min_value)

normalized_intensities = min_max_normalize_intensities(intensities)

# Set display image and DPI for plotting
display_image = original_image_np
DPI = 200

# Plot the image with detected spots
fig = plt.figure(figsize=(4, 4), dpi=DPI)
plt.imshow(display_image, cmap="gray")

if points.size != 0:
    plt.scatter(x_coords, y_coords, s=20, linewidth=1, edgecolors='black', facecolors="yellow", alpha=0.5)
    plt.title(f'{len(x_coords)} spots detected')
else:
    plt.title('No spots detected')

plt.axis("off")
plt.tight_layout()

# Display the plot in Streamlit
col1, col2, col3 = st.columns(3)
with col2:
    st.pyplot(fig)

# Save and offer download of the result image
buf = BytesIO()
plt.savefig(buf, format='png', bbox_inches='tight', dpi=DPI, pad_inches=0)
buf.seek(0)
pil_img = Image.open(buf)
buf = BytesIO()
pil_img.save(buf, format="TIFF")
byte_im = buf.getvalue()
st.download_button(label="Download Result Image", data=byte_im, file_name="Result1.tif")
buf.close()

st.divider()

# Create a 1x3 grid of plots
fig, axs = plt.subplots(1, 3, figsize=(12, 4), dpi=DPI)

# Plot original image
axs[0].imshow(display_image, cmap="gray")
axs[0].axis("off")
axs[0].set_title('Image')

if points.size != 0:
    # Plot image with spots colored by intensity
    scatter = axs[1].imshow(display_image, cmap="gray")
    scatter = axs[1].scatter(x_coords, y_coords, s=20, linewidth=0.5, edgecolors='black', c=normalized_intensities, 
                             cmap='coolwarm', alpha=0.5)
    axs[1].axis("off")
    axs[1].set_title('Spot intensities')

    # Add colorbar
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="3%", pad=0.07)
    cbar = fig.colorbar(scatter, cax=cax)
    cbar.set_label('Normalized Spot intensity')

    # Plot histogram of spot intensities
    n, bins, patches = axs[2].hist(normalized_intensities, bins=10, color='tab:blue', density=True, alpha=0.5, rwidth=0.5)
    axs[2].set_title('Normalized Spot intensities')
    axs[2].set_xlabel('Normalized Intensity')
    axs[2].set_ylabel('Frequency')
    axs[2].grid(which='both', axis='y', alpha=0.1)
    axs[2].grid(which='both', axis='x', alpha=0.1)
    axs[2].set_ylim(0, np.ceil(max(n) * 10) / 10 + 0.5)
    axs[2].set_xlim(-0.01, 1.01)
else:
    axs[1].axis("off")
    axs[1].set_title('No spots detected')
    axs[2].axis("off")
    axs[2].set_title('No intensity data')

plt.tight_layout()
st.pyplot(fig)

# Save and offer download of the intensity image
buf = BytesIO()
plt.savefig(buf, format='png', bbox_inches='tight', dpi=DPI, pad_inches=0)
buf.seek(0)
pil_img = Image.open(buf)
buf = BytesIO()
pil_img.save(buf, format="TIFF")
byte_im = buf.getvalue()
st.download_button(label="Download Intensity Image", data=byte_im, file_name="Result2.tif")
buf.close()

st.divider()

def create_intensity_stats(normalized_intensities):
    """
    Create a DataFrame with intensity statistics.
    
    Args:
        normalized_intensities (numpy.ndarray): Array of normalized intensity values.
    
    Returns:
        pandas.DataFrame: DataFrame containing intensity statistics.
    """
    if len(normalized_intensities) == 0:
        stats = {
            "Spots": 0,
            "Intensity Min": 0,
            "Intensity Max": 0,
            "Intensity Mean": 0,
            "Intensity Median": 0,
            "Intensity Std Dev": 0,
            "Intensity Variance": 0,
            "Intensity Mode": 0
        }
    else:
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

# Create and display the intensity statistics DataFrame
dataframe = create_intensity_stats(normalized_intensities)
st.dataframe(dataframe.style.format("{:.2f}"), use_container_width=True)

@st.cache_data
def convert_df(df):
    """
    Convert DataFrame to CSV string.
    
    Args:
        df (pandas.DataFrame): Input DataFrame.
    
    Returns:
        str: CSV string representation of the DataFrame.
    """
    return df.to_csv().encode('utf-8')

csv = convert_df(dataframe)

# Offer download of the DataFrame as CSV
st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='dataframe.csv',
    mime='text/csv',
)

st.stop()
