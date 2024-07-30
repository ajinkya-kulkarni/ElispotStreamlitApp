import streamlit as st
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from PIL import Image
from spotiflow.model import Spotiflow
from typing import Tuple, List, Optional
import logging

# Constants
ALLOWED_IMAGE_SIZE = 1000
DPI = 200
ALLOWED_FILE_TYPES = ["tif", "tiff", "png", "jpg", "jpeg"]

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def load_model() -> Spotiflow:
    """Load the Spotiflow model."""
    try:
        return Spotiflow.from_folder("Pre_Trained_Model")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        st.error("Failed to load the model. Please try again later.")
        st.stop()

@st.cache_data
def process_image(uploaded_file) -> Tuple[np.ndarray, np.ndarray]:
    """Process the uploaded image file."""
    try:
        original_image_pil = Image.open(uploaded_file)
        original_image_grayscale_pil = original_image_pil.convert('L')
        original_image_np = np.array(original_image_pil)
        original_image_grayscale_np = np.array(original_image_grayscale_pil)
        
        if original_image_grayscale_np.shape[0] > ALLOWED_IMAGE_SIZE or original_image_grayscale_np.shape[1] > ALLOWED_IMAGE_SIZE:
            raise ValueError(f"Image size exceeds {ALLOWED_IMAGE_SIZE}x{ALLOWED_IMAGE_SIZE}")
        
        return original_image_np, original_image_grayscale_np
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        st.error(f"Error processing image: {e}")
        st.stop()

@st.cache_data
def normalize_image_percentile(image: np.ndarray, small_diff_threshold: float = 1e-4) -> np.ndarray:
    """Normalize the image using percentile-based contrast stretching."""
    image = image.astype(np.float32)
    p1, p99 = np.percentile(image, (1, 99))
    
    if p99 - p1 < small_diff_threshold:
        return np.zeros_like(image)
    
    normalized_image = np.clip((image - p1) / (p99 - p1), 0, 1)
    return normalized_image

@st.cache_data
def predict_spots(model: Spotiflow, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Predict spots using the Spotiflow model."""
    try:
        points, details = model.predict(image, exclude_border=False, verbose=False, device="auto", subpix=True)
        x_coords = points[:, 1].astype(int)
        y_coords = points[:, 0].astype(int)
        intensities = details.intens.flatten()
        return np.column_stack((x_coords, y_coords)), intensities
    except Exception as e:
        logger.error(f"Error predicting spots: {e}")
        st.error("Failed to predict spots. Please try again.")
        st.stop()

@st.cache_data
def min_max_normalize_intensities(intensities: np.ndarray) -> np.ndarray:
    """Normalize intensities to the range [0, 1] using min-max normalization."""
    if intensities.size == 0:
        return np.array([])
    
    min_value, max_value = np.min(intensities), np.max(intensities)
    
    if max_value == min_value:
        return np.zeros_like(intensities)
    
    return (intensities - min_value) / (max_value - min_value)

@st.cache_data
def create_intensity_stats(normalized_intensities: np.ndarray) -> pd.DataFrame:
    """Create a DataFrame with intensity statistics."""
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

@st.cache_data
def convert_df(df: pd.DataFrame) -> str:
    """Convert DataFrame to CSV string."""
    return df.to_csv().encode('utf-8')

def plot_image_with_spots(display_image: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray) -> plt.Figure:
    """Plot the image with detected spots."""
    fig = plt.figure(figsize=(4, 4), dpi=DPI)
    plt.imshow(display_image, cmap="gray")

    if x_coords.size != 0:
        plt.scatter(x_coords, y_coords, s=20, linewidth=1, edgecolors='black', facecolors="yellow", alpha=0.5)
        plt.title(f'{len(x_coords)} spots detected')
    else:
        plt.title('No spots detected')

    plt.axis("off")
    plt.tight_layout()
    return fig

def plot_intensity_analysis(display_image: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray, normalized_intensities: np.ndarray) -> plt.Figure:
    """Plot intensity analysis including original image, spots colored by intensity, and histogram."""
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), dpi=DPI, gridspec_kw={'width_ratios': [1, 1.1, 1]})

    axs[0].imshow(display_image, cmap="gray")
    axs[0].axis("off")
    axs[0].set_title('Image')

    if x_coords.size != 0:
        scatter = axs[1].imshow(display_image, cmap="gray")
        scatter = axs[1].scatter(x_coords, y_coords, s=20, linewidth=0.5, edgecolors='black', c=normalized_intensities, 
                                 cmap='coolwarm', alpha=0.5)
        axs[1].axis("off")
        axs[1].set_title('Spot intensities')

        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(scatter, cax=cax)
        cbar.set_label('Normalized Spot intensity')

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
    return fig

def main():
    # Configure the Streamlit page
    st.set_page_config(
        page_title='PyElispotAnalysis',
        page_icon=Image.open("logo.jpg"),
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get help': 'mailto:kulkajinkya@gmail.com',
            'Report a bug': 'mailto:kulkajinkya@gmail.com',
            'About': 'Developed, tested, and maintained by Ajinkya Kulkarni: https://github.com/ajinkya-kulkarni'
        }
    )

    st.title(':blue[Deep learning based spot detection]')
    st.caption('Sample image to test this application is available [here](https://github.com/ajinkya-kulkarni/ElispotStreamlitApp/blob/main/image.tif).', 
               unsafe_allow_html=False)

    model = load_model()

    st.markdown(':blue[Upload the image to be analyzed.]')
    uploaded_file = st.file_uploader("Upload a file", type=ALLOWED_FILE_TYPES, 
                                     accept_multiple_files=False, label_visibility='collapsed')

    if uploaded_file is None:
        st.stop()

    original_image_np, original_image_grayscale_np = process_image(uploaded_file)
    normalized_original_image_grayscale_np = normalize_image_percentile(original_image_grayscale_np)
    inverted_image_array_np = 1 - normalized_original_image_grayscale_np

    spot_points, intensities = predict_spots(model, inverted_image_array_np)
    x_coords, y_coords = spot_points[:, 0], spot_points[:, 1]
    normalized_intensities = min_max_normalize_intensities(intensities)

    fig1 = plot_image_with_spots(original_image_np, x_coords, y_coords)
    col1, col2, col3 = st.columns(3)
    with col2:
        st.pyplot(fig1)

    buf = BytesIO()
    fig1.savefig(buf, format='png', bbox_inches='tight', dpi=DPI, pad_inches=0)
    buf.seek(0)
    pil_img = Image.open(buf)
    buf = BytesIO()
    pil_img.save(buf, format="TIFF")
    byte_im = buf.getvalue()
    st.download_button(label="Download Result Image", data=byte_im, file_name="Result1.tif")
    buf.close()

    st.divider()

    fig2 = plot_intensity_analysis(original_image_np, x_coords, y_coords, normalized_intensities)
    st.pyplot(fig2)

    buf = BytesIO()
    fig2.savefig(buf, format='png', bbox_inches='tight', dpi=DPI, pad_inches=0)
    buf.seek(0)
    pil_img = Image.open(buf)
    buf = BytesIO()
    pil_img.save(buf, format="TIFF")
    byte_im = buf.getvalue()
    st.download_button(label="Download Intensity Image", data=byte_im, file_name="Result2.tif")
    buf.close()

    st.divider()

    dataframe = create_intensity_stats(normalized_intensities)
    st.dataframe(dataframe.style.format("{:.2f}"), use_container_width=True)

    csv = convert_df(dataframe)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='dataframe.csv',
        mime='text/csv',
    )

if __name__ == "__main__":
    main()
