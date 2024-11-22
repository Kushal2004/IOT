
import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from model import PotholeModel  # Assuming you have defined the model in a separate file
import matplotlib.pyplot as plt

# Constants
IMG_SIZE = 224  # Assuming input image size, modify as necessary

# Function to calculate perceived focal length
def calculate_perceived_focal_length(bbox):
    length = bbox[3] - bbox[1]
    width = bbox[2] - bbox[0]
    pixel_length = length  # Assuming length represents pixel length
    camera_distance = 90  # Fixed camera distance in centimeters
    return (pixel_length * camera_distance) / width

# Function to estimate dimensions
def estimate_dimensions(image, gt_bbox, model):
    with torch.no_grad():
        # Preprocess the image
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image = F.to_tensor(image).unsqueeze(0)

        # Perform inference
        pred_bboxes = model(image)
        
        # Calculate perceived focal length for each detected pothole
        perceived_focal_lengths = []
        for pred_bbox in pred_bboxes:
            perceived_focal_length = calculate_perceived_focal_length(pred_bbox)
            perceived_focal_lengths.append(perceived_focal_length)

        # Calculate average perceived focal length
        average_perceived_focal_length = torch.mean(torch.tensor(perceived_focal_lengths)).item()

        return average_perceived_focal_length, pred_bboxes

# Function to calculate dimensions from bounding box
def calculate_dimensions(bbox, perceived_focal_length):
    # Extract bounding box coordinates
    xmin, ymin, xmax, ymax = bbox.squeeze().tolist()

    # Calculate length and width of the bounding box
    length = ymax - ymin
    width = xmax - xmin

    # Calculate area of the pothole
    area = length * width

    # Estimate actual dimensions using the perceived focal length
    actual_length = (perceived_focal_length * length) / IMG_SIZE  # Assuming image size is used for scaling
    actual_width = (perceived_focal_length * width) / IMG_SIZE

    return actual_length, actual_width, area

# Function to estimate raw materials
def estimate_raw_materials(area, depth_cm=5, material_density_kg_per_m3=2400):
    """
    Estimates raw materials needed to fill a pothole.
    
    Parameters:
    - area: Area of the pothole in cm²
    - depth_cm: Average depth of the pothole in cm (default = 5 cm)
    - material_density_kg_per_m3: Density of the material in kg/m³ (default = 2400 for asphalt)
    
    Returns:
    - Volume in liters
    - Weight of raw materials in kilograms
    """
    # Convert area from cm² to m²
    area_m2 = area / 10000  # 1 m² = 10,000 cm²

    # Convert depth from cm to meters
    depth_m = depth_cm / 100  # 1 m = 100 cm

    # Calculate volume in m³
    volume_m3 = area_m2 * depth_m

    # Convert volume to liters (1 m³ = 1000 liters)
    volume_liters = volume_m3 * 1000

    # Calculate weight in kilograms
    weight_kg = volume_m3 * material_density_kg_per_m3

    return volume_liters, weight_kg

# Function to integrate raw material estimation with dimension estimation
def compare_plots_with_dimension_and_materials(image, gt_bbox, out_bbox, perceived_focal_length):
    # Perform dimension estimation
    actual_length, actual_width, area = calculate_dimensions(out_bbox, perceived_focal_length)

    # Estimate raw materials
    volume_liters, weight_kg = estimate_raw_materials(area)

    # Convert image to a PyTorch tensor
    image_tensor = F.to_tensor(image).unsqueeze(0)  # Add batch dimension

    # Plot the image with bounding boxes, dimension estimation, and raw material info
    fig, ax = plt.subplots()
    ax.imshow(image_tensor.squeeze().permute(1, 2, 0).cpu().numpy())

    # Plot ground truth bounding box
    gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_bbox
    gt_length = gt_ymax - gt_ymin
    gt_width = gt_xmax - gt_xmin
    gt_area = gt_length * gt_width
    ax.add_patch(plt.Rectangle((gt_xmin, gt_ymin), gt_width, gt_length, edgecolor='g', facecolor='none', linewidth=2))
    ax.text(gt_xmin, gt_ymin - 10, f'GT Area: {gt_area:.2f}', color='g', fontsize=10)

    # Plot detected bounding box
    out_xmin, out_ymin, out_xmax, out_ymax = out_bbox.squeeze().tolist()
    out_length = out_ymax - out_ymin
    out_width = out_xmax - out_xmin
    ax.add_patch(plt.Rectangle((out_xmin, out_ymin), out_width, out_length, edgecolor='r', facecolor='none', linewidth=2))
    ax.text(out_xmin, out_ymax + 20, f'Estimated Area: {area:.2f}', color='r', fontsize=10)
    ax.text(out_xmin, out_ymax + 40, f'Estimated Length: {actual_length:.2f} cm', color='r', fontsize=10)
    ax.text(out_xmin, out_ymax + 60, f'Estimated Width: {actual_width:.2f} cm', color='r', fontsize=10)
    ax.text(out_xmin, out_ymax + 80, f'Volume: {volume_liters:.2f} L', color='b', fontsize=10)
    ax.text(out_xmin, out_ymax + 100, f'Materials: {weight_kg:.2f} kg', color='b', fontsize=10)

    # Display the plot in Streamlit
    # st.pyplot(fig)

    # Display results in Streamlit
    st.write(f"Estimated Dimensions:")
    st.write(f"Length: {actual_length:.2f} cm")
    st.write(f"Width: {actual_width:.2f} cm")
    st.write(f"Area: {area:.2f} cm²")
    st.write(f"Raw Materials Needed:")
    st.write(f"Volume: {volume_liters:.2f} L")
    st.write(f"Weight: {weight_kg:.2f} kg")


# Function to compare bounding boxes
def compare_plots(image, gt_bbox, out_bbox):
    xmin, ymin, xmax, ymax = gt_bbox

    pt1 = (int(xmin), int(ymin))
    pt2 = (int(xmax), int(ymax))

    out_xmin, out_ymin, out_xmax, out_ymax = out_bbox[0]

    out_pt1 = (int(out_xmin), int(out_ymin))
    out_pt2 = (int(out_xmax), int(out_ymax))

    out_img = cv2.rectangle(image.squeeze().permute(1, 2, 0).cpu().numpy(), pt1, pt2, (0, 255, 0), 2)
    out_img = cv2.rectangle(out_img, out_pt1, out_pt2, (255, 0, 0), 2)
    plt.imshow(out_img)
    plt.show()

# Load the model
model = PotholeModel()
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

# Streamlit app
st.title("Pothole Dimension and Raw Material Estimation")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Estimate Dimensions and Raw Materials'):
        # Ground truth bounding box coordinates (xmin, ymin, xmax, ymax)
        gt_bbox = [0, 0, 100, 100]  # Replace with actual ground truth bbox

        # Estimate dimensions and raw materials
        estimated_length, pred_bboxes = estimate_dimensions(image, gt_bbox, model)

        # Assuming we want to show the first bounding box
        out_bbox = pred_bboxes[0]  # Choose first bounding box for demo

        # Use perceived focal length to estimate raw materials and compare
        perceived_focal_length = calculate_perceived_focal_length(out_bbox)
        compare_plots_with_dimension_and_materials(image, gt_bbox, out_bbox, perceived_focal_length)

        st.write(f"Estimated Average Perceived Focal Length: {estimated_length} cm")
