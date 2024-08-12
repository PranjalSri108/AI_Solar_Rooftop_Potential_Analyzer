import streamlit as st
import torch
import cv2
import numpy as np
import albumentations as A
from PIL import Image
import segmentation_models_pytorch as smp
import torch.nn as nn
import time
import pandas as pd
import webbrowser
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go

st.title('AI Solar Roof Potential Analyzer')

IMG_SIZE = 320
PIXEL_AREA = 0.2
PANEL_EFFICIENCY = 0.15
TEMP_COEF = -0.005
ELEC_RATE = 7

USGS_WEBSITE = 'https://earthexplorer.usgs.gov/'

ENCODER = 'timm-efficientnet-b4'
WEIGHTS = 'imagenet'

class SegmentationModel(nn.Module):
    def __init__(self):
        super(SegmentationModel, self).__init__()
        self.backbone = smp.Unet(
            encoder_name=ENCODER,
            encoder_weights=WEIGHTS,
            in_channels=3,
            classes=1,
            activation=None
        )

    def forward(self, images):
        return self.backbone(images)

@st.cache_resource
def load_model():
    model = SegmentationModel()
    model.load_state_dict(torch.load('vgg_best-model.pt', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

def preprocess_image(image):
    aug = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE)])
    augmented = aug(image=image)
    image = augmented['image']
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = torch.Tensor(image) / 255.0
    return image.unsqueeze(0), augmented['image'].shape[1:]

def postprocess_mask(mask, original_size):
    mask = mask.squeeze().cpu().numpy()
    mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
    return mask

def roof_area_calculate(mask):
    roof_area = PIXEL_AREA * np.sum(np.any(mask != [0, 0, 0], axis=-1))
    return roof_area

def apply_heat_map(mask):
    gradient = np.linspace(0, 1, mask.shape[1]) 
    gradient = np.tile(gradient, (mask.shape[0], 1))  

    orange = np.array([255, 125, 0], dtype=np.uint8)
    yellow = np.array([255, 255, 0], dtype=np.uint8)

    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i in range(3):
        colored_mask[:, :, i] = np.uint8((orange[i] * (1 - gradient) + yellow[i] * gradient) * mask)
    return colored_mask

def load_states():
    df = pd.read_csv('Datasets/solar_irradiance.csv')
    return [state.title() for state in df.iloc[:, 0].tolist()]

def load_temperature_data():
    return pd.read_csv('Datasets/temperature.csv')

def load_irradiance_data():
    return pd.read_csv('Datasets/solar_irradiance.csv')

temperature_df = load_temperature_data()
irradiance_df = load_irradiance_data()

def calculate_monthly_solar_energy(state, roof_area):
    state_temp = temperature_df[temperature_df.iloc[:, 0] == state.upper()].iloc[0, 1:].tolist()
    state_irrd = irradiance_df[irradiance_df.iloc[:, 0] == state.upper()].iloc[0, 1:].tolist()
    
    monthly_energy = []
    for temp, irrd in zip(state_temp, state_irrd):
        eff = PANEL_EFFICIENCY * (1 + TEMP_COEF * (temp - 25))
        solar_energy = irrd * eff * roof_area * 30  # kWh
        monthly_energy.append(solar_energy//100)
    
    return monthly_energy

def main():
    states = load_states()

    with st.sidebar:

        st.header("Share your Contact info: ")

        st.sidebar.image("satellite.jpg", use_column_width=True)

        user_name = st.text_input("Enter your name:")

        col1, col2 = st.columns([2,1])

        with col1:
            age = st.slider("Age: ", 1, 100, 50)
        with col2:
            gender = ["M", "F"]
            gender = st.selectbox("Gender", gender)

        user_info = st.text_input("Enter your mail id:")

        if st.button("Submit"):
            i = 0

    st.success('We are here to make Bharat a GREEN country')

    st.markdown("""---""")
   
    col1, col2 = st.columns([3,1])

    with col1:
      st.subheader("Get satellite imagery of your rooftop :satellite::")

    with col2:
      if st.button("USGS Earth Explorer"):
        webbrowser.open_new_tab(USGS_WEBSITE)

    col1, col2 = st.columns(2)

    with col1:
        selected_state = st.selectbox("Select a State", states)

    with col2:
        current_bill = st.number_input('Enter your Current Annual Electric bill (in Rs.)', min_value=0, max_value=10000000, value=0)

    uploaded_file = st.file_uploader("Upload aerial imagery", type=["jpg", "jpeg", "png", "tif"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        width, height = image.size

        col1, col2 = st.columns(2)

        with col2:
            st.subheader("Crop Parameters")
            left = st.slider("Left", 0, width, 0)
            top = st.slider("Top", 0, height, 0)
            right = st.slider("Right", left, width, width)
            bottom = st.slider("Bottom", top, height, height)

        with col1:
            st.subheader("Image Preview")
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(np.array(image))
            rect = plt.Rectangle((left, top), right - left, bottom - top,
                                 fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            ax.axis('off')
            st.pyplot(fig)

        if st.button("Generate Segmentation Mask"):
            cropped_image = image.crop((left, top, right, bottom))
            image_np = np.array(cropped_image)
            preprocessed_image, original_size = preprocess_image(image_np)

            with st.spinner('Generating heatmap...'):
                time.sleep(2)  # Simulate some processing time
                with torch.no_grad():
                    logits = model(preprocessed_image)
                    pred_mask = torch.sigmoid(logits)
                    pred_mask = (pred_mask > 0.5).float()

                resized_mask = postprocess_mask(pred_mask, (right - left, bottom - top))
                heat_map_mask = apply_heat_map(resized_mask)

                roof_area = roof_area_calculate(heat_map_mask)

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Rooftop")
                    st.image(cropped_image, use_column_width=True)
                with col2:
                    st.subheader("Generated Heatmap")
                    st.image(heat_map_mask, use_column_width=True, clamp=True)

                st.info(f"Roof Area: {roof_area:.2f} sq. m")

                monthly_energy = calculate_monthly_solar_energy(selected_state, roof_area)
                
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=months, y=monthly_energy, mode='lines+markers', name='Energy Production'))
                fig.update_layout(
                    title=f'Monthly Solar Energy Production of your house',
                    xaxis_title='Months',
                    yaxis_title='Energy Production (kWh)',
                    height=350
                )

                st.plotly_chart(fig)

                total_annual_energy = sum(monthly_energy)

                if current_bill > 0:
                    energy_savings = total_annual_energy * ELEC_RATE

                    monthly_savings = [energy * ELEC_RATE for energy in monthly_energy]
                    monthly_bill = [(current_bill / 12) - (savings) for savings in monthly_savings]

                    fig = go.Figure()

                    fig.add_trace(go.Scatter(
                       x=months,
                       y=[current_bill / 12] * len(months),
                       mode='lines',
                       name='Current Monthly Bill',
                       line=dict(color='red', width=2, dash='dash') 
                    ))

                    fig.add_trace(go.Scatter(
                        x=months,
                        y=monthly_bill,
                        mode='lines+markers',
                        name='Solar Monthly Bill',
                        marker=dict(color='green', size=10),  
                        line=dict(color='green', width=2) 
                    ))

                    fig.update_layout(
                        title=f'Monthly Bill After Solar Savings of you house',
                        xaxis_title='Months',
                        yaxis_title='Monthly Bill (Rs.)',
                        height=300
                    )

                    st.plotly_chart(fig)

                col1, col2 = st.columns(2)

                with col1:            
                  x = ['Current Bill', 'Solar Bill']
                  y = [current_bill, energy_savings]

                  fig = go.Figure()

                  fig.add_trace(go.Bar(
                    x=x,
                    y=y,
                    name='Data',
                    marker=dict(color='orange') 
                  ))

                  fig.update_layout(
                    title='Annual Electricity Bill',
                    xaxis_title='',
                    yaxis_title='Electric Bill',
                    height=500
                  )

                  st.plotly_chart(fig)

                with col2:
                  st.info(
                          f"""
                          Your State: {selected_state}

                          Your Roof Area: {roof_area:.2f} sq. m

                          Total Solar Panel area: {(roof_area*0.8):.2f} sq. m

                          Total Annual Solar Energy Production: {total_annual_energy:.2f} kWh
                          """)
                  
                  st.success(f"Estimated Annual Energy Savings: Rs. {(current_bill - energy_savings):.2f}")

if __name__ == "__main__":
    main()
