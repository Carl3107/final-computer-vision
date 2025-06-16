import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os 

# Set page config
st.set_page_config(
    page_title="Dog Breed Classification",
    page_icon="🐕",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #2c3e50;
        font-size: 3rem !important;
    }
    .stSubheader {
        color: #34495e;
    }
    .css-1v0mbdj {
        padding-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("🐕 Dog Breed Classification")
st.subheader("Compare Performance of Different Models")

# Sidebar
st.sidebar.header("Model Settings")
model_name = st.sidebar.selectbox(
    "Select Model",
    ["MobileNetV2", "ResNet50", "SimpleCNN"]
)

# Load model and labels
@st.cache_resource#để cache mô hình, tránh tải lại mỗi lần 
def load_model(name):#tải mô hình 
    model_path = f"models/model_{name}.keras"
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Preprocess image
def preprocess_image(image, target_size=(224, 224)):#resize ảnh về kích thước 224x224
    img = image.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)#biến ảnh thành mảng numpy
    img_array = np.expand_dims(img_array, axis=0)#mở rộng mảng để phù hợp với đầu vào của mô hình
    return img_array

# Load performance metrics (simulated data)
def get_metrics():#trả về dataframe chứa các chỉ số hiệu suất của mô hình
    return pd.DataFrame({
        'Model': ['MobileNetV2', 'ResNet50', 'SimpleCNN'],
        'Accuracy': [0.92, 0.89, 0.85],# độ chính xác của mô hình
        'Precision': [0.91, 0.88, 0.84],# độ chính xác
        'Recall': [0.90, 0.87, 0.83],# độ nhạy
        'F1-Score': [0.91, 0.88, 0.84]# điểm F1= trung bình hài hòa giữa độ chính xác và độ nhạy
    })

# Define breeds list properly with correct syntax
breeds = [
    "n02085620-Chihuahua",
    "n02085782-Japanese_spaniel",
    "n02085936-Maltese_dog",
    "n02086079-Pekinese",
    "n02086240-Shih-Tzu",
    "n02086646-Blenheim_spaniel",
    "n02086910-papillon",
    "n02087046-toy_terrier",
    "n02087394-Rhodesian_ridgeback",
    "n02088094-Afghan_hound",
    "n02088238-basset",
    "n02088364-beagle",
    "n02088466-bloodhound",
    "n02088632-bluetick",
    "n02089078-black-and-tan_coonhound",
    "n02089867-Walker_hound",
    "n02089973-English_foxhound",
    "n02090379-redbone",
    "n02090622-borzoi",
    "n02090721-Irish_wolfhound",
    "n02091032-Italian_greyhound",
    "n02091134-whippet",
    "n02091244-Ibizan_hound",
    "n02091467-Norwegian_elkhound",
    "n02091635-otterhound",
    "n02091831-Saluki",
    "n02092002-Scottish_deerhound",
    "n02092339-Weimaraner",
    "n02093256-Staffordshire_bullterrier",
    "n02093428-American_Staffordshire_terrier",
    "n02093647-Bedlington_terrier",
    "n02093754-Border_terrier",
    "n02093859-Kerry_blue_terrier",
    "n02093991-Irish_terrier",
    "n02094114-Norfolk_terrier",
    "n02094258-Norwich_terrier",
    "n02094433-Yorkshire_terrier",
    "n02095314-wire-haired_fox_terrier",
    "n02095570-Lakeland_terrier",
    "n02095889-Sealyham_terrier",
    "n02096051-Airedale",
    "n02096177-cairn",
    "n02096294-Australian_terrier",
    "n02096437-Dandie_Dinmont",
    "n02096585-Boston_bull",
    "n02097047-miniature_schnauzer",
    "n02097130-giant_schnauzer",
    "n02097209-standard_schnauzer",
    "n02097298-Scotch_terrier",
    "n02097474-Tibetan_terrier",
    "n02097658-silky_terrier",
    "n02098105-soft-coated_wheaten_terrier",
    "n02098286-West_Highland_white_terrier",
    "n02098413-Lhasa",
    "n02099267-flat-coated_retriever",
    "n02099429-curly-coated_retriever",
    "n02099601-golden_retriever",
    "n02099712-Labrador_retriever",
    "n02099849-Chesapeake_Bay_retriever",
    "n02100236-German_short-haired_pointer",
    "n02100583-vizsla",
    "n02100735-English_setter",
    "n02100877-Irish_setter",
    "n02101006-Gordon_setter",
    "n02101388-Brittany_spaniel",
    "n02101556-clumber",
    "n02102040-English_springer",
    "n02102177-Welsh_springer_spaniel",
    "n02102318-cocker_spaniel",
    "n02102480-Sussex_spaniel",
    "n02102973-Irish_water_spaniel",
    "n02104029-kuvasz",
    "n02104365-schipperke",
    "n02105056-groenendael",
    "n02105162-malinois",
    "n02105251-briard",
    "n02105412-kelpie",
    "n02105505-komondor",
    "n02105641-Old_English_sheepdog",
    "n02105855-Shetland_sheepdog",
    "n02106030-collie",
    "n02106166-Border_collie",
    "n02106382-Bouvier_des_Flandres",
    "n02106550-Rottweiler",
    "n02106662-German_shepherd",
    "n02107142-Doberman",
    "n02107312-miniature_pinscher",
    "n02107574-Greater_Swiss_Mountain_dog",
    "n02107683-Bernese_mountain_dog",
    "n02107908-Appenzeller",
    "n02108000-EntleBucher",
    "n02108089-boxer",
    "n02108422-bull_mastiff",
    "n02108551-Tibetan_mastiff",
    "n02108915-French_bulldog",
    "n02109047-Great_Dane",
    "n02109525-Saint_Bernard",
    "n02109961-Eskimo_dog",
    "n02110063-malamute",
    "n02110185-Siberian_husky",
    "n02110627-affenpinscher",
    "n02110806-basenji",
    "n02110958-pug",
    "n02111129-Leonberg",
    "n02111277-Newfoundland",
    "n02111500-Great_Pyrenees",
    "n02111889-Samoyed",
    "n02112018-Pomeranian",
    "n02112137-chow",
    "n02112350-keeshond",
    "n02112706-Brabancon_griffon",
    "n02113023-Pembroke",
    "n02113186-Cardigan",
    "n02113624-toy_poodle",
    "n02113712-miniature_poodle",
    "n02113799-standard_poodle",
    "n02113978-Mexican_hairless",
    "n02115641-dingo",
    "n02115913-dhole",
    "n02116738-African_hunting_dog"
]

# Clean breed name
def clean_breed_name(breed_name):
    """Clean breed name by removing code and formatting text"""
    clean_name = breed_name.split('-')[-1].replace('_', ' ')
    return ' '.join(word.capitalize() for word in clean_name.split())

# Get prediction results
def get_prediction_results(prediction, breeds):
    """Process prediction results and return top predictions"""
    try:
        # Debug information
        st.write("Number of breeds:", len(breeds))
        st.write("Prediction shape:", prediction.shape)
        
        predicted_class = np.argmax(prediction[0])
        
        # Validate prediction index
        if predicted_class >= len(breeds):
            st.error(f"Invalid prediction index: {predicted_class} for {len(breeds)} breeds")
            return None, None, None
            
        # Get top 5 predictions
        num_classes = min(5, len(breeds))
        top_k_indices = np.argsort(prediction[0])[-num_classes:][::-1]
        
        # Ensure indices are within bounds
        top_k_indices = [i for i in top_k_indices if i < len(breeds)]
        top_k_breeds = [clean_breed_name(breeds[i]) for i in top_k_indices]
        top_k_probs = prediction[0][top_k_indices] * 100
        
        return top_k_breeds, top_k_probs, predicted_class
        
    except Exception as e:
        st.error(f"Error in prediction processing: {str(e)}")
        return None, None, None

# Main content
tabs = st.tabs(["Model Prediction", "Performance Metrics", "Model Comparison"])

with tabs[0]:
    st.header("Dog Breed Classification")
    uploaded_file = st.file_uploader("Choose a dog image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            model = load_model(model_name)
            if model:
                processed_image = preprocess_image(image)
                prediction = model.predict(processed_image)
                
                # Get prediction results
                top_k_breeds, top_k_probs, predicted_class = get_prediction_results(prediction, breeds)
                
                if top_k_breeds and top_k_probs is not None:
                    # Display main prediction
                    st.markdown(f"""
                    ### 🏆 Top Prediction
                    ## {top_k_breeds[0]}
                    Confidence: {top_k_probs[0]:.1f}%
                    """)
                    
                    # Create DataFrame for visualization
                    top_pred_df = pd.DataFrame({
                        'Breed': top_k_breeds,
                        'Confidence': top_k_probs
                    })
                    
                    # Display chart
                    fig = px.bar(top_pred_df,
                                x='Confidence',
                                y='Breed',
                                orientation='h',
                                text='Confidence')
                    fig.update_traces(texttemplate='%{text:.1f}%')
                    st.plotly_chart(fig)
                else:
                    st.warning("Could not process prediction. Please try another image.")

with tabs[1]:
    st.header("Model Performance Metrics")
    metrics_df = get_metrics()
    
    # Metrics table
    st.dataframe(metrics_df.style.highlight_max(axis=0))
    
    # Metrics visualization
    fig = px.line_polar(metrics_df, r=metrics_df.iloc[0][1:], 
                        theta=metrics_df.columns[1:],
                        line_close=True,
                        title=f"Model Performance Metrics - {model_name}")
    st.plotly_chart(fig)

with tabs[2]:
    st.header("Model Comparison")
    
    # Comparison chart
    fig = px.bar(metrics_df.melt(id_vars=['Model'], 
                                var_name='Metric', 
                                value_name='Score'),
                x='Model', y='Score', color='Metric',
                barmode='group',
                title="Model Performance Comparison")
    st.plotly_chart(fig)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Powered by Streamlit & TensorFlow</p>
    </div>
    """, unsafe_allow_html=True)