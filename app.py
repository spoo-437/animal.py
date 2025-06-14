import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

# Set page config
st.set_page_config(
    page_title="Animal Classifier",
    page_icon="🐾",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the MobileNetV2 model (cached for performance)"""
    model = MobileNetV2(weights='imagenet', include_top=True)
    return model

@st.cache_data
def get_animal_classes():
    """Get the set of animal classes (cached for performance)"""
    return {
        'tabby', 'tiger', 'Persian_cat', 'Siamese_cat', 'Egyptian_cat',
        'lion', 'leopard', 'cheetah', 'jaguar', 'lynx', 'snow_leopard',
        'German_shepherd', 'collie', 'border_collie', 'beagle', 'bloodhound',
        'basset', 'Afghan_hound', 'golden_retriever', 'Labrador_retriever',
        'cocker_spaniel', 'Irish_setter', 'English_setter', 'Gordon_setter',
        'Brittany_spaniel', 'English_springer', 'Welsh_springer_spaniel',
        'kuvasz', 'schipperke', 'groenendael', 'malinois', 'briard',
        'Australian_terrier', 'Dandie_Dinmont', 'Boston_bull', 'miniature_schnauzer',
        'giant_schnauzer', 'standard_schnauzer', 'Scotch_terrier', 'Tibetan_terrier',
        'wire-haired_fox_terrier', 'Lakeland_terrier', 'Sealyham_terrier',
        'Airedale', 'cairn', 'Yorkshire_terrier', 'silky_terrier',
        'soft-coated_wheaten_terrier', 'West_Highland_white_terrier', 'Lhasa',
        'flat-coated_retriever', 'curly-coated_retriever', 'Nova_Scotia_duck_tolling_retriever',
        'borzoi', 'Irish_wolfhound', 'Italian_greyhound', 'whippet', 'Ibizan_hound',
        'Norwegian_elkhound', 'otterhound', 'Saluki', 'Scottish_deerhound', 'Weimaraner',
        'Staffordshire_bullterrier', 'American_Staffordshire_terrier', 'Bedlington_terrier',
        'Border_terrier', 'Kerry_blue_terrier', 'Irish_terrier', 'Norfolk_terrier',
        'Norwich_terrier', 'Pomeranian', 'Chihuahua', 'Japanese_spaniel', 'Maltese_dog',
        'Pekinese', 'Shih-Tzu', 'Blenheim_spaniel', 'papillon', 'toy_terrier',
        'Rhodesian_ridgeback', 'bluetick', 'black-and-tan_coonhound', 'Walker_hound',
        'English_foxhound', 'redbone', 'elephant', 'giraffe', 'zebra', 'rhinoceros',
        'hippopotamus', 'gazelle', 'impala', 'hartebeest', 'wildebeest', 'water_buffalo',
        'bison', 'ox', 'ram', 'bighorn', 'ibex', 'chamois', 'brown_bear',
        'American_black_bear', 'ice_bear', 'sloth_bear', 'mongoose', 'meerkat',
        'tiger_beetle', 'ladybug', 'ground_beetle', 'long-horned_beetle', 'leaf_beetle',
        'dung_beetle', 'rhinoceros_beetle', 'weevil', 'fly', 'bee', 'ant',
        'grasshopper', 'cricket', 'walking_stick', 'cockroach', 'mantis', 'cicada',
        'leafhopper', 'lacewing', 'dragonfly', 'damselfly', 'admiral', 'ringlet',
        'monarch', 'cabbage_butterfly', 'sulphur_butterfly', 'lycaenid', 'starfish',
        'sea_urchin', 'sea_cucumber', 'wood_rabbit', 'hare', 'Angora', 'hamster',
        'porcupine', 'fox_squirrel', 'marmot', 'beaver', 'guinea_pig', 'sorrel',
        'hog', 'wild_boar', 'warthog', 'Arabian_camel', 'llama', 'weasel', 'mink',
        'polecat', 'black-footed_ferret', 'otter', 'skunk', 'badger', 'armadillo',
        'three-toed_sloth', 'orangutan', 'gorilla', 'chimpanzee', 'gibbon', 'siamang',
        'guenon', 'patas', 'baboon', 'macaque', 'langur', 'colobus', 'proboscis_monkey',
        'marmoset', 'capuchin', 'howler_monkey', 'titi', 'spider_monkey', 'squirrel_monkey',
        'Madagascar_cat', 'indri', 'Indian_elephant', 'African_elephant', 'lesser_panda',
        'giant_panda', 'barracouta', 'eel', 'coho', 'rock_beauty', 'anemone_fish',
        'sturgeon', 'gar', 'lionfish', 'puffer', 'American_alligator', 'amphibian',
        'Arctic_fox', 'anteater', 'antelope'
    }

def preprocess_image(uploaded_file):
    """Preprocess the uploaded image for prediction"""
    try:
        # Load image from uploaded file
        img = Image.open(uploaded_file)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to 224x224 (MobileNetV2 input size)
        img_resized = img.resize((224, 224))
        
        # Convert to array
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        return img_array, img
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None, None

def classify_animal(model, processed_img, animal_classes):
    """Classify the animal in the image"""
    try:
        # Make prediction
        with st.spinner('Analyzing image...'):
            predictions = model.predict(processed_img, verbose=0)
        
        # Decode predictions
        decoded_predictions = decode_predictions(predictions, top=10)[0]
        
        # Find animals in predictions
        animal_predictions = []
        other_predictions = []
        
        for _, class_name, confidence in decoded_predictions:
            clean_name = class_name.replace('_', ' ').title()
            
            if class_name in animal_classes:
                animal_predictions.append((clean_name, confidence))
            else:
                other_predictions.append((clean_name, confidence))
        
        return animal_predictions, other_predictions, decoded_predictions
        
    except Exception as e:
        st.error(f"Error during classification: {str(e)}")
        return [], [], []

def main():
    # Title and description
    st.title("🐾 Animal Classifier")
    st.markdown("Upload an image and let AI identify the animal!")
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This app uses **MobileNetV2** trained on ImageNet to classify animals in images.
        
        **How to use:**
        1. Upload an image (JPG, JPEG, PNG)
        2. Wait for the AI to analyze
        3. View the results!
        
        **Supported formats:** JPG, JPEG, PNG
        """)
        
        st.header("🎯 Features")
        st.markdown("""
        - Real-time animal classification
        - Confidence scores
        - Top 10 predictions
        - Support for 100+ animal species
        """)
    
    # Load model
    try:
        model = load_model()
        animal_classes = get_animal_classes()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image containing an animal"
    )
    
    if uploaded_file is not None:
        # Create two columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 Uploaded Image")
            
            # Display the uploaded image
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", use_container_width=True)
            
            # Show image info
            st.info(f"**Image size:** {img.size[0]} x {img.size[1]} pixels")
        
        with col2:
            st.subheader("🔍 Classification Results")
            
            # Preprocess image
            processed_img, original_img = preprocess_image(uploaded_file)
            
            if processed_img is not None:
                # Classify the image
                animal_predictions, other_predictions, all_predictions = classify_animal(
                    model, processed_img, animal_classes
                )
                
                if animal_predictions:
                    st.success("🎉 Animals detected!")
                    
                    # Display top animal prediction
                    top_animal, top_confidence = animal_predictions[0]
                    st.markdown(f"### 🏆 **{top_animal}**")
                    
                    # Create progress bar with proper value handling
                    confidence_value = float(top_confidence)
                    st.progress(confidence_value, text=f"Confidence: {confidence_value:.2%}")
                    
                    # Show all animal predictions
                    if len(animal_predictions) > 1:
                        st.markdown("#### Other possible animals:")
                        for animal, conf in animal_predictions[1:]:
                            st.markdown(f"- **{animal}**: {conf:.2%}")
                
                else:
                    st.warning("🤔 No animals detected in this image")
                    if all_predictions:
                        top_pred = all_predictions[0]
                        clean_name = top_pred[1].replace('_', ' ').title()
                        st.markdown(f"**Top prediction:** {clean_name} ({top_pred[2]:.2%})")
                
                # Expandable section for all predictions
                with st.expander("📊 View all top 10 predictions"):
                    st.markdown("#### All Predictions:")
                    for i, (_, class_name, confidence) in enumerate(all_predictions, 1):
                        clean_name = class_name.replace('_', ' ').title()
                        is_animal = "🐾" if class_name in animal_classes else "📦"
                        st.markdown(f"{i}. {is_animal} **{clean_name}**: {confidence:.2%}")
    
    else:
        # Show example when no file is uploaded
        st.info("👆 Please upload an image to get started!")
        
        # You could add some example images here
        st.markdown("### 📝 Tips for best results:")
        st.markdown("""
        - Use clear, well-lit images
        - Ensure the animal is the main subject
        - Avoid heavily cropped or blurry images
        - JPG, JPEG, and PNG formats are supported
        """)

if __name__ == "__main__":
    main()