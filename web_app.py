import streamlit as st # type: ignore
import tensorflow as tf # type: ignore
import numpy as np # type: ignore


# Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) 
    predictions = model.predict(input_arr)
    return np.argmax(predictions)


# Menubar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Detection"])


# Home Page
if(app_mode=="Home"):
    st.header("PLANT DISEASE DETECTION SYSTEM")
    image_path = "leaf.jpg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Detection System! 🌱🔬
    
    Our goal is to support accurate and timely identification of plant diseases. Upload a plant image, and our system will analyze it to detect possible diseases, helping to safeguard crops and promote a healthier harvest.

    ### How It Works
    1. **Upload Image:** Head over to the **Disease Detection** page and submit an image of the plant you’d like examined.
    2. **Image Analysis:** Our system uses advanced algorithms to examine the image and identify any signs of disease.
    3. **Receive Results:** Get an instant assessment along with suggested next steps.

    ### Why Choose Us?
    - **High Accuracy:** Our system leverages cutting-edge machine learning models to ensure precise disease detection.
    - **Easy to Use:** With a streamlined and intuitive interface, identifying plant diseases is straightforward.
    - **Quick Results:** Obtain insights within seconds, facilitating fast decision-making.

    ### Get Started
    Visit the **Disease Detection** page in the sidebar to upload an image and see how our Plant Disease Detection System can help.
                
    ### About Us
    Discover more about our mission, the team, and project goals on the **About** page.
    """)


# About Page
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About the Dataset
                This dataset has been generated through offline augmentation based on the original collection, which is available on GitHub. It contains approximately 87,000 RGB images of crop leaves, both healthy and affected by various diseases, organized into 38 distinct classes. For effective training, the dataset has been split into training and validation sets in an 80/20 ratio, maintaining the original folder structure.

                Additionally, a separate folder with 33 test images has been created specifically for prediction purposes.
                
                #### Dataset Structure
                1. Train Set: 70,295 images
                2. Validation Set: 17,572 images
                3. Test Set: 33 images

                """)
    

#Prediction Page
elif app_mode == "Disease Detection":
    st.header("Disease Detection")
    test_image = st.file_uploader("Choose an Image:")
    if test_image is not None:
        st.image(test_image, width=400)
    if st.button("Predict Disease"):
        if test_image is not None:
            st.write("Our Prediction")
            result_index = model_prediction(test_image)
            class_name = [
                'an Apple leaf with **Apple scab** disease', 'an Apple leaf with **Black rot** disease', 'an Apple leaf with **Cedar apple rust** disease', 'a **healthy** Apple leaf',
                'a **healthy** Blueberry leaf', 'a Cherry(including sour) leaf with **Powdery mildew** disease', 
                'a **healthy** Cherry(including sour) leaf', 'a Corn(maize) leaf with **Cercospora leaf spot Gray leaf spot** disease', 
                'a Corn(maize) leaf with **Common rust** disease', 'a Corn(maize) leaf with **Northern Leaf Blight** disease', 'a **healthy** Corn(maize) leaf', 
                'a Grape leaf with **Black rot** disease', 'a Grape leaf with **Esca(Black_Measles)** disease', 'a Grape leaf with **Leaf blight (Isariopsis Leaf Spot)** disease', 
                'a **healthy** Grape leaf', 'a Orange leaf with **Haunglongbing(Citrus greening)** disease', 'a Peach leaf with **Bacterial spot** disease',
                'a **healthy** Peach leaf', 'a Pepper, bell leaf with **Bacterial spot** disease', 'a **healthy** Pepper, bell leaf', 
                'a Potato leaf with **Early_blight** disease', 'a Potato leaf with **Late blight** disease', 'a **healthy** Potato leaf', 
                'a **healthy** Raspberry leaf', 'a **healthy** Soyabean leaf', 'a Squash leaf with **Powdery mildew** disease', 
                'a Strawberry leaf with **Leaf scorch** disease', 'a **healthy** Strawberry leaf', 'a Tomato leaf with **Bacterial spot** disease', 
                'a Tomato leaf with **Early blight** disease', 'a Tomato leaf withLate blight** disease', 'a Tomato leaf with **Leaf Mold** disease', 
                'a Tomato leaf with **Septoria leaf spot** disease', 'a Tomato leaf with **Spider mites Two-spotted spider mite** disease', 
                'a Tomato leaf with **Target Spot** disease', 'a Tomato leaf with **Tomato Yellow Leaf Curl Virus**', 'a Tomato leaf with **Tomato mosaic virus**',
                'a **healthy** Tomato leaf'
            ]
            st.success("This is {}".format(class_name[result_index]))
        else:
            st.warning("Please upload an image first!")
