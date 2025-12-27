import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tempfile import NamedTemporaryFile
from streamlit_option_menu import option_menu

import os
import streamlit as st
import tensorflow as tf

@st.cache_resource
def load_cnn():
    model_path = "cnn_model/cnn_model.h5"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found")
    return tf.keras.models.load_model(model_path)

try:
    cnn_model = load_cnn()
    cnn_enabled = True
except Exception as e:
    cnn_enabled = False
    st.error(f"CNN disabled: {e}")

import os
import joblib
import streamlit as st

@st.cache_resource
def load_ml_model():
    model_path = os.path.join("models", "classifier.pkl")
    st.write("Checking model path:", model_path)
    st.write("File exists:", os.path.exists(model_path))

    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    return joblib.load(model_path)

try:
    ml_model = load_ml_model()
    ml_enabled = True
except Exception as e:
    ml_enabled = False
    st.error(f"Classical ML model not found — ML prediction disabled: {e}")



import joblib

@st.cache_resource
def load_tabular_model():
    return joblib.load("models/classifier.pkl")

cancer_model = load_tabular_model()




# optional tensorflow
try:
    import tensorflow as tf
except Exception:
    tf = None

MODEL_PATH = "models/keras_model.h5"
ML_MODEL_PATHS = ["models/cancer_model.pkl", "models/classifier.pkl"]

st.set_page_config(page_title="Lung Cancer Detection")
st.title("🫁 Lung Cancer Detection")


@st.cache_resource
def load_keras_model(path=MODEL_PATH):
    if tf is None or not os.path.exists(path):
        return None
    try:
        return tf.keras.models.load_model(path)
    except Exception:
        return None


@st.cache_resource
def load_ml_model(paths=ML_MODEL_PATHS):
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    return pickle.load(f)
            except Exception:
                return None
    return None


keras_model = load_keras_model()
ml_model = load_ml_model()

if keras_model is not None:
    st.success("✅ Keras CNN model loaded")
else:
    st.info("Keras model / TensorFlow not available — CNN features disabled")

if ml_model is not None:
    st.success("✅ Classical ML model loaded")
else:
    st.info("Classical ML model not found — ML prediction disabled")


with st.sidebar:
    selection = option_menu(
        "Lung Cancer Detection System",
        [
            "Introduction",
            "About the Dataset",
            "Lung Cancer Prediction",
            "CNN Based disease Prediction",
        ],
        icons=["activity", "heart", "person", "heart"],
        default_index=0,
    )


if selection == "Introduction":
    gg = Image.open("images/lung-cancer.jpg")
    st.image(gg, caption="Introduction to Lung Cancer", width=600)
    st.title("How common is lung cancer?")
    st.write(
        "Lung cancer (both small cell and non-small cell) is the second most common cancer in both men and women in the United States (not counting skin cancer). In men, prostate cancer is more common, while in women breast cancer is more common.")
    st.markdown(
    """
    The American Cancer Society's estimates for lung cancer in the US for 2023 are:
    - About 238,340 new cases of lung cancer (117,550 in men and 120,790 in women)
    - About 127,070 deaths from lung cancer (67,160 in men and 59,910 in women)

    
    """
    )

    st.write("")
    st.title("Is Smoking the only cause ?")
    mawen = Image.open("images/menwa.png")

    st.image(mawen, caption='Smoking is not the major cause',width=650)
    #page title
    
    st.write("The association between air pollution and lung cancer has been well established for decades. The International Agency for Research on Cancer (IARC), the specialised cancer agency of the World Health Organization, classified outdoor air pollution as carcinogenic to humans in 2013, citing an increased risk of lung cancer from greater exposure to particulate matter and air pollution.")




    st.markdown(
    """
    The following list won't indent no matter what I try:
    - A 2012 study by Mumbai's Tata Memorial Hospital found that 52.1 per cent of lung cancer patients had no history of smoking. 
    - The study contrasted this with a Singapore study that put the number of non-smoking lung cancer patients at 32.5 per cent, and another in the US that found the number to be about 10 per cent.
    - The Tata Memorial study found that 88 per cent of female lung cancer patients were non-smokers, compared with 41.8 per cent of males. It concluded that in the case of non-smokers, environmental and genetic factors were implicated.
    """
    )

    st.title("Not just a Delhi phenomenon ")
    stove = Image.open("images/stove.png")

    st.image(stove, caption='Smoking is not the major cause',width=650)
    #page title
    st.markdown(
    """
    The following list won't indent no matter what I try:
    - In January 2017, researchers at AIIMS, Bhubaneswar, published a demographic profile of lung cancer in eastern India, which found that 48 per cent of patients had not been exposed to active or passive smoking
    - 89 per cent of women patients had never smoked, while the figure for men was 28 per cent.
    - From available research, very little is understood about lung cancer among non-smokers in India. “We need more robust data to identify how strong is the risk and link,” Guleria of AIIMS says.
    """
    )


if selection == "About the Dataset":
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Dataset analysis", "Training Data", "Test Data", "Algorithms Used", "CNN Based Indentification"]
    )

    with tab1:
        st.header("Lung Cancer Dataset")
        data = pd.read_csv("datasets/data.csv")
        st.write(data.head(10))
        code = '''
        Index(['Age', 'Gender', 'Air Pollution', 'Alcohol use', 'Dust Allergy',
       'OccuPational Hazards', 'Genetic Risk', 'chronic Lung Disease',
       'Balanced Diet', 'Obesity', 'Smoking', 'Passive Smoker', 'Chest Pain',
       'Coughing of Blood', 'Fatigue', 'Weight Loss', 'Shortness of Breath',
       'Wheezing', 'Swallowing Difficulty', 'Clubbing of Finger Nails',
       'Frequent Cold', 'Dry Cough', 'Snoring', 'Level'],
      dtype='object')'''
        st.code(code, language='python')

        st.header("Pearson Correlation Matrix")
        coors = Image.open("images/coors.png")
        st.image(coors, caption="Pearson Correlation Matrix", width=800)
        st.write("From the above co-relation matrix we did apply a function which picks out values based on their high correlation with a particular attribute which could be dropped to improve Machine Learning Models Performance")
        st.markdown( """
            
            - The Following Attributed are as follows :-
            """)
        code = '''{'Chest Pain','Coughing of Blood', 'Dust Allergy','Genetic Risk','OccuPational Hazards','chronic Lung Disease'}'''
        st.code(code, language='python')

    with tab2:
        st.header("Lung Cancer Training Dataset")
        data = pd.read_csv("datasets/train.csv", index_col=0)
        st.write(data)
        code = ''' Index(['Age', 'Gender', 'Air Pollution', 'Alcohol use', 'Balanced Diet',
       'Obesity', 'Smoking', 'Passive Smoker', 'Fatigue', 'Weight Loss',
       'Shortness of Breath', 'Wheezing', 'Swallowing Difficulty',
       'Clubbing of Finger Nails', 'Frequent Cold', 'Dry Cough', 'Snoring'],
        dtype='object')'''
        st.code(code, language='python')
        data = pd.read_csv("datasets/trainy.csv", index_col=0)
        st.subheader("Y_Train Data")
        st.dataframe(data, use_container_width=True)

    with tab3:
        st.header("Lung Cancer Test Dataset")
        data = pd.read_csv("datasets/testx.csv", index_col=0)
        st.write(data)
        code = ''' Index(['Age', 'Gender', 'Air Pollution', 'Alcohol use', 'Balanced Diet',
       'Obesity', 'Smoking', 'Passive Smoker', 'Fatigue', 'Weight Loss',
       'Shortness of Breath', 'Wheezing', 'Swallowing Difficulty',
       'Clubbing of Finger Nails', 'Frequent Cold', 'Dry Cough', 'Snoring'],
        dtype='object')'''
        st.code(code, language='python')
        data = pd.read_csv("datasets/testy.csv", index_col=0)
        st.subheader("Y_Test Data")
        st.dataframe(data, use_container_width=True)

    with tab4:
        st.header("List of Algorithms Used")
        algo = Image.open("images/algo.png")
        st.image(algo, caption="ML Algorithms", width=500)
        st.write("Since this is a Mutlti-Class Classification we have used Algorithms which are maily used for Supervised Learning for the following Problem Statement ")

        st.markdown(
            """
            Supervised Learning Algorithms:
            - Linear Regression
            - Support Vector Machine
            - K-Nearest Neighbours (KNN)
            - Decision Tree Classifier
            """
            )
        
        st.write("The accuracy of all the above algorithms is as follows:- ")
        code = '''The accuracy of the SVM is: 95 %
        The accuracy of the SVM is: 100 %
        The accuracy of Decision Tree is: 100 %
        The accuracy of KNN is: 100 %'''
        st.code(code, language='python')

        st.header("Confusion Matrix")

        col1, col2 = st.columns(2)

        with col1:
            algo = Image.open("images/lg.png")

            st.image(algo, caption='LG Confusion Matrix',width=350)

        with col2:
            algo = Image.open("images/svm.png")

            st.image(algo, caption='SVM Confusion Matrix',width=390)

    with tab5:
        st.header("Convolutional Neural Network Model")
        st.write("Approach and model summary")
        url = "https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images"
        st.write("Check out this [Images Dataset](%s)" % url)

        st.subheader("Approach Followed :- ")
        st.markdown(
            """
            - For training our model we have used the Keras API.
            - We have used 2D Convolution Layer along with consecutive MaxPooling Layers to improve the models performance.
            - Because we are facing a two-class classification problem, i.e. a binary classification problem, we will end the network with a sigmoid activation. The output of the network will be a single scalar between 0 and 1, encoding the probability that the current image is class 1 (as opposed to class 0).
            """
            )
        st.subheader("Model Summary")
        summ = Image.open("images/summary.png")
        st.image(summ, caption="Model Summary", width=700)

        st.subheader("Model Compile ")
        st.write(" You will train our model with the binary_crossentropy loss, because it's a binary classification problem and your final activation is a sigmoid. We will use the rmsprop optimizer with a learning rate of 0.001. During training, you will want to monitor classification accuracy.")
        code = '''from tensorflow.keras.optimizers import RMSprop

        model.compile(optimizer=RMSprop(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics = ['accuracy'])'''
        st.code(code, language='python')

        st.subheader("Fitting Data to the Model")
        st.write(" You will train our model with the binary_crossentropy loss, because it's a binary classification problem and your final activation is a sigmoid. We will use the rmsprop optimizer with a learning rate of 0.001. During training, you will want to monitor classification accuracy.")
        code = '''model.fit(
        train_generator,
        epochs=15,
        validation_data=validation_generator,
        verbose=2
            )'''
        st.code(code, language='python')

        epoc = Image.open("images/epoc.png")

        st.image(epoc, caption='Number of Epocs',width=700)

        st.subheader("Plotting the Traning vs Validation (Accuracy and Loss)")
        col1, col2 = st.columns(2)

        with col1:
            acc = Image.open("images/acc.png")

            st.image(acc, caption='Traning vs Validation Accuracy',width=350)

        with col2:
            loss = Image.open("images/loss.png")

            st.image(loss, caption='Traning vs Validation Loss',width=350)

        st.write("As we can see from the above diagram that our Models performs well on the Training as well as Validation Data")







if selection == "Lung Cancer Prediction":
    st.title("Lung Cancer Prediction using ML")

    testx = pd.read_csv("datasets/testx.csv", index_col=0)
    testy = pd.read_csv("datasets/testy.csv", index_col=0)
    testx.reset_index(drop=True, inplace=True)
    testy.reset_index(drop=True, inplace=True)
    concate_data = pd.concat([testx, testy], axis=1)
    st.title('Lung Cancer Prediction using ML')

    idn = st.slider("Select any index from Testing Data", 0, max(0, len(concate_data) - 1), 0)
    a_row = concate_data.iloc[idn]
    st.write("Displaying values of index ", idn)

    # prefill inputs with selected row values where available
    inputs = []
    for i, col in enumerate(testx.columns):
        val = a_row[i] if i < len(a_row) else ""
        inputs.append(st.text_input(col, value=str(val), key=f"input_{i}"))

    if st.button("Predict (ML)"):
        if ml_model is None:
            st.warning("ML model not available. Place a pickle model in models/ and restart.")
        else:
            # attempt to coerce inputs to numeric array; fallback to strings if needed
            try:
                arr = np.array([float(x) for x in inputs]).reshape(1, -1)
            except Exception:
                arr = np.array(inputs).reshape(1, -1)
            try:
                pred = ml_model.predict(arr)
                st.write("Prediction:", pred)
            except Exception as e:
                st.error("Prediction failed")
                st.exception(e)




 # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        heart_prediction = cancer_model.predict([[Age, Gender, AirPollution, Alcoholuse,
        BalancedDiet, Obesity, Smoking, PassiveSmoker, Fatigue, WeightLoss,
        ShortnessofBreath, Wheezing, SwallowingDifficulty,
        ClubbingofFingerNails, FrequentCold, DryCough, Snoring]])
        st.success(f"Prediction: {prediction[0]}")

        
        if (heart_prediction[0] == 'High'):
          heart_diagnosis = 'The person is having heart disease'
          st.error(heart_diagnosis) 

        elif(heart_prediction[0] == 'Medium'):
          heart_diagnosis = 'The person is chance of having heart disease'
          st.warning(heart_diagnosis)
        else:
          heart_diagnosis = 'The person does not have any heart disease'
          st.balloons()
          st.success(heart_diagnosis)
        
        

    expander = st.expander("Here are some more random values from Test Set")
    
    expander.write(concate_data.head(5))
    

if selection == "CNN Based disease Prediction":
    st.title("Lung Cancer Detection using CNN and CT-Scan Images")

    if keras_model is None:
        st.warning("CNN model or TensorFlow not loaded. Place models/keras_model.h5 and ensure TensorFlow is installed.")
    else:
        uploaded = st.file_uploader("Upload CT-Scan Image", type=["png", "jpeg", "jpg"])
        if uploaded is not None:
            tmp = NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
            tmp.write(uploaded.getvalue())
            tmp.flush()

            try:
                img = Image.open(tmp.name).convert("RGB")
                img_resized = img.resize((224, 224))
                arr = np.array(img_resized) / 255.0
                arr = np.expand_dims(arr, axis=0)
                preds = keras_model.predict(arr)
                score = float(preds[0][0]) if preds.size else float(preds[0])
                if score >= 0.5:
                    st.success(f"I am {score:.2%} confident this is a Normal case.")
                else:
                    st.error(f"I am {(1-score):.2%} confident this is a Lung Cancer case.")
                st.image(img, use_column_width=True)
            except Exception as e:
                st.error("Failed to process image")
                st.exception(e)


hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)