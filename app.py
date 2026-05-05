import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences



@st.cache_resource
def load_assets():
    custom_dict = {
        "MultiHeadAttention": tf.keras.layers.MultiHeadAttention,
        "LayerNormalization": tf.keras.layers.LayerNormalization
    }
    
    # Update this line with the custom_objects argument
    model = tf.keras.models.load_model(
        'multi_class_transformer.keras', 
        custom_objects=custom_dict
    )
    
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return model, tokenizer, le
model, tokenizer, le = load_assets()

st.title("📰 News Topic Classifier")
st.markdown("Powered by **Transformer Architecture**")

user_input = st.text_area("Enter News Headline:", placeholder="Type here...")

if st.button("Analyze Sequence"):
    if user_input:
        
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=37, padding='post')
        
       
        prediction = model.predict(padded)
        class_idx = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        
        label = le.classes_[class_idx]
        
        st.subheader(f"Prediction: {label}")
        st.progress(int(confidence))
        st.write(f"Confidence: {confidence:.2f}%")
        
        # Display probability breakdown
        st.write("### Probability Distribution")
        probs = dict(zip(le.classes_, prediction[0]))
        st.bar_chart(probs)
    else:
        st.warning("Please enter some text first.")
