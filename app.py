import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Add, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, Dense, Dropout
@st.cache_resource
def load_assets():
    seq_len = 37 
    inputs = Input(shape=(seq_len,))
    e = Embedding(input_dim=20000, output_dim=64)(inputs)
    pos_indices = tf.range(start=0, limit=seq_len, delta=1)
    pos_indices = tf.expand_dims(pos_indices, axis=0)
    p = Embedding(input_dim=seq_len, output_dim=64)(pos_indices)
    
    x = Add()([e, p])
    x = LayerNormalization()(x)
    
    attentions = MultiHeadAttention(num_heads=2, key_dim=64)(x, x)
    norm = LayerNormalization()(attentions)
    drop1 = Dropout(0.5)(norm)
    
    x_flat = GlobalAveragePooling1D()(drop1)
    hidden = Dense(64, activation='relu')(x_flat)
    drop2 = Dropout(0.3)(hidden)
    outputs = Dense(4, activation='softmax')(drop2)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.load_weights('multi_class_transformer.keras')
    
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
