import streamlit as st
import tensorflow as tf
import streamlit as st

class_names = ['Dead Mild Steel', 'Decarburised High Carbon Steel', 'Hardened Steel', 'Low Carbon Steel', 'Tempered Steel', 'Tool Steel']
@st.cache_data()
def load_model():
  model=tf.keras.models.load_model('my_model.hdf5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Steel Microstructure Classification

         -by Hariharasai Mohan (20BME0147)
         """
         )  

file = st.file_uploader("Please upload an image file of microstructure", type=["jpg", "png"])
import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):
        size = (180,180)    
        image = ImageOps.fit(image_data, size)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])
    st.write(
    "This image most likely belongs to {}."
    .format(class_names[np.argmax(score)])
)
