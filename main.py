import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

# from util import classify, set_background


# set_background('./bgs/bg5.png')

# # set title
# st.title('Pneumonia classification')

# # set header
# st.header('Please upload a chest X-ray image')

# # upload file
# file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# # load classifier
# model = load_model('./model/pneumonia_classifier.h5')

# # load class names
# with open('./model/labels.txt', 'r') as f:
#     class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
#     f.close()

# # display image
# if file is not None:
#     image = Image.open(file).convert('RGB')
#     st.image(image, use_column_width=True)

#     # classify image
#     class_name, conf_score = classify(image, model, class_names)

#     # write classification
#     st.write("## {}".format(class_name))
#     st.write("### score: {}%".format(int(conf_score * 1000) / 10))
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf

from tempfile import NamedTemporaryFile
from tensorflow.keras.preprocessing import image

st.set_option('deprecation.showfileUploaderEncoding', False)


@st.cache_resource
def loading_model():
    fp = "final_model.keras"
    model_loader = load_model(fp)
    return model_loader


cnn = loading_model()
st.write("""
# AI-based chest-Xray diagnosis system
""")
st.write("""
# Chest X-Ray Classification (Pneumonia/Normal)
""")

temp = st.file_uploader("Upload X-Ray Image")

buffer = temp
temp_file = NamedTemporaryFile(delete=False)
if buffer:
    temp_file.write(buffer.getvalue())
    st.write(image.load_img(temp_file.name))

if buffer is None:
    st.text("Oops! that doesn't look like an image. Try again.")
else:
    hardik_img = image.load_img(
        temp_file.name, target_size=(500, 500), color_mode='grayscale')

    # Preprocessing the image
    pp_hardik_img = image.img_to_array(hardik_img)
    pp_hardik_img = pp_hardik_img / 255
    pp_hardik_img = np.expand_dims(pp_hardik_img, axis=0)

    # Predict
    hardik_preds = cnn.predict(pp_hardik_img)
    if hardik_preds >= 0.5:
        out = 'the image has {:.2%} of being Pneumonia case'.format(
            hardik_preds[0][0])
    else:
        out = 'the image has {:.2%} of being Normal case'.format(
            1 - hardik_preds[0][0])

    st.success(out)

    image = Image.open(temp)
    st.image(image, use_column_width=True)
