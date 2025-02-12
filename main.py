# import streamlit as st
# import numpy as np
# from PIL import Image
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# from tempfile import NamedTemporaryFile

# st.set_option('deprecation.showfileUploaderEncoding', False)

# # Load the model without caching
# def loading_model():
#     fp = "final_model.keras"
#     model_loader = load_model(fp)
#     return model_loader

# cnn = loading_model()

# st.write("""
# # AI-based chest-Xray diagnosis system
# """)
# st.write("""
# # Chest X-Ray Classification (Pneumonia/Normal)
# """)

# temp = st.file_uploader("Upload X-Ray Image")

# buffer = temp
# temp_file = NamedTemporaryFile(delete=False)
# if buffer:
#     temp_file.write(buffer.getvalue())
#     st.write(image.load_img(temp_file.name))

# if buffer is None:
#     st.text("Oops! that doesn't look like an image. Try again.")
# else:
#     hardik_img = image.load_img(
#         temp_file.name, target_size=(500, 500), color_mode='grayscale')

#     # Preprocessing the image
#     pp_hardik_img = image.img_to_array(hardik_img)
#     pp_hardik_img = pp_hardik_img / 255.0
#     pp_hardik_img = np.expand_dims(pp_hardik_img, axis=0)

#     # Predict
#     hardik_preds = cnn.predict(pp_hardik_img)
#     if hardik_preds >= 0.5:
#         out = 'The image has {:.2%} of being a Pneumonia case'.format(hardik_preds[0][0])
#     else:
#         out = 'The image has {:.2%} of being a Normal case'.format(1 - hardik_preds[0][0])

#     st.success(out)

#     image = Image.open(temp)
#     st.image(image, use_column_width=True)


#     st.success(out)

#     image = Image.open(temp)
#     st.image(image, use_column_width=True)
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tempfile import NamedTemporaryFile

st.set_option('deprecation.showfileUploaderEncoding', False)

# Load the model
def load_model_h5():
    return load_model('final_model.h5')

cnn = load_model_h5()

st.write("""
# Culture and science city
# project: AI-based chest-Xray diagnosis system
# Chest X-Ray Classification (Pneumonia/Normal)
""")
st.write("""
# Made by : Ammar yasser ,Omar essam, Mohammed Afifi 
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
    pp_hardik_img = pp_hardik_img / 255.0
    pp_hardik_img = np.expand_dims(pp_hardik_img, axis=0)

    # Predict
    hardik_preds = cnn.predict(pp_hardik_img)
    if hardik_preds >= 0.5:
        out = 'The image has {:.2%} of being a Pneumonia case'.format(hardik_preds[0][0])
    else:
        out = 'The image has {:.2%} of being a Normal case'.format(1 - hardik_preds[0][0])

    st.success(out)

    image = Image.open(temp)
    st.image(image, use_column_width=True)

