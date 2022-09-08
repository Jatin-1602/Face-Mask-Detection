import streamlit as st
import keras
import cv2
import numpy as np
from PIL import Image
from keras.preprocessing import image


model = keras.models.load_model('Trained Model/face_detection_mask')


def upload(image_file):
    img = Image.open(image_file)

    st.image(img)

    test_image = prep_image(img)

    result = predict_result(test_image)

    if result[0][0] == 1:
        prediction = 'no mask'
    else:
        prediction = 'mask'

    st.text(prediction)


def live():
    # run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    c = 1

    # while run:
    while cap.isOpened():
        _, img = cap.read()
        face = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)
        for (x, y, w, h) in face:
            face_img = img[y:y + h, x:x + w]
            cv2.imwrite('temp.jpg', face_img)

            test_image = image.load_img('temp.jpg', target_size=(64, 64, 3))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)

            pred = model.predict(test_image)[0][0]

            if pred == 1:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
                cv2.putText(img, 'NO MASK', ((x + w) // 2, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(img, 'MASK', ((x + w) // 2, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            # datet = str(datetime.datetime.now())
            # cv2.putText(img, datet, (400, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('img', img)

        if cv2.waitKey(1) == ord('q'):
            c = 0
            break

    cap.release()
    cv2.destroyAllWindows()

    if c == 0:
        st.text("Stopped")
        # break


def main():
    """Face Detection App"""

    st.title("Face Detection App")
    st.text("Build with Streamlit, OpenCV")

    activities = ["Live_Detection", "Upload"]
    choice = st.sidebar.selectbox("Select Activty", activities)

    # st.subheader("Face Detection")
    if choice == "Live_Detection":
        live()
    else:
        image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
        # our_image = Image.open(image_file)
        if (image_file):
            st.text("Original Image")
            upload(image_file)


if __name__ == '__main__':
    main()