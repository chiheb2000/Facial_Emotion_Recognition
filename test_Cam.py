import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

# Load the model architecture from the JSON file
with open("model_architecture.json", "r") as json_file:
    loaded_model_json = json_file.read()

# Recreate the model from the JSON architecture
model = model_from_json(loaded_model_json)

# Load the model weights
model.load_weights("./model/emotion_modelV01.h5")

# Emotion labels corresponding to your model's output
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the frame from the webcam
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Convert the image to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract the face from the image
        face = gray_frame[y:y+h, x:x+w]
        
        # Resize the face image to (48, 48) to match the model's input size
        resized_face = cv2.resize(face, (48, 48))
        
        # Preprocessing: resize and normalize the face image
        normalized_face = resized_face / 255.0
        reshaped_face = np.reshape(normalized_face, (1, 48, 48, 1))
        
        # Predict the emotion
        predictions = model.predict(reshaped_face)
        max_index = np.argmax(predictions[0])
        predicted_emotion = emotion_labels[max_index]
        
        # Display the predicted emotion text on the image around the detected face
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Show the frame with the prediction
    cv2.imshow('Emotion Recognition', frame)
    
    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
