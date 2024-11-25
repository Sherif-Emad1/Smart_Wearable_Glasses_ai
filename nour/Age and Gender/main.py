import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Load the trained model
model = load_model('age_gender_model.h5')

# Define the gender dictionary
gender_dict = {0: 'Male', 1: 'Female'}

# Start video capture from webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Resize the frame to 128x128
    resized = cv2.resize(gray, (128, 128))
    
    # Normalize and reshape the image to match model input
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 128, 128, 1))
    
    # Make predictions
    pred = model.predict(reshaped)
    pred_gender = gender_dict[round(pred[0][0][0])]
    pred_age = round(pred[1][0][0])

    # Display the predictions on the frame
    cv2.putText(frame, f'Gender: {pred_gender}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f'Age: {pred_age}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the frame with predictions
    cv2.imshow('Age and Gender Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
