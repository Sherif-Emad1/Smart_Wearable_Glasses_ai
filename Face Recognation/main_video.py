import cv2
from simple_facerec_cv import SimpleFacerec

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# Load Camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not found or could not be opened.")
    exit()

# Counters for accuracy calculation
total_faces_detected = 0
correct_matches = 0

def calculate_accuracy(total, correct):
    if total == 0:
        return 0.0
    return (correct / total) * 100

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Detect Faces
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

        # Update counters
        total_faces_detected += 1
        if name != "Unknown":
            correct_matches += 1

    # Calculate and display accuracy
    accuracy = calculate_accuracy(total_faces_detected, correct_matches)
    cv2.putText(frame, f"Accuracy: {accuracy:.2f}%", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
