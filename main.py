import cv2

def get_color_name(hue, saturation, value):
    if saturation < 40:
        return "WHITE"
    elif value < 20:
        return "BLACK"
    elif hue < 5 or hue >= 170:
        return "RED"
    elif hue < 22:
        return "ORANGE"
    elif hue < 33:
        return "YELLOW"
    elif hue < 45:
        return "LIGHT GREEN"
    elif hue < 78:
        return "GREEN"
    elif hue < 90:
        return "CYAN"
    elif hue < 131:
        return "BLUE"
    elif hue < 145:
        return "INDIGO"
    elif hue < 160:
        return "VIOLET"
    elif hue < 170:
        return "MAGENTA"
    else:
        return "No Recognition"

def process_frame(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height, width, _ = frame.shape
    center_x, center_y = width // 2, height // 2

    pixel_center = hsv_frame[center_y, center_x]
    hue, saturation, value = pixel_center[0], pixel_center[1], pixel_center[2]

    color_name = get_color_name(hue, saturation, value)

    pixel_center_bgr = frame[center_y, center_x]
    b, g, r = int(pixel_center_bgr[0]), int(pixel_center_bgr[1]), int(pixel_center_bgr[2])

    cv2.rectangle(frame, (center_x - 220, 10), (center_x + 200, 120), (255, 255, 255), -1)
    cv2.putText(frame, color_name, (center_x - 200, 100), 0, 3, (b, g, r), 5)
    cv2.circle(frame, (center_x, center_y), 5, (25, 25, 25), 3)

    return frame

def main():
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame)
        cv2.imshow("Frame", processed_frame)

        if cv2.waitKey(1) == 27:  # ESC key to break
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
