import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Define the array of Kannada letters
kannada_letters = ['ಕ', 'ಖ', 'ಗ', 'ಘ', 'ಙ', 'ಙ-1', 'ಚ', 'ಛ', 'ಜ', 'ಝ', 'ಞ', 'ಟ', 'ಠ', 'ಡ', 'ಢ', 'ಣ', 'ತ', 'ಥ', 'ದ', 'ಧ',
                   'ನ', 'ಪ', 'ಫ್', 'ಬ', 'ಭ', 'ಮ', 'ಯ', 'ರ', 'ಲ', 'ಳ', 'ವ', 'ಶ', 'ಷ', 'ಷ-1', 'ಸ', 'ಹ']

# Load your pre-trained model
model = load_model('C:\\Users\\shash\\PycharmProjects\\pythonProject3\\kannada1.h5')

# Initialize MediaPipe Hand Solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)


# Function to detect hand using MediaPipe and extract the bounding box with padding
def detect_hand(image):
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract the bounding box
            x_min = min([lm.x for lm in hand_landmarks.landmark], default=0)
            x_max = max([lm.x for lm in hand_landmarks.landmark], default=1)
            y_min = min([lm.y for lm in hand_landmarks.landmark], default=0)
            y_max = max([lm.y for lm in hand_landmarks.landmark], default=1)
            h, w, _ = image.shape
            x_min, x_max = int(x_min * w), int(x_max * w)
            y_min, y_max = int(y_min * h), int(y_max * h)

            # Apply padding to the bounding box
            padding_width = int((x_max - x_min) * 0.1)
            padding_height = int((y_max - y_min) * 0.1)
            x_min = max(x_min - padding_width, 0)
            x_max = min(x_max + padding_width, w)
            y_min = max(y_min - padding_height, 0)
            y_max = min(y_max + padding_height, h)

            if x_max > x_min and y_max > y_min:
                hand_region = image[y_min:y_max, x_min:x_max]
                return hand_region, (x_min, y_min, x_max - x_min, y_max - y_min)
    return None, None


# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hand_region, bbox = detect_hand(frame)
    if hand_region is not None and hand_region.size != 0:
        # Flip the hand region horizontally
        hand_region = cv2.flip(hand_region, 1)

        # Create a white background and place the hand region in the center
        background = np.full((224, 224, 3), 255, dtype=np.uint8)  # White background
        resized_hand = cv2.resize(hand_region, (224, 224))
        y_offset = (224 - resized_hand.shape[0]) // 2
        x_offset = (224 - resized_hand.shape[1]) // 2
        background[y_offset:y_offset + resized_hand.shape[0], x_offset:x_offset + resized_hand.shape[1]] = resized_hand

        # Prepare the image for the model
        hand_region_array = img_to_array(background)
        hand_region_array = np.expand_dims(hand_region_array, axis=0)

        predictions = model.predict(hand_region_array)
        predicted_index = np.argmax(predictions)
        prediction_text = f'Index: {predicted_index}'
        predicted_letter = kannada_letters[predicted_index]
        print(f'Predicted letter: {predicted_letter}')

        cv2.putText(frame, prediction_text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Display the hand region with the white background in a separate window
        cv2.imshow('Hand Region', background)

    cv2.imshow('Real-time Hand Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
