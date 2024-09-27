import cv2
import mediapipe as mp
from PyPDF2 import PdfReader, PdfWriter
from time import sleep

# Initialize Mediapipe's face mesh detector
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Function to get eyebrow movement from landmarks
def get_eyebrow_movement(landmarks):
    # Mediapipe facial landmarks for eyebrows
    left_eyebrow_ids = [55, 65, 52, 53, 46]  # Adjust these indices based on Mediapipe's mesh
    right_eyebrow_ids = [285, 295, 282, 283, 276]  # Adjust these indices based on Mediapipe's mesh
    
    left_avg_y = sum([landmarks[i].y for i in left_eyebrow_ids]) / len(left_eyebrow_ids)
    right_avg_y = sum([landmarks[i].y for i in right_eyebrow_ids]) / len(right_eyebrow_ids)

    return left_avg_y, right_avg_y

# Function to process each video frame
def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    action = None

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_y, right_y = get_eyebrow_movement(face_landmarks.landmark)
            
            # Example thresholds for eyebrow positions (scaled 0 to 1)
            if left_y < 0.4:  # Eyebrow raised
                action = 'scroll_up'
            elif left_y > 0.6:  # Eyebrow lowered
                action = 'scroll_down'
    
    return action

# Function to scroll through PDF based on action
def scroll_pdf(action):
    pdf_path = 'commitment-2024 (1).pdf'
    output_path = './output.pdf'
    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    
    current_page = 0
    
    # Find current page index
    for i, page in enumerate(reader.pages):
        if page in writer.pages:
            current_page = i
            break

    if action == 'scroll_up':
        # Move to the previous page
        current_page = max(0, current_page - 1)
    elif action == 'scroll_down':
        # Move to the next page
        current_page = min(len(reader.pages) - 1, current_page + 1)

    # Add the page to the writer
    writer.add_page(reader.pages[current_page])
    
    with open(output_path, 'wb') as f:
        writer.write(f)

# Main function to capture video and process gestures
def main():
    cap = cv2.VideoCapture(0)  # Start video capture
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        action = process_frame(frame)
        if action:
            print(f"Detected action: {action}")
            scroll_pdf(action)
            sleep(1)  # Add delay to avoid rapid scrolling

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
