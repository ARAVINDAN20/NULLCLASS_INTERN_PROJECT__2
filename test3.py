import cv2
import dlib
import numpy as np
from tensorflow.keras.models import load_model
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

# Function to calculate the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Load the face detector and shape predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load the age prediction model without optimizer state
age_model = load_model("age_model.h5", compile=False)

# Define the indexes for the left and right eyes
(l_start, l_end) = (42, 48)
(r_start, r_end) = (36, 42)

# Initialize the main window
root = Tk()
root.title("Drowsiness and Age Detection")
root.geometry("800x600")

# Label for displaying the video feed
video_label = Label(root)
video_label.pack()

# Initialize video capture
cap = None
running = False

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    sleeping_count = 0

    for face in faces:
        shape = predictor(gray, face)
        shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        left_eye = shape[l_start:l_end]
        right_eye = shape[r_start:r_end]

        ear_left = eye_aspect_ratio(np.array(left_eye))
        ear_right = eye_aspect_ratio(np.array(right_eye))
        ear_avg = (ear_left + ear_right) / 2.0

        # Convert face region to RGB and resize
        face_img = frame[face.top():face.bottom(), face.left():face.right()]
        if face_img.size > 0:
            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_img_rgb = cv2.resize(face_img_rgb, (198, 198))  # Resize to 198x198
            face_img_rgb = face_img_rgb.astype('float32') / 255.0  # Normalize pixel values
            face_img_rgb = np.expand_dims(face_img_rgb, axis=0)  # Add batch dimension

            # Predict age
            age_pred = age_model.predict(face_img_rgb)
            age = int(age_pred[0][0])  # Adjust based on your model's output
        else:
            age = "N/A"

        # Display age and sleepiness status
        cv2.putText(frame, f"Age: {age}", (face.left(), face.bottom() + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if ear_avg < 0.25:
            cv2.putText(frame, "Drowsy!", (face.left(), face.top() - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            sleeping_count += 1

    cv2.putText(frame, f"Sleeping Count: {sleeping_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return frame

def update_frame():
    global cap, running
    if not running:
        return
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't read a frame.")
        return
    frame = process_frame(frame)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    video_label.after(10, update_frame)

def start_camera():
    global cap, running
    if cap is not None:
        cap.release()
    cap = cv2.VideoCapture(0)
    running = True
    update_frame()

def stop_camera():
    global cap, running
    running = False
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

def upload_file():
    global cap, running
    file_path = filedialog.askopenfilename()
    if file_path:
        cap = cv2.VideoCapture(file_path)
        running = True
        update_frame()

def cancel_file():
    global cap, running
    running = False
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

# Buttons for controlling the camera and uploading files
upload_btn = Button(root, text="Upload Video/Image", command=upload_file)
upload_btn.pack(side=LEFT, padx=10, pady=10)

start_btn = Button(root, text="Start Camera", command=start_camera)
start_btn.pack(side=LEFT, padx=10, pady=10)

stop_btn = Button(root, text="Stop Camera", command=stop_camera)
stop_btn.pack(side=LEFT, padx=10, pady=10)

cancel_btn = Button(root, text="Cancel", command=cancel_file)
cancel_btn.pack(side=LEFT, padx=10, pady=10)

# Run the main loop
root.mainloop()

# Release video capture and close all windows
if cap is not None:
    cap.release()
cv2.destroyAllWindows()
