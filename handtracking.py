# import cv2 
# from cvzone import HandTrackingModule

# cap=cv2.VideoCapture(0)
# detector=HandTrackingModule.HandDetector()

# while(True):
#     success,img=cap.read()
#     detector.findHands(img)
#     cv2.imshow('Image', img)
#     cv2.waitKey(1)

# import cv2
# import face_recognition

# # Load sample images and encode known faces
# known_face_encodings = []
# known_face_names = []

# # TODO: Load known faces and encode them using face_recognition.face_encodings()

# # Open a video capture
# video_capture = cv2.VideoCapture(0)

# while True:
#     # Capture frame-by-frame
#     ret, frame = video_capture.read()

#     # Convert the image from BGR color to RGB (required for face_recognition library)
#     rgb_frame = frame[:, :, ::-1]

#     # Find all the faces and their encodings in the frame
#     face_locations = face_recognition.face_locations(rgb_frame)
#     face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

#     # Loop through each detected face
#     for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#         # See if the face is a match for the known faces
#         # match = face_recognition.compare_faces(known_face_encodings, face_encoding)
#         # name = "Unknown"
        
#         # TODO: Identify the known faces and their names based on match
        
#         # Draw a rectangle around the face
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#         # Draw a label with a name below the face
#         cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

#     # Display the resulting image
#     cv2.imshow('Video', frame)

#     # Break the loop when 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture and close all OpenCV windows
# video_capture.release()
# cv2.destroyAllWindows()
import cv2
import face_recognition
import os

# def capture_face():
#     # Start video capture from webcam (0 for default webcam)
#     video_capture = cv2.VideoCapture(0)

#     while True:
#         # Capture frame-by-frame
#         ret, frame = video_capture.read()

#         # Convert the frame from BGR color to RGB (required for face_recognition library)
#         rgb_frame = frame[:, :, ::-1]

#         # Find all the faces in the current frame
#         face_locations = face_recognition.face_locations(rgb_frame)

#         # If a face is detected, capture it
#         if face_locations:
#             # Save the captured face image
#             folder_name = 'employees'
#             if not os.path.exists(folder_name):
#                 os.makedirs(folder_name)

#             cv2.imwrite(os.path.join(folder_name, 'employee_face.jpg'), frame)
#             break  # Break out of the loop once the face is captured

#         # Display the video feed
#         cv2.imshow('Video', frame)

#         # Break the loop on 'q' press
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release the video capture and close OpenCV windows
#     video_capture.release()
#     cv2.destroyAllWindows()

import cv2
import time
import os

def detect_faces():
    # Load the pre-trained Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Start video capture from webcam (0 for default webcam)
    video_capture = cv2.VideoCapture(0)

    # Variable to track face detection start time
    start_time = None

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Calculate time remaining for capture after face detection
            if start_time is not None:
                time_elapsed = int(time.time() - start_time)
                time_remaining = max(0, 5 - time_elapsed)

                # Display countdown timer on the video feed
                cv2.putText(frame, f"Capture in: {time_remaining}s", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Check if face is detected and start the timer
            if start_time is None:
                start_time = time.time()

            # Capture and save image after 5 seconds of face detection
            if time.time() - start_time >= 5:
                # Save the captured face image in the 'employees' folder
                folder_name = 'employees'
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)

                cv2.imwrite(os.path.join(folder_name, 'employee_face.jpg'), frame)
                print("Face captured and saved to 'employees' folder")
                video_capture.release()  # Release the video capture
                cv2.destroyAllWindows()  # Close OpenCV windows
                return  # Exit the function and stop face detection

        # Display the video feed with face detection and countdown
        cv2.imshow('Video', frame)

        # Break the loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()



    # attendance
    
import cv2
import face_recognition
import os

def detect_face():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Resizing frame to 50% of its original size
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Scale the face coordinates back to the original frame size if needed
            x *= 2
            y *= 2
            w *= 2
            h *= 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Face Detection', frame)

        if len(faces) > 0:
            recognize_faces(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


def recognize_faces(frame):
    employees_folder = 'employees'
    employee_names = []

    for filename in os.listdir(employees_folder):
        if filename.endswith(".jpg"):
            employee_names.append(os.path.splitext(filename)[0])

    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = []

        for filename in os.listdir(employees_folder):
            if filename.endswith(".jpg"):
                employee_image = face_recognition.load_image_file(os.path.join(employees_folder, filename))
                employee_face_encoding = face_recognition.face_encodings(employee_image)[0]
                match = face_recognition.compare_faces([employee_face_encoding], face_encoding)
                matches.append(match[0])

        if any(matches):
            matched_index = matches.index(True)
            name = employee_names[matched_index]
        else:
            name = "Unowned"  # Set default label for unrecognized faces

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)[0]
        text_x = left + (right - left) // 2 - text_size[0] // 2
        text_y = top - 10

        cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)

    cv2.imshow('Face Detection', frame)







import tkinter as tk

# Create the main window
root = tk.Tk()
root.title("Attendance Form")


# Get screen width and height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Calculate half of the screen size
min_width = int(screen_width * 0.5)
min_height = int(screen_height * 0.5)

# Set minimum width and height
root.minsize(min_width, min_height)



# Function to capture face when button is clicked
def on_button_click():
    # capture_face()
    detect_faces()
    label.config(text="Employee face captured!")



# Function to capture face when button is clicked
def on_button_click2():
    # capture_face()
    detect_face()
    label.config(text="Attendance face captured!")
    

# Create a label
label = tk.Label(root, text="Employee Attendance")
label.pack()



add_employee_button = tk.Button(root, text="Add Employee", command=on_button_click)
add_employee_button.pack()

# Create Button 2
button2 = tk.Button(root, text="Attendance", command=on_button_click2)
button2.pack()

# Run the main loop
root.mainloop()
