import mediapipe as mp
import time
from pathlib import Path

import cv2 #cv2 is for the importing opencv library
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math

def analyze_face_shape(landmarks):
    def distance(p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    # Key landmark indices (approximate for face mesh)
    face_length = distance(landmarks[10], landmarks[152]) # Top to bottom
    forehead_width = distance(landmarks[54], landmarks[284]) # Left to right forehead
    cheekbone_width = distance(landmarks[234], landmarks[454]) # Left to right cheekbones
    jaw_width = distance(landmarks[132], landmarks[361]) # Left to right jaw

    if face_length > 1.3 * cheekbone_width:
        if jaw_width >= 0.9 * cheekbone_width:
            return "Rectangle"
        else:
            return "Oval"
    elif abs(face_length - cheekbone_width) < 0.15 * face_length:
        if jaw_width >= 0.8 * cheekbone_width:
            return "Square"
        else:
            return "Round"
    elif cheekbone_width > forehead_width and cheekbone_width > jaw_width:
        return "Diamond"
    elif forehead_width > cheekbone_width and forehead_width > jaw_width:
        return "Heart"
    else:
        return "Oval"

def draw_virtual_glasses(image, landmarks, width, height):
    # Find eye centers
    left_x = int((landmarks[159].x + landmarks[145].x) / 2 * width)
    left_y = int((landmarks[159].y + landmarks[145].y) / 2 * height)
    
    right_x = int((landmarks[386].x + landmarks[374].x) / 2 * width)
    right_y = int((landmarks[386].y + landmarks[374].y) / 2 * height)

    # Size based on distance between eyes
    eye_distance = math.sqrt((right_x - left_x)**2 + (right_y - left_y)**2)
    radius = int(eye_distance * 0.45)

    # Neon Pink Frame (BGR format)
    color = (255, 0, 255)
    thickness = 5
    
    # 1. Draw the two lenses
    cv2.circle(image, (left_x, left_y), radius, color, thickness)
    cv2.circle(image, (right_x, right_y), radius, color, thickness)
    
    # 2. Draw the nose bridge (connecting inner eyes)
    inner_l_x, inner_l_y = int(landmarks[133].x * width), int(landmarks[133].y * height)
    inner_r_x, inner_r_y = int(landmarks[362].x * width), int(landmarks[362].y * height)
    cv2.line(image, (inner_l_x, inner_l_y), (inner_r_x, inner_r_y), color, thickness)

    # 3. Draw the arms going to the ears
    outer_l_x, outer_l_y = int(landmarks[33].x * width), int(landmarks[33].y * height)
    ear_l_x, ear_l_y = int(landmarks[234].x * width), int(landmarks[234].y * height)
    cv2.line(image, (outer_l_x, outer_l_y), (ear_l_x, ear_l_y), color, thickness)

    outer_r_x, outer_r_y = int(landmarks[263].x * width), int(landmarks[263].y * height)
    ear_r_x, ear_r_y = int(landmarks[454].x * width), int(landmarks[454].y * height)
    cv2.line(image, (outer_r_x, outer_r_y), (ear_r_x, ear_r_y), color, thickness)


model_path = Path(__file__).with_name("face_landmarker.task")
base_options = python.BaseOptions(model_asset_path=str(model_path))
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

video=cv2.VideoCapture(0) # Capture video from the default camera as 0 is used
# if extra or other camer are used then instead of 0, the 1 can be used.
#video=variable to store the video capture object
# VideoCapture() is a function that is used to capture video from the camera. The argument 0 is used to specify the default camera.

with vision.FaceLandmarker.create_from_options(options) as face_landmarker:
    start_time = time.monotonic()
    
    # State variables for scanning effect
    scanning_start_time = None
    final_face_shape = None
    SCAN_DURATION = 3.0 # Duration to show scanning mesh in seconds

    while True:
        ret, image = video.read()# Read a frame from the video capture object and store it in the variable 'image'. The 'ret' variable is a boolean that indicates whether the frame was read successfully.
        if not ret: # If the frame was not read successfully, break the loop
            break
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image.flags.writeable=False
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        timestamp_ms = int((time.monotonic() - start_time) * 1000)
        results = face_landmarker.detect_for_video(mp_image, timestamp_ms)
        #print(results)
        image.flags.writeable=True
        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        if results.face_landmarks:
            if scanning_start_time is None:
                scanning_start_time = time.monotonic()
                final_face_shape = None
            
            elapsed_time = time.monotonic() - scanning_start_time
            
            if elapsed_time < SCAN_DURATION:
                cv2.putText(image, "Scanning Face...", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                for face_landmarks in results.face_landmarks:
                    # Draw landmarks manually since mp.solutions is not available in this environment
                    height, width, _ = image.shape
                    
                    # Draw the tessellation lines (the mesh)
                    try:
                        for connection in vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION:
                            start_idx = connection.start
                            end_idx = connection.end
                            
                            start_pt = face_landmarks[start_idx]
                            end_pt = face_landmarks[end_idx]
                            
                            x1, y1 = int(start_pt.x * width), int(start_pt.y * height)
                            x2, y2 = int(end_pt.x * width), int(end_pt.y * height)
                            
                            cv2.line(image, (x1, y1), (x2, y2), (255, 255, 0), 1)
                    except AttributeError:
                        pass

                    # Draw the points
                    for landmark in face_landmarks:
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)
                        cv2.circle(image, (x, y), 1, (255, 255, 0), 1)
            else:
                if final_face_shape is None:
                    final_face_shape = analyze_face_shape(results.face_landmarks[0])
                
                cv2.putText(image, "Scan Complete!", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image, f"Face Shape: {final_face_shape}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
                # Try on the glasses!
                height, width, _ = image.shape
                draw_virtual_glasses(image, results.face_landmarks[0], width, height)
        else:
            # Reset scan if face is lost
            scanning_start_time = None
            final_face_shape = None

        cv2.imshow("Webcam",image)# Display the captured image in a window named "Webcam". The imshow() function is used to display the image in a window. The first argument is the name of the window, and the second argument is the image to be displayed.
        k=cv2.waitKey(1)# Wait for a key event for 1 millisecond. The waitKey() function is used to wait for a key event. The argument 1 specifies the time in milliseconds to wait for a key event. If a key is pressed during this time, the function returns the ASCII value of the key pressed.
        if k==ord('q'): # If the 'q' key is pressed, exit the loop
        #ord('q')is used to get the ASCII value of the 'q' key. The ord() function is used to get the ASCII value of a character. The if statement checks if the ASCII value of the key pressed is equal to the ASCII value of the 'q' key. If it is, the loop is exited using the break statement.
        #as all the alphabetical characters have their own numerical identification.
            break

    video.release() # Release the video capture object
    cv2.destroyAllWindows() # Close all OpenCV windows
