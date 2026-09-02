import mediapipe as mp
import time
from pathlib import Path

import cv2 #cv2 is for the importing opencv library
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


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

        cv2.imshow("Webcam",image)# Display the captured image in a window named "Webcam". The imshow() function is used to display the image in a window. The first argument is the name of the window, and the second argument is the image to be displayed.
        k=cv2.waitKey(1)# Wait for a key event for 1 millisecond. The waitKey() function is used to wait for a key event. The argument 1 specifies the time in milliseconds to wait for a key event. If a key is pressed during this time, the function returns the ASCII value of the key pressed.
        if k==ord('q'): # If the 'q' key is pressed, exit the loop
        #ord('q')is used to get the ASCII value of the 'q' key. The ord() function is used to get the ASCII value of a character. The if statement checks if the ASCII value of the key pressed is equal to the ASCII value of the 'q' key. If it is, the loop is exited using the break statement.
        #as all the alphabetical characters have their own numerical identification.
            break

    video.release() # Release the video capture object
    cv2.destroyAllWindows() # Close all OpenCV windows
