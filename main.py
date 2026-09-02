import cv2 #cv2 is for the importing opencv library

video=cv2.VideoCapture(0) # Capture video from the default camera as 0 is used
# if extra or other camer are used then instead of 0, the 1 can be used.
#video=variable to store the video capture object
# VideoCapture() is a function that is used to capture video from the camera. The argument 0 is used to specify the default camera.

while True:
    ret, image = video.read()# Read a frame from the video capture object and store it in the variable 'image'. The 'ret' variable is a boolean that indicates whether the frame was read successfully.
    if not ret: # If the frame was not read successfully, break the loop
        break

    cv2.imshow("Webcam",image)# Display the captured image in a window named "Webcam". The imshow() function is used to display the image in a window. The first argument is the name of the window, and the second argument is the image to be displayed.
    k=cv2.waitKey(1)# Wait for a key event for 1 millisecond. The waitKey() function is used to wait for a key event. The argument 1 specifies the time in milliseconds to wait for a key event. If a key is pressed during this time, the function returns the ASCII value of the key pressed.
    if k==ord('q'): # If the 'q' key is pressed, exit the loop
    #ord('q')is used to get the ASCII value of the 'q' key. The ord() function is used to get the ASCII value of a character. The if statement checks if the ASCII value of the key pressed is equal to the ASCII value of the 'q' key. If it is, the loop is exited using the break statement.
    #as all the alphabetical characters have their own numerical identification.
        break

video.release() # Release the video capture object
cv2.destroyAllWindows() # Close all OpenCV windows