import cv2, os

haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'
sub_data = 'GURU'

# Create the directory for storing images if it doesn't exist
path = os.path.join(datasets, sub_data)
if not os.path.isdir(path):
    os.mkdir(path)

(width, height) = (130, 100)
face_cascade = cv2.CascadeClassifier(haar_file)

webcam = cv2.VideoCapture(0)  # Try camera index 0

if not webcam.isOpened():
    print("Error: Camera not detected")
    exit()

count = 1
while count <= 120:  # Capture exactly 70 images
    ret, im = webcam.read()
    if not ret:
        print("Error: Unable to capture image")
        break
    
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw rectangle around face
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))  # Resize the face to the required dimensions
        cv2.imwrite('%s/%s.png' % (path, count), face_resize)  # Save image with the current count as filename
        print(f"Captured image {count}")  # Print which image was captured
        count += 1
        break  # Capture only one face per iteration to avoid multiple faces in a single image
    
    # Display the captured frame with bounding box
    cv2.imshow('OpenCV', im)
    
    # Allow time to stabilize and capture the next frame
    key = cv2.waitKey(200)  # Adjust the wait time (in milliseconds)
    if key == 27:  # Exit if ESC key is pressed
        break

webcam.release()
cv2.destroyAllWindows()
