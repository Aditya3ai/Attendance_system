import cv2
import numpy as np
import os

# Set up file paths
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'

# Initialize the face recognition model (use LBPH if FisherFace is unavailable)
try:
    model = cv2.face.FisherFaceRecognizer_create()  # Use this if opencv-contrib-python is installed
except AttributeError:
    print("FisherFaceRecognizer not available, using LBPHRecognizer.")
    model = cv2.face.LBPHFaceRecognizer_create()

# Prepare data for training
print('Training...')
(images, labels, names, id) = ([], [], {}, 0)

# Walk through the directories to get images and labels
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        print(f"Processing {subjectpath}...")  # Debugging statement
        for filename in os.listdir(subjectpath):
            path = os.path.join(subjectpath, filename)
            label = id
            image = cv2.imread(path, 0)
            if image is not None:  # Check if the image is read correctly
                images.append(image)
                labels.append(int(label))
            else:
                print(f"Skipping {path}, unable to read image.")
        id += 1

# Check if images and labels are not empty
if len(images) == 0 or len(labels) == 0:
    print("Error: No images or labels found for training.")
    exit()

# Convert images and labels into numpy arrays
(images, labels) = [np.array(lis) for lis in [images, labels]]

# Train the model
model.train(images, labels)
print("Model trained successfully.")

# Set up face detection
face_cascade = cv2.CascadeClassifier(haar_file)

# Initialize webcam (use camera index 0 if there is an issue with 1)
webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

cnt = 0
while True:
    _, im = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        print("No faces detected.")
    
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (130, 100))

        prediction = model.predict(face_resize)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        if prediction[1] < 800:  # Threshold for classification
            cv2.putText(im, '%s - %.0f' % (names[prediction[0]], prediction[1]), 
                        (x-10, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (51, 255, 255))
            print(names[prediction[0]])
            cnt = 0
        else:
            cnt += 1
            cv2.putText(im, 'Unknown', (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            if cnt > 100:  # Save image after 100 frames if face is still unknown
                print("Unknown Person")
                timestamp = cv2.getTickCount()  # Add timestamp for the image file
                cv2.imwrite(f"unknown/unknown_{timestamp}.jpg", im)  # Save the image of the unknown person
                cnt = 0
    
    cv2.imshow('OpenCV', im)
    key = cv2.waitKey(10)
    if key == 27:  # Exit on ESC
        break

webcam.release()
cv2.destroyAllWindows()
