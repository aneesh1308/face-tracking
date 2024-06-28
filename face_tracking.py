import cv2
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import torch

# Load pre-trained Inception ResNet model
resnet = InceptionResnetV1(pretrained='casia-webface').eval()
# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Variables to hold the face data
face_boxes = []
face_embeddings = []

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Convert the frame to a PIL image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Detect faces
    boxes, _ = mtcnn.detect(img)

    if boxes is not None:
        # For each detected face, extract embeddings
        aligned_faces = [mtcnn(img.crop(box)) for box in boxes]
        aligned_faces = [face for face in aligned_faces if face is not None]

        if aligned_faces:
            aligned_faces = torch.stack(aligned_faces)
            embeddings = resnet(aligned_faces).detach()

            # Update the face data
            face_boxes = boxes
            face_embeddings = embeddings

            # Draw bounding boxes around faces
            for box in face_boxes:
                cv2.rectangle(frame, 
                              (int(box[0]), int(box[1])), 
                              (int(box[2]), int(box[3])), 
                              (0, 255, 0), 
                              2)
    
    # Display the resulting frame
    cv2.imshow('Face Detection and Tracking', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
