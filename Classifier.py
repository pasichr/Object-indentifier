import time
import picamera
import picamera.array
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('my_model.h5')

# Define the classes to be recognized 

#sample only
classes = ['cat', 'dog', 'bird']

with picamera.PiCamera() as camera:
    # Set camera resolution
    camera.resolution = (640, 480)

    # Start preview
    camera.start_preview()
    time.sleep(2)

    with picamera.array.PiRGBArray(camera) as output:
        while True:
            # Capture an image from the camera
            camera.capture(output, 'rgb')

            # Preprocess the image for the model
            image = output.array
            image = cv2.resize(image, (224, 224))
            image = np.expand_dims(image, axis=0)
            image = image / 255.0

            # Use the model to classify the image
            prediction = model.predict(image)[0]
            class_index = np.argmax(prediction)
            class_name = classes[class_index]

            # Draw the class label on the image
            cv2.putText(image, class_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the resulting image
            cv2.imshow('image', image)

            # Clear the output array
            output.truncate(0)

            # Exit the loop if the 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# Close all windows
cv2.destroyAllWindows()
