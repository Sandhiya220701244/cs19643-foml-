import cv2   
from ultralytics import YOLO
import sys

def main():
    # Path to the exported YOLO model (update with the actual model path)
    # Use a raw string (r'...') to avoid issues with backslashes
    model_path = r'C:\Users\sandh\Documents\DL_SIGN_LAN_PROJ[1]\model_- 4 november 2024 19_04.pt'  # Update as needed

    print("Loading YOLO model from:", model_path)
    try:
        # Load the YOLO model (this will load your exported model)
        model = YOLO(model_path)
        print("YOLO model loaded successfully.")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        sys.exit(1)  # Exit the program if the model fails to load

    print("Attempting to open webcam...")
    # Open the webcam (use 0 for default webcam)
    cap = cv2.VideoCapture(0)  # 0 means the first available webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)  # Exit the program if the webcam fails to open
    else:
        print("Webcam opened successfully.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture image.")
            break  # Exit the loop if frame capture fails

        # Optional: Resize the frame for faster processing
        # frame = cv2.resize(frame, (640, 480))

        # Run inference on the captured frame
        try:
            results = model(frame)  # Pass the current frame to the model
            print("Inference completed.")
        except Exception as e:
            print(f"Error during model inference: {e}")
            break  # Exit the loop if inference fails

        # Render the results on the frame
        try:
            annotated_frame = results[0].plot()  # Get the first result and plot
        except Exception as e:
            print(f"Error annotating frame: {e}")
            annotated_frame = frame  # Fallback to the original frame

        # Display the annotated frame
        cv2.imshow('YOLO Detection', annotated_frame)

        # Extract detections as a pandas DataFrame and print them
        try:
            detections = results[0].pandas().xywh  # Detections
            print("Detections:\n", detections)
        except Exception as e:
            print(f"Error extracting detections: {e}")

        # Break the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

    # Release the capture object and close any open windows
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released and windows closed.")

if __name__ == "__main__":
    main()
