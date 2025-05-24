import cv2
import face_recognition
import os

def load_known_faces(known_faces_dir):
    """
    Loads known face encodings and their corresponding names from a directory.

    Args:
        known_faces_dir (str): The path to the directory containing images of known faces.

    Returns:
        tuple: A tuple containing:
            - list: A list of known face encodings.
            - list: A list of names corresponding to the known face encodings.
    """
    known_face_encodings = []
    known_face_names = []

    if not os.path.exists(known_faces_dir):
        print(f"Error: The directory '{known_faces_dir}' does not exist.")
        return known_face_encodings, known_face_names

    for filename in os.listdir(known_faces_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(known_faces_dir, filename)
            print(f"Loading face from: {image_path}")
            try:
                image = face_recognition.load_image_file(image_path)
                # Check if any faces are found in the image
                face_encodings_in_image = face_recognition.face_encodings(image)
                if face_encodings_in_image:
                    face_encoding = face_encodings_in_image[0]
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(os.path.splitext(filename)[0])  # Name is the filename without extension
                else:
                    print(f"Warning: No face found in {filename}. Skipping.")
            except Exception as e:
                print(f"Error loading image {filename}: {e}")

    print(f"Loaded {len(known_face_encodings)} known faces.")
    return known_face_encodings, known_face_names

def process_video(video_path, known_face_encodings, known_face_names):
    """
    Processes a video, identifies known faces, and saves the output to a new video file.

    Args:
        video_path (str): The path to the input video file.
        known_face_encodings (list): A list of known face encodings.
        known_face_names (list): A list of names corresponding to the known face encodings.
    """
    if not os.path.exists(video_path):
        print(f"Error: The video file '{video_path}' does not exist.")
        return

    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"Error: Could not open video file '{video_path}'.")
        return

    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))

    output_video_path = 'output.avi'
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') # Use MJPG for broader compatibility
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        print(f"Error: Could not create output video file '{output_video_path}'. Ensure you have the necessary codecs.")
        video_capture.release()
        return

    print(f"Processing video: {video_path}")
    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_count += 1
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"

            if known_face_encodings: # Only compare if there are known faces
                # Compare current face to known faces
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                
                # If a match was found in known_face_encodings, use the one with the smallest distance
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = face_distances.argmin()
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        
        # Write the resulting image to the output video file
        out.write(frame)

        if frame_count % 30 == 0: # Print progress every 30 frames
            print(f"Processed {frame_count} frames...")

    print(f"Finished processing video. Output saved to {output_video_path}")
    video_capture.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Define the directory for known faces and the path to your video file
    # IMPORTANT: Replace these with your actual paths!
    known_faces_dir = 'known_faces'
    video_path = 'input_video.mp4' 

    # Create the 'known_faces' directory if it doesn't exist
    if not os.path.exists(known_faces_dir):
        os.makedirs(known_faces_dir)
        print(f"Created directory: {known_faces_dir}. Please place your known face images here.")
        print("Each image filename (e.g., 'John_Doe.jpg') will be used as the person's name.")
        print("Exiting. Please add known faces and run again.")
    else:
        known_face_encodings, known_face_names = load_known_faces(known_faces_dir)
        if known_face_encodings: # Only proceed if known faces are loaded
            process_video(video_path, known_face_encodings, known_face_names)
        else:
            print("No known faces found. Please add images to the 'known_faces' directory.")
            print("Exiting.")
