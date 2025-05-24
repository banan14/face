# Face Recognition from Video

This project provides a Python script to perform real-time (or near real-time, depending on your hardware) face recognition on a video file. It identifies known faces in the video and labels them, while unknown faces are marked as "Unknown".

## Features

- **Load Known Faces**: Easily load images of known individuals from a specified directory. The filename of each image will be used as the person's name.
- **Video Processing**: Analyze a video file frame by frame to detect and recognize faces.
- **Output Video**: Generates a new video file with bounding boxes and names drawn around detected faces.

## Prerequisites

Before running the script, ensure you have the following installed:

- Python 3.6 or higher
- OpenCV
- `face_recognition` library

## Installation

1. **Clone this repository (or download the files):**

   ```bash
   git clone [https://github.com/your-username/face-recognition-project.git](https://github.com/your-username/face-recognition-project.git)
   cd face-recognition-project
