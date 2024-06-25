# NULLCLASS_INTERN_PROJECT_2JECT1
# Drowsiness and Age Detection

This project is a Python application that detects drowsiness and estimates the age of individuals from live camera feed, video files, or images. It uses computer vision techniques and deep learning models to achieve these tasks.

## Features
![image](https://github.com/ARAVINDAN20/NULLCLASS_INTERN_PROJECT_2JECT1/assets/116174602/3a2f0107-18ae-40b1-a82e-ff549eb8aca6)


- **Live Camera Feed**: Detects drowsiness and estimates age in real-time.
- **Upload Video/Image**: Processes uploaded videos or images to detect drowsiness and estimate age.
- **Cancel and Stop**: Options to stop the live camera feed or cancel the processing of uploaded files.

## Setup and Installation

### Prerequisites

- Python 3.x
- Required Python packages:
  - `opencv-python`
  - `dlib`
  - `numpy`
  - `tensorflow`
  - `Pillow`
  - `tkinter`

### Installing Dependencies

Install the required packages using pip:

```sh
pip install opencv-python dlib numpy tensorflow Pillow
```

### Model Files

1. **Face Landmarks Predictor**: Download the `shape_predictor_68_face_landmarks.dat` from [this Google Drive link](https://drive.google.com/file/d/1N3lJNmN44SbyEC6w_siu39t3JTBRXmXH/view?usp=sharing) and place the file in the project directory.

2. **Age Prediction Model**: Ensure you have the `age_model.h5` file in the project directory.(or run this age_model.py)
        DataSet:https://www.kaggle.com/datasets/jangedoo/utkface-new

   
## Download the pre-trained models from the provided links and place them in the appropriate directories:
[age_model.h5 ](https://drive.google.com/file/d/1O0Xztmb0J5pmEiUsbtbc89DKelIOfL6N/view?usp=sharing)


[shape_predictor_68_face_landmarks.dat](https://drive.google.com/file/d/1N3lJNmN44SbyEC6w_siu39t3JTBRXmXH/view?usp=sharing)


### Directory Structure

Ensure your project directory looks something like this:

```bash
NULLCLASS_INTERN_PROJECT_2JECT1/
README.md
shape_predictor_68_face_landmarks.dat
age_model.h5
your_script_name.py
```

## Usage

### Running the Application

1. Clone this repository:

```sh
git clone https://github.com/your-username/drowsiness-age-detection.git
```

2. Navigate to the project directory:

```sh
cd drowsiness-age-detection
```

3. Run the application:

```sh
python your_script_name.py
```

### Interface

- **Start Camera**: Begins the live camera feed for real-time drowsiness and age detection.
- **Upload Video/Image**: Opens a file dialog to select a video or image for processing.
- **Stop Camera**: Stops the live camera feed.
- **Cancel**: Stops the currently processing video or image.

## Contributing

Contributions are welcome. Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- `dlib` library
- `OpenCV` library
- `TensorFlow` library
