# Hand Gesture Recognition Project

This project aims to recognize hand gestures using Python. It utilizes computer vision techniques to detect and interpret hand movements and gestures captured through a camera.

## Overview

Hand gesture recognition has applications in various fields such as human-computer interaction, sign language recognition, and virtual reality control. This project focuses on developing a Python-based system to accurately recognize and interpret hand gestures in real-time.

## Features

- Real-time hand detection and tracking using computer vision techniques.
- Recognition of predefined hand gestures such as thumbs up, peace sign, and some sign language letters.
- Support for training custom hand gestures using machine learning algorithms.
- Integration with graphical user interfaces (GUIs) for user interaction.

## How to Use

1. Install the required dependencies by running the following command in your terminal:
`pip install -r requirements.txt`
2. Once the requirements are installed, you can run the application by executing the following command:
`python app.py`

This will start the hand gesture recognition app, allowing you to interact with it using your camera. The list of the recognizable gestures can be found in the model folder.

If you want to add more data to the training set, add an argument to the command like this:
`python app.py --add <label>` where label is a non negative integer. When the program is running, press l on the keyboard to save the landmarks as a new instance in the dataset.
