# Task4-AlyShawer

This repository contains a Python image recognition tool using TensorFlow and Keras with the ResNet50 model.

## Features

- Loads and preprocesses images from local files or URLs
- Uses ResNet50 pre-trained on ImageNet for predictions
- Displays prediction results with a confidence bar
- Saves results to JSON

## Requirements

- Python 3.8+
- TensorFlow
- NumPy
- Pillow
- requests

## Usage

1. Install dependencies:
   ```bash
   pip install tensorflow numpy pillow requests
   ```

2. Run the script:
   ```bash
   python Task4.py
   ```

3. Follow prompts or modify the script to use your own images.

## Notes

- The first model load may take time while downloading weights.
- Ensure you have internet access if running URL-based image recognition.
