# Tomato Classification System

This project is a web application that classifies images of tomatoes as either "ripped" or "unripped" using a pre-trained deep learning model. The application is built using Flask, OpenCV, and TensorFlow, and it features a modern, red-themed glass effect UI using Bootstrap.

## Features

- Upload an image of a tomato to classify it as "ripped" or "unripped".
- Modern, responsive UI with a red-themed glass effect.
- Uses a pre-trained TensorFlow model for classification.

## Prerequisites

- Python 3.6 or higher
- Flask
- OpenCV
- TensorFlow
- NumPy
- Bootstrap (included via CDN)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/tomato-classification-system.git
    cd tomato-classification-system
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    venv\Scripts\activate  # On Windows
    # source venv/bin/activate  # On macOS/Linux
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Ensure you have the pre-trained model file `tomato_classifier.h5` in the project directory.

## Usage

1. Run the Flask application:
    ```sh
    python app.py
    ```

2. Open your web browser and go to `http://127.0.0.1:5000`.

3. Upload an image of a tomato to classify it.

## Project Structure

- `app.py`: The main Flask application file.
- `templates/index.html`: The HTML template for the web application.
- `uploads/`: Directory where uploaded images are stored.
- `tomato_classifier.h5`: Pre-trained TensorFlow model for tomato classification.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Bootstrap for the front-end framework.
- TensorFlow for the deep learning model.
- Flask for the web framework.
- OpenCV for image processing.

## Contact

For any questions or suggestions, please contact [ashutoshpatra616@gmail.com](mailto:ashutoshpatra616@gmail.com).