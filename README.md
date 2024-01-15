# CLIP Model Inference with FastAPI
# Overview
This project sets up a FastAPI server to perform inference using the CLIP (Contrastive Language-Image Pre-training) model. The provided endpoint allows you to pass a model path, text descriptions, and an image URL to get predictions along with the labeled image.

## Requirements
* Python 3.x
* FastAPI
* Transformers
* PIL (Pillow)
* Requests

## Installation
1. Clone the repository:

* git clone https://github.com/your-username/your-repository.git
* cd your-repository

2. Install dependencies:
* pip install -r requirements.txt
## Usage
Run the FastAPI server:
* uvicorn main:app --reload
* The server will be accessible at http://127.0.0.1:8000.

Open your web browser or use a tool like curl to make requests:

curl -X 'GET' \
  'http://127.0.0.1:8000/predict/?model_path=openai%2Fclip-vit-base-patch32&text_descriptions=%22table%2Cchair%2Crabbit%22&image_url=http%3A%2F%2Fimages.cocodataset.org%2Fval2017%2F000000039769.jpg' \
  -H 'accept: application/json'

Replace the query parameters with your own model path, text descriptions, and image URL.

## Response
The response will include the predicted label and the base64-encoded image. The base64-encoded image can be decoded and displayed in your application.

## Issues
If you encounter any issues or have questions, feel free to open an issue in this repository.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

