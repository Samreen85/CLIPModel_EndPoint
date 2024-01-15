from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import base64
from io import BytesIO

app = FastAPI()

class Inference:
    def __init__(self, model_path, text_descriptions, image_url):
        self.model_path = model_path
        self.text_descriptions = text_descriptions
        self.image_url = image_url

    def run_inference(self):
        try:
            # Load pre-trained CLIP model and processor
            model = CLIPModel.from_pretrained(self.model_path)
            processor = CLIPProcessor.from_pretrained(self.model_path)

            # Download and open the image
            image = Image.open(requests.get(self.image_url, stream=True).raw)

            # Process inputs
            inputs = processor(text=self.text_descriptions, images=image, return_tensors="pt", padding=True)

            # Run inference
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            
            # Get the probabilities tensor
            probabilities = probs[0].tolist()

            # Iterate over the labels and their corresponding probabilities
            for i, (label, prob) in enumerate(zip(self.text_descriptions, probabilities)):
                print(f"Probability of '{label}' is: {prob}")

            # Get the label with the maximum probability
            max_prob_index = probabilities.index(max(probabilities))
            max_prob_label = self.text_descriptions[max_prob_index]

            # Convert image to base64
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            return {
                "result": f"The provided photo is of '{max_prob_label}' with maximum probability.",
                "image": image_base64
            }

        except Exception as e:
            print(f"An error occurred: {e}")
            raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/predict/")
def predict(model_path: str, text_descriptions: str, image_url: str):
    # Split the comma-separated text descriptions
    text_descriptions = text_descriptions.split(',')

    # Create Inference Class Object
    inference = Inference(model_path, text_descriptions, image_url)
    result = inference.run_inference()

    return JSONResponse(content=result)

# you can run the script with the following command: uvicorn main:app --reload

# Model Path: openai/clip-vit-base-patch32 
# Description: "table,chair,rabbit" 
# Image Url: http://images.cocodataset.org/val2017/000000039769.jpg