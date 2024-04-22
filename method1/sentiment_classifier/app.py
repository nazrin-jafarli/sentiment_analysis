import os 
from train import main
from inference import run_inference
from fastapi import FastAPI 
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException


app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello from FastAPI!"}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/train_model")
async def train_model():
    # Your model training logic here
    main()
    return {"message": "Model trained successfully"}



import os

from fastapi import HTTPException

@app.post("/write_to_folder")
async def write_to_folder(sentence: str, label: str):
    # Validate label to prevent directory traversal attacks
    allowed_labels = ['positive', 'negative', 'neutral', 'unlabelled']
    if label.lower() not in allowed_labels:
        return {"error": "Invalid label"}

    sentence = sentence.lower()
    folder_path = f"./main_data/{label.lower()}"
    
    try:
        os.makedirs(folder_path, exist_ok=True)
        
        # Change folder permissions
        os.chmod(folder_path, 0o755)
        
        with open(os.path.join(folder_path, f"{label.lower()}_sentences.txt"), "a") as f:
            f.write(sentence + "\n")
        
        return {"message": "Sentence written to folder successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error writing sentence to folder: {e}")


    

@app.post("/inference")
async def infer_sentiment(sentence: str):

    # print(sentence)
    if not sentence.strip():
        raise HTTPException(status_code=400, detail="Empty sentence provided")
    
    sanitized_sentence = sentence.strip()

    try:
        sentiment = run_inference(sanitized_sentence)
        return {"sentiment": sentiment}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run("app:app", port=8000, host="localhost", reload=True)
