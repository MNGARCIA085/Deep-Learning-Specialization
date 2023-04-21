import tensorflow as tf

from typing import Annotated

from fastapi import FastAPI,File, UploadFile
from pydantic import BaseModel

import numpy as np

MODEL = tf.keras.models.load_model('saved_model/my_model')

app = FastAPI()

class UserInput(BaseModel):
    user_input: float

@app.get('/')
async def index():
    return {"Message": "This is Index"}

@app.post('/predict/')
async def predict(UserInput: UserInput):

    prediction = MODEL.predict([UserInput.user_input])

    return {"prediction": float(prediction)}
    
    
    

@app.post("/files/")
async def create_file(file: Annotated[bytes, File()]):
    print(file)
    return {"file_size": len(file)}


@app.post("/pred/")
async def pred(file: UploadFile):
    print(file)
    prediction = MODEL.predict([file.filename])
    return {"filename": file.filename}
    
    


@app.post("/files2")
async def UploadImage(file: bytes = File(...)):
    with open('image.jpg','wb') as image:
        print(image)
        image.write(file)
        print(image)
        image.close()
    return 'got it'
    
    
    
    
    
    
    
    
    
    

