from fastapi import FastAPI, HTTPException, Body
from contextlib import asynccontextmanager
import detector


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
    # Carica il modello e il vettorizzatore all'avvio dell'applicazione
        detector.loadVectorizer()
        detector.loadModel()
        yield
    except Exception as e:
        print(f"Errore durante il caricamento del modello: {e}")
    
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)



app.post("/predict/")
async def predict(text: str = Body(..., embed=True)):
    try:
        prediction = detector.predict(text)
        return {"result": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))