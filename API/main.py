from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
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

# Configurazione CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], #TODO: Inserire URL FE
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/")
async def predict(text: str = Body(..., embed=True)):
    try:
        prediction = detector.predict(text)
        return {"result": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))