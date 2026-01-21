from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health():
    return {"status": "ok"}

@app.post("/process")
async def process_vocal(
    file: UploadFile = File(...),
    tightness: float = Form(0.5),
    humanize: float = Form(0.2),
    key: str = Form(None),
    scale: str = Form(None),
):
    audio_bytes = await file.read()
    detected_key = key if key else "C"
    detected_scale = scale if scale else "major"
    
    return JSONResponse({
        "detected_key": f"{detected_key} {detected_scale}",
        "processed_audio": audio_bytes.hex(),
    })
