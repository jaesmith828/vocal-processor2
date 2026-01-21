"""
Professional Vocal Processor API - Formant-Preserving Edition
Uses pyrubberband for studio-quality pitch correction with formant preservation.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import librosa
import soundfile as sf
import pyrubberband as pyrb
import io
import traceback
import httpx
import gc

app = FastAPI(title="Vocal Processor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
SCALES = {
    'major': [0, 2, 4, 5, 7, 9, 11],
    'minor': [0, 2, 3, 5, 7, 8, 10],
}
PROCESS_SR = 22050
MAX_DURATION = 120

def detect_key(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
    chroma_avg = np.mean(chroma, axis=1)
    if chroma_avg.sum() > 0:
        chroma_avg /= chroma_avg.sum()
    best_score = 0
    best_key = 0
    best_scale = 'major'
    for root in range(12):
        for scale_name, pattern in SCALES.items():
            score = sum(chroma_avg[(root + note) % 12] for note in pattern)
            if score > best_score:
                best_score = score
                best_key = root
                best_scale = scale_name
    confidence = min(best_score / 0.7, 1.0)
    del chroma
    gc.collect()
    return NOTE_NAMES[best_key], best_scale, confidence

def get_scale_notes_hz(root_name, scale_type):
    root_idx = NOTE_NAMES.index(root_name)
    pattern = SCALES.get(scale_type, SCALES['major'])
    scale_notes = []
    for octave in range(1, 8):
        for degree in pattern:
            midi = (octave + 1) * 12 + (root_idx + degree) % 12
            scale_notes.append(librosa.midi_to_hz(midi))
    return np.array(scale_notes)

def snap_frequency_to_scale(freq, scale_notes):
    freq_val = float(freq) if hasattr(freq, 'item') else freq
    if freq_val <= 0 or np.isnan(freq_val):
        return freq_val
    diffs = np.abs(scale_notes - freq_val)
    nearest_idx = np.argmin(diffs)
    return float(scale_notes[nearest_idx])

def pitch_correct_formant(y, sr, tightness=0.5, humanize=0.2, target_key=None, target_scale='major'):
    if target_key is None:
        detected_key, detected_scale, confidence = detect_key(y, sr)
        target_key = detected_key
        target_scale = detected_scale
    else:
        confidence = 1.0
        detected_key = target_key
        detected_scale = target_scale
    print(f"Using key: {target_key} {target_scale} (confidence: {confidence:.2f})")
    scale_notes = get_scale_notes_hz(target_key, target_scale)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'),
        sr=sr, frame_length=2048, hop_length=512
    )
    voiced_pitches = f0[voiced_flag & ~np.isnan(f0)]
    if len(voiced_pitches) == 0:
        print("No voiced segments detected, returning original")
        return y, detected_key, detected_scale, confidence
    target_pitches = np.array([
        snap_frequency_to_scale(p, scale_notes) if not np.isnan(p) and p > 0 else p
        for p in f0
    ])
    valid_mask = ~np.isnan(f0) & ~np.isnan(target_pitches) & (f0 > 0)
    if not np.any(valid_mask):
        return y, detected_key, detected_scale, confidence
    pitch_ratios = target_pitches[valid_mask] / f0[valid_mask]
    median_ratio = np.median(pitch_ratios)
    n_steps = 12 * np.log2(median_ratio) if median_ratio > 0 else 0
    n_steps = n_steps * tightness
    if humanize > 0:
        n_steps += np.random.normal(0, humanize * 0.1)
    n_steps = np.clip(n_steps, -12, 12)
    print(f"Applying pitch shift: {n_steps:.2f} semitones")
    if abs(n_steps) < 0.05:
        print("Shift too small, returning original")
        return y, detected_key, detected_scale, confidence
    try:
        y_shifted = pyrb.pitch_shift(y, sr, n_steps, rbargs={'--formant': ''})
    except Exception as e:
        print(f"pyrubberband error: {e}, trying without formant flag")
        try:
            y_shifted = pyrb.pitch_shift(y, sr, n_steps)
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            return y, detected_key, detected_scale, confidence
    del f0, voiced_flag, voiced_probs, target_pitches
    gc.collect()
    return y_shifted, detected_key, detected_scale, confidence

@app.get("/")
def health_check():
    return {"status": "ok", "service": "vocal-processor", "version": "4.0-formant"}

@app.get("/health")
def health():
    return {"status": "healthy", "features": ["pitch-correction", "key-detection", "formant-preservation", "direct-upload"]}

@app.post("/process")
async def process_vocal_endpoint(
    file: UploadFile = File(...),
    tightness: float = Form(0.5),
    humanize: float = Form(0.2),
    key: str = Form(None),
    scale: str = Form(None),
    upload_url: str = Form(None),
):
    try:
        print(f"Processing vocal file: {file.filename}")
        audio_bytes = await file.read()
        audio_buffer = io.BytesIO(audio_bytes)
        del audio_bytes
        gc.collect()
        y, sr = librosa.load(audio_buffer, sr=PROCESS_SR, mono=True, duration=MAX_DURATION)
        del audio_buffer
        gc.collect()
        duration = len(y) / sr
        print(f"Loaded audio: {len(y)} samples at {sr}Hz ({duration:.1f}s)")
        y_processed, detected_key, detected_scale, confidence = pitch_correct_formant(
            y, sr, tightness=tightness, humanize=humanize,
            target_key=key if key else None, target_scale=scale if scale else 'major',
        )
        del y
        gc.collect()
        max_val = np.max(np.abs(y_processed))
        if max_val > 0:
            y_processed = y_processed / max_val * 0.95
        output_buffer = io.BytesIO()
        sf.write(output_buffer, y_processed, sr, format='WAV', subtype='PCM_16')
        output_buffer.seek(0)
        processed_bytes = output_buffer.read()
        del y_processed
        gc.collect()
        print(f"Generated {len(processed_bytes)} bytes of processed audio")
        if upload_url:
            print("Uploading to storage...")
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.put(upload_url, content=processed_bytes, headers={"Content-Type": "audio/wav"})
                if response.status_code not in [200, 201]:
                    print(f"Upload failed: {response.status_code} - {response.text}")
                    raise HTTPException(status_code=500, detail=f"Failed to upload to storage: {response.status_code}")
                print("Upload successful!")
            return JSONResponse({
                "success": True, "detected_key": f"{detected_key} {detected_scale}",
                "uploaded": True, "tightness_applied": tightness, "humanize_applied": humanize,
                "confidence": confidence, "sample_rate": sr,
            })
        else:
            return JSONResponse({
                "success": True, "detected_key": f"{detected_key} {detected_scale}",
                "processed_audio": processed_bytes.hex(), "tightness_applied": tightness,
                "humanize_applied": humanize, "confidence": confidence, "sample_rate": sr,
            })
    except Exception as e:
        traceback.print_exc()
        gc.collect()
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
