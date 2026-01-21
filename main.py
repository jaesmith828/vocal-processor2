"""
Professional Vocal Processor API
Real pitch correction using librosa + pYIN
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import librosa
import soundfile as sf
import io
import traceback
from scipy.interpolate import interp1d

app = FastAPI(title="Vocal Processor API")

# CORS for edge function access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Musical note frequencies (A4 = 440Hz)
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Scale patterns (semitones from root)
SCALES = {
    'major': [0, 2, 4, 5, 7, 9, 11],
    'minor': [0, 2, 3, 5, 7, 8, 10],
}

def hz_to_midi(hz):
    if hz <= 0:
        return 0
    return 12 * np.log2(hz / 440.0) + 69

def midi_to_hz(midi):
    return 440.0 * (2 ** ((midi - 69) / 12))

def get_note_name(midi_note):
    note_idx = int(round(midi_note)) % 12
    octave = int(round(midi_note)) // 12 - 1
    return f"{NOTE_NAMES[note_idx]}{octave}"

def detect_key(pitches, magnitudes):
    valid_pitches = pitches[pitches > 0]
    if len(valid_pitches) == 0:
        return 'C', 'major', 0.5
    
    midi_notes = hz_to_midi(valid_pitches)
    note_classes = np.round(midi_notes) % 12
    
    chroma_counts = np.zeros(12)
    for nc in note_classes:
        chroma_counts[int(nc)] += 1
    
    if chroma_counts.sum() > 0:
        chroma_counts /= chroma_counts.sum()
    
    best_score = 0
    best_key = 0
    best_scale = 'major'
    
    for root in range(12):
        for scale_name, pattern in SCALES.items():
            score = sum(chroma_counts[(root + note) % 12] for note in pattern)
            if score > best_score:
                best_score = score
                best_key = root
                best_scale = scale_name
    
    confidence = min(best_score / 0.7, 1.0)
    return NOTE_NAMES[best_key], best_scale, confidence

def snap_to_scale(midi_note, root_note, scale_pattern, tightness=0.5):
    root_idx = NOTE_NAMES.index(root_note)
    note_class = midi_note % 12
    octave = midi_note // 12
    
    min_dist = 12
    target_note = note_class
    
    for degree in scale_pattern:
        scale_note = (root_idx + degree) % 12
        dist = min(abs(note_class - scale_note), 12 - abs(note_class - scale_note))
        if dist < min_dist:
            min_dist = dist
            target_note = scale_note
    
    target_midi = octave * 12 + target_note
    
    if target_note < note_class - 6:
        target_midi += 12
    elif target_note > note_class + 6:
        target_midi -= 12
    
    corrected = midi_note + (target_midi - midi_note) * tightness
    return corrected

def process_vocal(audio_data, sr, tightness=0.5, humanize=0.2, 
                  target_key=None, target_scale='major', formant_shift=0):
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio_data,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C6'),
        sr=sr,
        frame_length=2048,
        hop_length=512
    )
    
    S = np.abs(librosa.stft(audio_data))
    pitches, magnitudes = librosa.piptrack(S=S, sr=sr)
    
    if target_key is None:
        pitch_vals = pitches.max(axis=0)
        detected_key, detected_scale, confidence = detect_key(pitch_vals, magnitudes)
        target_key = detected_key
        target_scale = detected_scale
    else:
        confidence = 1.0
        detected_key = target_key
        detected_scale = target_scale
    
    scale_pattern = SCALES.get(target_scale, SCALES['major'])
    hop_length = 512
    n_frames = len(f0)
    y_out = audio_data.copy()
    frame_length = 2048
    
    for i in range(n_frames):
        if f0[i] is None or np.isnan(f0[i]) or f0[i] <= 0:
            continue
        if not voiced_flag[i]:
            continue
        
        current_midi = hz_to_midi(f0[i])
        target_midi = snap_to_scale(current_midi, target_key, scale_pattern, tightness)
        
        if humanize > 0:
            variation = np.random.normal(0, humanize * 0.15)
            target_midi += variation
        
        pitch_shift_cents = (target_midi - current_midi) * 100
        start_sample = i * hop_length
        end_sample = min(start_sample + frame_length, len(audio_data))
        
        if end_sample > start_sample and abs(pitch_shift_cents) > 5:
            segment = audio_data[start_sample:end_sample]
            n_steps = pitch_shift_cents / 100.0
            try:
                corrected = librosa.effects.pitch_shift(
                    segment, sr=sr, n_steps=n_steps, res_type='kaiser_fast'
                )
                fade_len = min(256, len(corrected) // 4)
                if fade_len > 0 and len(corrected) == end_sample - start_sample:
                    fade_in = np.linspace(0, 1, fade_len)
                    fade_out = np.linspace(1, 0, fade_len)
                    corrected[:fade_len] *= fade_in
                    corrected[-fade_len:] *= fade_out
                    y_out[start_sample:start_sample + fade_len] *= fade_out
                    y_out[end_sample - fade_len:end_sample] *= fade_in
                y_out[start_sample:end_sample] = corrected
            except Exception:
                pass
    
    if formant_shift != 0:
        y_out = librosa.effects.pitch_shift(y_out, sr=sr, n_steps=formant_shift)
    
    return y_out, detected_key, detected_scale, confidence

@app.get("/")
def health_check():
    return {"status": "ok", "service": "vocal-processor", "version": "2.0-pro"}

@app.get("/health")
def health():
    return {"status": "healthy", "features": ["pitch-correction", "key-detection"]}

@app.post("/process")
async def process_vocal_endpoint(
    file: UploadFile = File(...),
    tightness: float = Form(0.5),
    humanize: float = Form(0.2),
    key: str = Form(None),
    scale: str = Form(None),
    formant_shift: float = Form(0.0),
):
    try:
        audio_bytes = await file.read()
        audio_buffer = io.BytesIO(audio_bytes)
        y, sr = librosa.load(audio_buffer, sr=None, mono=True)
        
        if sr > 48000:
            y = librosa.resample(y, orig_sr=sr, target_sr=48000)
            sr = 48000
        
        y_processed, detected_key, detected_scale, confidence = process_vocal(
            y, sr,
            tightness=tightness,
            humanize=humanize,
            target_key=key if key else None,
            target_scale=scale if scale else 'major',
            formant_shift=formant_shift
        )
        
        max_val = np.max(np.abs(y_processed))
        if max_val > 0:
            y_processed = y_processed / max_val * 0.95
        
        output_buffer = io.BytesIO()
        sf.write(output_buffer, y_processed, sr, format='WAV', subtype='PCM_16')
        output_buffer.seek(0)
        processed_bytes = output_buffer.read()
        
        return JSONResponse({
            "detected_key": f"{detected_key} {detected_scale}",
            "processed_audio": processed_bytes.hex(),
            "tightness_applied": tightness,
            "humanize_applied": humanize,
            "confidence": confidence,
            "sample_rate": sr,
        })
        
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
