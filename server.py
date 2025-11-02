from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from speechbrain.pretrained import SpeakerRecognition
import torchaudio
import numpy as np
from scipy.spatial.distance import cosine
import os
import tempfile
from pathlib import Path
from pydub import AudioSegment
import subprocess

# Set ffmpeg path for pydub (if needed)
try:
    result = subprocess.run(['which', 'ffmpeg'], capture_output=True, text=True)
    if result.returncode == 0:
        ffmpeg_path = result.stdout.strip()
        AudioSegment.converter = ffmpeg_path
        print(f"✓ FFmpeg found at: {ffmpeg_path}")
except Exception as e:
    print(f"Warning: Could not locate ffmpeg: {e}")

app = FastAPI(title="Smart Doorbell API")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize SpeakerRecognition model
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# Paths
AUDIO_DIR = Path("/workspaces/IOT_smarthome/audio/")
REGISTERED_FILES = [
    AUDIO_DIR / "soham.mp3"
]
WELCOME_AUDIO = AUDIO_DIR / "welcome.mp3"
REJECT_AUDIO = AUDIO_DIR / "reject.mp3"

# Pre-compute registered embeddings
registered_embeddings = []

def load_and_resample(file_path):
    """Load audio file and resample to 16kHz if needed"""
    # Convert WebM to WAV if necessary
    file_ext = Path(file_path).suffix.lower()
    actual_file_path = file_path
    
    if file_ext in ['.webm', '.ogg', '.m4a']:
        # Convert to WAV using pydub
        try:
            print(f"Converting {file_path} to WAV...")
            # Read the file
            audio = AudioSegment.from_file(file_path)
            wav_path = str(Path(file_path).with_suffix('.wav'))
            # Export as WAV
            audio.export(wav_path, format='wav')
            actual_file_path = wav_path
            print(f"✓ Converted {file_ext} to WAV: {wav_path}")
        except Exception as e:
            print(f"❌ Error converting audio: {e}")
            print(f"File: {file_path}, Extension: {file_ext}")
            raise HTTPException(status_code=500, detail=f"Audio conversion failed: {str(e)}")
    
    try:
        waveform, sample_rate = torchaudio.load(actual_file_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        return waveform
    except Exception as e:
        print(f"❌ Error loading audio with torchaudio: {e}")
        raise HTTPException(status_code=500, detail=f"Audio loading failed: {str(e)}")

def compute_embeddings():
    """Pre-compute embeddings for registered speakers"""
    global registered_embeddings
    registered_embeddings = []
    
    for file_path in REGISTERED_FILES:
        if not file_path.exists():
            print(f"Warning: {file_path} not found")
            continue
            
        waveform = load_and_resample(file_path)
        emb = verification.encode_batch(waveform)
        registered_embeddings.append(emb.squeeze().cpu().numpy())
    
    print(f"✓ Loaded {len(registered_embeddings)} registered speaker(s)")

def verify_speaker(audio_path: str, threshold: float = 0.4):
    """Verify if the speaker in audio_path matches registered speakers"""
    waveform = load_and_resample(audio_path)
    test_emb = verification.encode_batch(waveform).squeeze().cpu().numpy()

    max_similarity = 0
    for reg_emb in registered_embeddings:
        sim = 1 - cosine(test_emb, reg_emb)
        max_similarity = max(max_similarity, sim)

    is_match = max_similarity > threshold
    return is_match, max_similarity

@app.on_event("startup")
async def startup_event():
    """Initialize embeddings on server startup"""
    compute_embeddings()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "registered_speakers": len(registered_embeddings),
        "model": "speechbrain/spkrec-ecapa-voxceleb"
    }

@app.post("/open")
async def verify_and_respond(audio: UploadFile = File(...)):
    """
    Accept audio file, verify speaker, and return appropriate response audio
    
    Args:
        audio: Audio file (mp3, wav, etc.)
    
    Returns:
        FileResponse: welcome.mp3 if authorized, reject.mp3 if not
    """
    
    # Validate file
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Determine file extension from content type or filename
    file_ext = ".webm"  # Default for browser recordings
    if audio.content_type:
        if "webm" in audio.content_type:
            file_ext = ".webm"
        elif "mp3" in audio.content_type:
            file_ext = ".mp3"
        elif "wav" in audio.content_type:
            file_ext = ".wav"
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        content = await audio.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    print(f"📁 Received file: {audio.filename}")
    print(f"📁 Content type: {audio.content_type}")
    print(f"📁 Saved to: {tmp_path}")
    print(f"📁 File size: {os.path.getsize(tmp_path)} bytes")
    
    try:
        # Verify speaker
        is_authorized, confidence = verify_speaker(tmp_path, threshold=0.4)
        
        # Choose response audio
        if is_authorized:
            response_audio = WELCOME_AUDIO
            status = "authorized"
            message = "Access granted! Confidence: {:.2f}".format(confidence)
        else:
            response_audio = REJECT_AUDIO
            status = "rejected"
            message = "Access denied. Confidence: {:.2f} (below 0.7)".format(confidence)
        
        print(f"✅ {message}" if is_authorized else f"❌ {message}")
        
        # Return appropriate audio file
        if not response_audio.exists():
            raise HTTPException(status_code=500, detail=f"Response audio not found: {response_audio}")
        
        return FileResponse(
            path=response_audio,
            media_type="audio/mpeg",
            headers={
                "X-Verification-Status": status,
                "X-Confidence-Score": str(confidence),
                "X-Message": message
            }
        )
    
    except Exception as e:
        print(f"Error during verification: {e}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")
    
    finally:
        # Clean up temporary files
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            # Also remove converted WAV if it exists
            wav_path = str(Path(tmp_path).with_suffix('.wav'))
            if os.path.exists(wav_path) and wav_path != tmp_path:
                os.remove(wav_path)
        except Exception as e:
            print(f"Error cleaning up files: {e}")

@app.post("/register")
async def register_speaker(audio: UploadFile = File(...), name: str = "new_speaker"):
    """
    Register a new speaker (optional endpoint for future use)
    
    Args:
        audio: Audio file of the speaker to register
        name: Name identifier for the speaker
    """
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Save to registered directory
    file_path = AUDIO_DIR / f"{name}.mp3"
    
    with open(file_path, "wb") as f:
        content = await audio.read()
        f.write(content)
    
    # Recompute embeddings
    compute_embeddings()
    
    return {
        "status": "success",
        "message": f"Registered new speaker: {name}",
        "total_registered": len(registered_embeddings)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)