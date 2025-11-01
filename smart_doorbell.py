from speechbrain.pretrained import SpeakerRecognition
from speechbrain.inference.speaker import SpeakerRecognition
import torchaudio
import numpy as np
from scipy.spatial.distance import cosine  


from gtts import gTTS

import IPython.display as ipd

def make_greeting(text, path):
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save(path)
    print(f"saved {path}")


def play_greeting(path):
    return ipd.display(ipd.Audio(path, autoplay=True))

verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

registered_files=[
    "/workspaces/IOT_smarthome/audio/soham.mp3"
]

registered_embeddings = []
for file_path in registered_files:
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != 16000: 
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)

    emb = verification.encode_batch(waveform)
    registered_embeddings.append(emb.squeeze().cpu().numpy())  # Flatten to 1D

known_embedding = np.mean(registered_embeddings, axis=0)

def verify_speaker(test_audio_path, threshold=0.7):
    waveform, sample_rate = torchaudio.load(test_audio_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)

    test_emb = verification.encode_batch(waveform).squeeze().cpu().numpy()

    max_similarity = 0
    for reg_emb in registered_embeddings:
        sim = 1 - cosine(test_emb, reg_emb)
        max_similarity = max(max_similarity, sim)

    if max_similarity > threshold:
        print(f"✅ Match! Confidence: {max_similarity:.2f}")
        return True
    else:
        print(f"❌ Rejected. Best match confidence: {max_similarity:.2f} (below {threshold})")
        return False

test_file = "/workspaces/IOT_smarthome/audio/soham.mp3"
is_authorized = verify_speaker(test_file, threshold=0.7)
if is_authorized:
    play_greeting("/workspaces/IOT_smarthome/audio/welcome.mp3")

else:
    play_greeting("/workspaces/IOT_smarthome/audio/reject.mp3")
    # audio = AudioSegment.from_mp3('/content/reject.mp3')
    # play(audio)

