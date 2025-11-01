pip install huggingface

pip install playsound

pip install -q speechbrain torchaudio pydub scipy
apt-get -qq update && apt-get -qq install -y ffmpeg

pip install -q speechbrain torchaudio scipy gtts

apt-get -qq update && apt-get -qq install -y ffmpeg > /dev/null
pip install -q pydub

pip install torch speechbrain torchaudio numpy scipy

export HF_HUB_DISABLE_TELEMETRY=1  # Optional: Silences telemetry
export HF_HUB_ENABLE_HF_TRANSFER=1  # Optional: Faster downloads

pip install huggingface-hub==0.20.0

pip install hf_transfer