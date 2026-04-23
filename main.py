from fastapi import FastAPI
from fastapi.responses import FileResponse
from TTS.api import TTS

app = FastAPI()

print("載入模型中...")
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

@app.get("/tts")
def generate(text: str):
    output = "output.wav"

    tts.tts_to_file(
        text=text,
        file_path=output,
        speaker_wav="soyorin.wav",  # 👈 重點
        language="ja"
    )

    return FileResponse(output, media_type="audio/wav")