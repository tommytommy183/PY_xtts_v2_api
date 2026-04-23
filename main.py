from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from TTS.api import TTS
import shutil, os, subprocess
import librosa
import numpy as np

app = FastAPI()

print("loading model...")
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

DEFAULT_SPEAKER = "soyorin.wav"


def preprocess_wav(input_path: str, output_path: str):
    """轉成 XTTS v2 最佳格式"""
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "22050",
        "-ac", "1",
        "-af", "silenceremove=start_periods=1:start_silence=0.1:start_threshold=-50dB,loudnorm",
        output_path
    ], check=True)


def check_audio(wav_path: str):
    y, sr = librosa.load(wav_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    f0, _, _ = librosa.pyin(y, fmin=50, fmax=500)
    mean_f0 = float(np.nanmean(f0)) if not np.all(np.isnan(f0)) else 0

    warnings = []
    if duration < 5:
        warnings.append("音檔過短，建議至少 6 秒")
    if mean_f0 > 250:
        warnings.append(f"音高偏高 ({mean_f0:.0f}Hz)，中文合成可能沙啞，建議降音調後再試")
    
    return warnings


@app.post("/tts/clone")
async def tts_clone(text: str, file: UploadFile = File(...), lang: str = "zh-cn"):
    temp_raw = "temp_raw.wav"
    temp_clean = "temp_clean.wav"
    output = "output_clone.wav"

    # 儲存原始音檔
    with open(temp_raw, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 前處理
    preprocess_wav(temp_raw, temp_clean)

    # 品質檢查（回傳 warning，不擋請求）
    warnings = check_audio(temp_clean)

    # 合成
    tts.tts_to_file(
        text=text,
        file_path=output,
        speaker_wav=temp_clean,
        language=lang  # ⚠️ 注意用 "zh-cn" 不是 "zh"
    )

    # 清理
    os.remove(temp_raw)
    os.remove(temp_clean)

    headers = {"X-Warnings": "; ".join(warnings)} if warnings else {}
    return FileResponse(output, media_type="audio/wav", headers=headers)