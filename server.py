import io
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from omnivoice import OmniVoice, VoiceClonePrompt
from omnivoice.utils.common import get_best_device

# Thư mục lưu trữ
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOICES_DIR = os.path.join(BASE_DIR, "saved_voices")
STATIC_DIR = os.path.join(BASE_DIR, "static")
INDEX_FILE = os.path.join(STATIC_DIR, "index.html")

os.makedirs(VOICES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Biến toàn cục lưu trữ mô hình và cache giọng nói
model: Optional[OmniVoice] = None
voice_cache: Dict[str, VoiceClonePrompt] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, voice_cache
    device = get_best_device()
    dtype = torch.float32 if str(device).startswith("mps") else torch.float16
    print(f"--> [OmniVoice] Đang khởi động model trên: {device} ({dtype})...")
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map=device,
        dtype=dtype,
        load_asr=True,
    )
    print("--> [OmniVoice] Model đã sẵn sàng!")

    # Tải trước các giọng đã lưu từ đĩa vào RAM Cache
    print("--> [OmniVoice] Đang tải các Voice Cache từ thư mục saved_voices/...")
    for filename in os.listdir(VOICES_DIR):
        if filename.endswith(".pt"):
            voice_id = os.path.splitext(filename)[0]
            try:
                voice_cache[voice_id] = VoiceClonePrompt.load(
                    os.path.join(VOICES_DIR, filename)
                )
                print(f"    Loaded voice cache: '{voice_id}'")
            except Exception as e:
                print(f"    Lỗi nạp voice '{voice_id}': {e}")
    print(f"--> [OmniVoice] Đã sẵn sàng phục vụ với {len(voice_cache)} giọng trong cache!")
    yield
    print("--> [OmniVoice] Đang dừng server...")


app = FastAPI(
    title="OmniVoice Studio Local API",
    description="API cục bộ chuyển đổi văn bản thành giọng nói có hỗ trợ Cache Giọng Mẫu (VoiceClonePrompt)",
    version="1.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Phục vụ static files và trang giao diện chính tại http://localhost:8000
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def serve_index():
    if os.path.exists(INDEX_FILE):
        return FileResponse(INDEX_FILE)
    return {"message": "OmniVoice API is running. Visit /docs for Swagger UI."}


class TTSRequest(BaseModel):
    text: str
    voice_id: Optional[str] = None  # Tên giọng đã cache (nếu dùng Voice Cloning)
    instruct: Optional[str] = None  # Thuộc tính giọng (nếu dùng Voice Design)
    speed: Optional[float] = 1.0  # Tốc độ đọc
    num_step: Optional[int] = 32  # Số bước (16: nhanh, 32: chuẩn chất lượng)
    language: Optional[str] = None  # Mã ngôn ngữ (vd: 'vi', 'en')


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "cached_voices": list(voice_cache.keys()),
    }


@app.get("/voices", summary="Lấy danh sách các giọng đã được lưu trong Cache")
def list_voices() -> List[str]:
    return list(voice_cache.keys())


@app.post("/voices/create", summary="Tải lên audio mẫu 1 lần để tạo và lưu Cache giọng (Voice Profile)")
async def create_and_cache_voice(
    voice_id: str = Form(..., description="ID định danh giọng (ví dụ: 'giong_mc_nam', 'giong_nu_1')"),
    ref_audio: UploadFile = File(..., description="File audio mẫu (WAV, MP3, 3-10 giây)"),
    ref_text: Optional[str] = Form(None, description="Nội dung câu nói trong file mẫu (tùy chọn)"),
):
    if not model:
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng")

    voice_id = voice_id.strip()
    if not voice_id:
        raise HTTPException(status_code=400, detail="voice_id không được để trống")

    # Lưu tạm audio để encode
    suffix = os.path.splitext(ref_audio.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        content = await ref_audio.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name

    try:
        # Encode giọng mẫu thành embedding token
        prompt = model.create_voice_clone_prompt(
            ref_audio=tmp_path,
            ref_text=ref_text,
        )

        # Lưu vào RAM cache
        voice_cache[voice_id] = prompt

        # Lưu vào đĩa để lần sau khởi động server không phải encode lại
        save_path = os.path.join(VOICES_DIR, f"{voice_id}.pt")
        prompt.save(save_path)

        return {
            "status": "success",
            "voice_id": voice_id,
            "message": f"Đã lưu cache giọng '{voice_id}' thành công!",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý giọng mẫu: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.delete("/voices/{voice_id}", summary="Xóa một giọng khỏi cache")
def delete_voice(voice_id: str):
    removed = voice_cache.pop(voice_id, None)
    file_path = os.path.join(VOICES_DIR, f"{voice_id}.pt")
    if os.path.exists(file_path):
        os.remove(file_path)

    if not removed and not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Không tìm thấy voice_id này")
    return {"status": "success", "message": f"Đã xóa giọng '{voice_id}'"}


@app.post("/tts", summary="Sinh giọng nói từ văn bản (Hỗ trợ gọi qua voice_id đã cache)")
async def text_to_speech(req: TTSRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng")

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Vui lòng nhập văn bản")

    prompt = None
    if req.voice_id:
        if req.voice_id not in voice_cache:
            raise HTTPException(
                status_code=404,
                detail=f"Giọng '{req.voice_id}' chưa tồn tại trong cache. Hãy tạo trước tại /voices/create",
            )
        prompt = voice_cache[req.voice_id]

    try:
        audios = model.generate(
            text=req.text,
            voice_clone_prompt=prompt,
            instruct=req.instruct,
            speed=req.speed,
            num_step=req.num_step,
            language=req.language,
        )

        buffer = io.BytesIO()
        sf.write(buffer, audios[0], model.sampling_rate, format="WAV")
        buffer.seek(0)

        return Response(content=buffer.read(), media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi sinh giọng: {str(e)}")


@app.post("/clone", summary="Clone giọng trực tiếp 1 lần (không lưu cache)")
async def clone_voice_direct(
    text: str = Form(..., description="Văn bản muốn đọc"),
    ref_audio: UploadFile = File(..., description="File audio mẫu"),
    ref_text: Optional[str] = Form(None, description="Transcript mẫu (tùy chọn)"),
    speed: Optional[float] = Form(1.0, description="Tốc độ đọc"),
    num_step: Optional[int] = Form(32, description="Số bước (16-32)"),
    language: Optional[str] = Form(None, description="Mã ngôn ngữ"),
):
    if not model:
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng")

    suffix = os.path.splitext(ref_audio.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        content = await ref_audio.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name

    try:
        audios = model.generate(
            text=text,
            ref_audio=tmp_path,
            ref_text=ref_text,
            speed=speed,
            num_step=num_step,
            language=language,
        )

        buffer = io.BytesIO()
        sf.write(buffer, audios[0], model.sampling_rate, format="WAV")
        buffer.seek(0)

        return Response(content=buffer.read(), media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi clone trực tiếp: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
