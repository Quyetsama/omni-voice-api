import asyncio
import io
import logging
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
from starlette.concurrency import run_in_threadpool

from omnivoice import OmniVoice, VoiceClonePrompt
from omnivoice.utils.common import get_best_device

# ==========================================
# Cấu hình Logging chuẩn
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("omnivoice_server")

# ==========================================
# Thư mục lưu trữ & Biến môi trường
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOICES_DIR = os.path.join(BASE_DIR, "saved_voices")
STATIC_DIR = os.path.join(BASE_DIR, "static")
INDEX_FILE = os.path.join(STATIC_DIR, "index.html")

os.makedirs(VOICES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Cấu hình Whisper ASR (mặc định True để tự nhận diện ref_text khi để trống, có thể tắt bằng OMNIVOICE_LOAD_ASR=false để tiết kiệm ~1.5GB RAM)
LOAD_ASR = os.getenv("OMNIVOICE_LOAD_ASR", "true").lower() in ("true", "1", "yes")

# ==========================================
# State toàn cục & Concurrency Control
# ==========================================
model: Optional[OmniVoice] = None
voice_cache: Dict[str, VoiceClonePrompt] = {}
inference_lock = asyncio.Lock()  # Đảm bảo chỉ 1 request inference chạy trên model tại 1 thời điểm


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, voice_cache
    device = get_best_device()
    
    # CHỈ dùng float16 trên NVIDIA CUDA. Với Apple MPS và CPU, bắt buộc dùng float32 để tránh lỗi / chậm.
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    
    logger.info(f"Đang nạp model OmniVoice trên thiết bị: {device} ({dtype}), load_asr={LOAD_ASR}...")
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map=device,
        dtype=dtype,
        load_asr=LOAD_ASR,
    )
    logger.info("Model OmniVoice đã sẵn sàng phục vụ!")

    # Tải trước các giọng đã lưu từ đĩa vào RAM Cache
    logger.info("Đang nạp các Voice Profile từ thư mục saved_voices/...")
    loaded_count = 0
    for filename in os.listdir(VOICES_DIR):
        if filename.endswith(".pt"):
            voice_id = os.path.splitext(filename)[0]
            try:
                voice_cache[voice_id] = VoiceClonePrompt.load(
                    os.path.join(VOICES_DIR, filename)
                )
                loaded_count += 1
                logger.debug(f"Đã nạp voice cache: '{voice_id}'")
            except Exception as e:
                logger.error(f"Lỗi nạp voice '{voice_id}': {e}")
    logger.info(f"Đã sẵn sàng với {loaded_count} giọng trong cache!")
    yield
    logger.info("Đang dừng OmniVoice server...")


app = FastAPI(
    title="OmniVoice Studio Local API",
    description="API chuyển đổi văn bản thành giọng nói (TTS) & Voice Cloning có hỗ trợ Cache Voice Profile",
    version="1.3.0",
    lifespan=lifespan,
)

# ==========================================
# Cấu hình CORS chuẩn
# ==========================================
cors_origins_env = os.getenv("CORS_ORIGINS", "*")
if cors_origins_env == "*":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,  # Wildcard không đi kèm allow_credentials=True theo chuẩn W3C
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    allowed_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Phục vụ static files và trang giao diện chính
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
def serve_index():
    if os.path.exists(INDEX_FILE):
        return FileResponse(INDEX_FILE)
    return {"message": "OmniVoice API is running. Visit /docs for Swagger UI."}


class TTSRequest(BaseModel):
    text: str
    voice_id: Optional[str] = None  # Tên giọng đã cache (nếu dùng Voice Cloning)
    instruct: Optional[str] = None  # Thuộc tính giọng (nếu dùng Voice Design)
    speed: Optional[float] = 1.0  # Tốc độ đọc
    num_step: Optional[int] = 32  # Số bước unmasking (16: nhanh, 32: chuẩn chất lượng)
    language: Optional[str] = None  # Mã ngôn ngữ (vd: 'vi', 'en')


@app.get("/health", summary="Kiểm tra trạng thái server")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "cached_voices": list(voice_cache.keys()),
        "asr_enabled": LOAD_ASR,
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

    if not LOAD_ASR and not ref_text:
        raise HTTPException(
            status_code=400,
            detail="Server đang tắt mô hình ASR (LOAD_ASR=False), vui lòng nhập nội dung 'ref_text' thủ công."
        )

    tmp_path = None
    try:
        suffix = os.path.splitext(ref_audio.filename or "")[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_path = tmp_file.name
            content = await ref_audio.read()
            tmp_file.write(content)

        # Trích xuất vector giọng nói an toàn trong threadpool và qua lock
        async with inference_lock:
            prompt = await run_in_threadpool(
                model.create_voice_clone_prompt,
                ref_audio=tmp_path,
                ref_text=ref_text,
            )

        # Lưu vào RAM cache
        voice_cache[voice_id] = prompt

        # Lưu vào đĩa để lần sau khởi động không phải encode lại
        save_path = os.path.join(VOICES_DIR, f"{voice_id}.pt")
        prompt.save(save_path)
        logger.info(f"Đã tạo và lưu cache giọng '{voice_id}' thành công.")

        return {
            "status": "success",
            "voice_id": voice_id,
            "message": f"Đã lưu cache giọng '{voice_id}' thành công!",
        }
    except Exception as e:
        logger.error(f"Lỗi khi xử lý tạo giọng '{voice_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý giọng mẫu: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError as e:
                logger.warning(f"Không thể xóa file tạm {tmp_path}: {e}")


@app.delete("/voices/{voice_id}", summary="Xóa một giọng khỏi cache")
def delete_voice(voice_id: str):
    removed = voice_cache.pop(voice_id, None)
    file_path = os.path.join(VOICES_DIR, f"{voice_id}.pt")
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except OSError as e:
            logger.error(f"Lỗi xóa file {file_path}: {e}")

    if not removed and not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Không tìm thấy voice_id này")
    logger.info(f"Đã xóa giọng '{voice_id}' khỏi cache.")
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
        # Chạy inference trong threadpool tách biệt, đồng thời bảo vệ qua inference_lock
        async with inference_lock:
            audios = await run_in_threadpool(
                model.generate,
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
        logger.error(f"Lỗi khi sinh giọng /tts: {e}", exc_info=True)
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

    if not LOAD_ASR and not ref_text:
        raise HTTPException(
            status_code=400,
            detail="Server đang tắt mô hình ASR (LOAD_ASR=False), vui lòng nhập nội dung 'ref_text' thủ công."
        )

    tmp_path = None
    try:
        suffix = os.path.splitext(ref_audio.filename or "")[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_path = tmp_file.name
            content = await ref_audio.read()
            tmp_file.write(content)

        async with inference_lock:
            audios = await run_in_threadpool(
                model.generate,
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
        logger.error(f"Lỗi khi clone trực tiếp: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi khi clone trực tiếp: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError as e:
                logger.warning(f"Không thể xóa file tạm {tmp_path}: {e}")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
