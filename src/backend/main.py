import os
import sys
import tempfile
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set cache directories to writable /tmp for serverless platforms (Vercel, AWS Lambda)
if not os.environ.get("HF_HOME"):
    os.environ["HF_HOME"] = os.path.join(tempfile.gettempdir(), "huggingface")
if not os.environ.get("TORCH_HOME"):
    os.environ["TORCH_HOME"] = os.path.join(tempfile.gettempdir(), "torch")
import csv
import random
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

# Ensure project root is in sys.path
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.ai.inference import score_audio_file, get_engine

app = FastAPI(
    title="VASA - Vinh Accent Prosody Assessment API",
    description="Mini Game AI chấm điểm ngữ điệu tiếng Nghệ An bằng Auto-Encoder F0 Contour",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(SRC_DIR, "frontend")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
AUDIOS_DIR = os.path.join(DATASET_DIR, "audios")
METADATA_PATH = os.path.join(DATASET_DIR, "metadata.csv")

# Startup event (lazy loading engine so web homepage loads instantly)
@app.on_event("startup")
async def startup_event():
    print("[VASA Backend] FastAPI initialized successfully. Ready to serve requests!")


@app.post("/api/score")
async def evaluate_prosody(
    file: UploadFile = File(...),
    reference_audio: Optional[str] = Form(None)
):
    """
    Receives an uploaded audio file (.wav), extracts multi-modal speech representations
    (Vietnamese Wav2Vec2 + Acoustic Features + 200-pt F0 Contour), and scores native similarity.
    """
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="File âm thanh rỗng.")
        
        result = score_audio_file(audio_bytes, reference_audio=reference_audio)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý âm thanh: {str(e)}")


@app.get("/api/challenges")
async def get_challenges():
    """
    Returns curated Nghe An dialect challenge sentences and real native samples from the dataset.
    Every challenge has 100% matching transcript and native audio file.
    """
    curated = [
        {
            "id": "ds_37_0348.wav",
            "title": "Mẫu thực tế #37_0348 (Từ địa phương: 'vô chợ', 'chợ ni')",
            "dialect_text": "Dịch dã này là họ không vô chợ, họ mua chủ yếu mua ngoài đường với siêu thị hết. Chị ngồi trong chợ ni rất là ế ẩm.",
            "meaning": "Người nói: spk_37_0264 (Nữ) | 'vô chợ' = vào chợ, 'chợ ni' = chợ này.",
            "tip": "Nghe giọng mẫu và chú ý cách nhấn trầm đặc trưng ở cụm 'vô chợ' và 'trong chợ ni rất là ế ẩm'.",
            "sample_audio": "/dataset_audios/37_0348.wav"
        },
        {
            "id": "ds_37_0003.wav",
            "title": "Mẫu thực tế #37_0003 (Từ địa phương: 'nỏ biết chơi ở mô cả')",
            "dialect_text": "Chơi dưới đường thì cũng không được an toàn bởi vì xe cộ đi lại nhiều, mà không chơi ở đây thì cháu cũng nỏ biết chơi ở mô cả, tại vì ở trong nhà mãi cũng chán.",
            "meaning": "Người nói: spk_37_0003 (Nam) | 'nỏ biết chơi ở mô cả' = chẳng biết chơi ở đâu cả.",
            "tip": "Nghe giọng mẫu và luyện cách hạ giọng trầm ở cụm 'nỏ biết chơi ở mô cả'.",
            "sample_audio": "/dataset_audios/37_0003.wav"
        },
        {
            "id": "ds_37_0315.wav",
            "title": "Mẫu thực tế #37_0315 (Từ địa phương: 'bựa ni', 'hắn', 'rất chi là')",
            "dialect_text": "Nhà máy gỗ nó ô nhiễm không khí là khói bụi, bựa ni thì hắn ô nhiễm tiếng ồn. Tiếng ồn quá là ảnh hưởng đến học hành của con cái, ảnh hưởng đến sức khỏe của người già, rất chi là khổ.",
            "meaning": "Người nói: spk_37_0235 (Nam) | 'bựa ni' = dạo này/hôm nay, 'hắn' = nó, 'rất chi là' = rất là.",
            "tip": "Chú ý nhịp ngắt câu và độ nhấn ở cụm 'bựa ni thì hắn ô nhiễm tiếng ồn' và 'rất chi là khổ'.",
            "sample_audio": "/dataset_audios/37_0315.wav"
        },
        {
            "id": "ds_37_0070.wav",
            "title": "Mẫu câu ngắn #37_0070 (Dễ luyện tập)",
            "dialect_text": "Ở nhà cũng có rồi nhưng mà mình muốn cho con ăn ở nhà cho nó yên tâm hơn.",
            "meaning": "Người nói: spk_37_0054 (Nữ) | Câu ngắn gọn, tốc độ vừa phải để làm quen với đồ thị F0.",
            "tip": "Bấm 'Nghe Giọng Mẫu' trước, sau đó giữ nút thu âm và đọc liền mạch cả câu.",
            "sample_audio": "/dataset_audios/37_0070.wav"
        }
    ]

    # Load additional random short samples from dataset metadata.csv
    curated_files = {"37_0348.wav", "37_0003.wav", "37_0315.wav", "37_0070.wav"}
    dataset_samples = []
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, mode="r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                rows = [
                    r for r in reader
                    if 40 <= len(r.get("text", "")) <= 170
                    and r.get("filename") not in curated_files
                    and os.path.exists(os.path.join(AUDIOS_DIR, r.get("filename", "")))
                ]
                sampled = random.sample(rows, min(6, len(rows))) if rows else []
                for r in sampled:
                    fname = r["filename"]
                    dataset_samples.append({
                        "id": f"ds_{fname}",
                        "title": f"Mẫu giọng bản địa #{fname.replace('.wav', '')}",
                        "dialect_text": r["text"],
                        "meaning": f"Người nói: {r['speakerID']} ({'Nam' if str(r['gender']) == '1' else 'Nữ'}) - Tập {r['split']}",
                        "tip": "Bấm 'Nghe Giọng Mẫu' để nghe người Nghệ An bản địa nói, sau đó giữ nút thu âm để luyện tập theo ngữ điệu!",
                        "sample_audio": f"/dataset_audios/{fname}"
                    })
        except Exception as e:
            print(f"Error reading metadata challenges: {e}")

    return {"challenges": curated + dataset_samples}


@app.post("/api/score-sample/{filename}")
async def score_dataset_sample(filename: str):
    """
    Allows testing the AI directly on a native dataset audio file to see the native benchmark contour.
    """
    safe_name = os.path.basename(filename)
    sample_path = os.path.join(AUDIOS_DIR, safe_name)
    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail="Không tìm thấy file âm thanh mẫu.")
    try:
        result = score_audio_file(sample_path, reference_audio=safe_name)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Mount dataset audios for reference playback
if os.path.exists(AUDIOS_DIR):
    app.mount("/dataset_audios", StaticFiles(directory=AUDIOS_DIR), name="dataset_audios")

# Mount static frontend at root (must be mounted last)
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
