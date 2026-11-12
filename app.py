import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Hugging Face ZeroGPU compatibility
try:
    import spaces
    @spaces.GPU(duration=60)
    def zerogpu_probe():
        import torch
        return torch.cuda.is_available()
    print("[HuggingFace] ZeroGPU support detected.")
except Exception as e:
    print(f"[HuggingFace] CPU mode: {e}")

from src.backend.main import app

# Hugging Face Spaces Gradio SDK compatibility (100% Free tier)
try:
    import gradio as gr
    with gr.Blocks(title="Thanh Âm Xứ Nghệ - VASA") as demo:
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h2>🌾 Thanh Âm Xứ Nghệ – Luyện Nói Tiếng Nghệ An</h2>
            <p>Ứng dụng đang chạy máy chủ AI đầy đủ. Hãy truy cập giao diện chính tại trang chủ root: <a href="/" target="_blank">Mở Web Game</a></p>
        </div>
        """)
    app = gr.mount_gradio_app(app, demo, path="/gradio")
except Exception as e:
    print(f"[App] Running pure FastAPI mode: {e}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
