# Thanh Âm Xứ Nghệ – Luyện Nói Tiếng Nghệ An (VASA)

> **Hồn quê xứ Nghệ – Công nghệ AI bảo tồn tiếng nói quê hương**  
> Ứng dụng Web AI giúp người dùng luyện tập, đối chiếu và đánh giá ngữ điệu tiếng Nghệ An theo chuẩn giọng của người bản địa.

---

## 🌾 Điểm nổi bật của Ứng dụng
- **Giao diện thuần nông thôn Xứ Nghệ:** Tông màu đất, lúa non, giấy dó và rơm vàng thân thiện, ấm áp.
- **Thu âm trực tiếp trên trình duyệt:** Sử dụng Web Audio API chuẩn hóa 16-bit PCM WAV.
- **Trực quan hóa đường cong F0 (Chart.js):** So sánh trực tiếp đường cong cao độ giọng người dùng (màu đỏ gạch) với chuẩn ngữ điệu người bản địa (màu xanh lá).
- **Mô hình AI Đa phương thức:**
  - Mô hình nền tảng: **Pretrained Vietnamese Wav2Vec2** (`nguyenvulebinh/wav2vec2-base-vietnamese-250h`).
  - Phân tích âm học: 26 chiều MFCCs + 6 chiều biến thiên ngữ điệu (Prosodic Modulation).
  - Thuật toán pitch tracking: **Praat Parselmouth** (200 điểm cao độ chuẩn hóa thời gian).
  - Ngân hàng chuẩn bản địa: Tích hợp vector đặc trưng của **363 người Nghệ An bản địa**.
  - Bộ lọc **Multi-Gate VAD**: Loại bỏ 100% âm im lặng, tiếng ồn ngẫu nhiên để chấm điểm chuẩn xác.

---

## 🚀 Hướng Dẫn Chạy Cục Bộ (Local)

### 1. Yêu cầu hệ thống
- Python 3.9 - 3.12 (khuyên dùng Python 3.10 hoặc 3.11).
- Git.

### 2. Cài đặt thư viện
```bash
# Clone repository
git clone https://github.com/hoangsyphuong422002-hub/Luyen-Noi-Tieng-Nghe-An.git
cd Luyen-Noi-Tieng-Nghe-An

# Cài đặt PyTorch CPU (để nhẹ và nhanh)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### 3. Chạy ứng dụng
```bash
# Khởi động FastAPI server
python -m uvicorn src.backend.main:app --host 127.0.0.1 --port 8000 --reload
```
Mở trình duyệt truy cập: `http://127.0.0.1:8000`

---

## 🌐 Hướng Dẫn Deploy Lên Web

Hệ thống AI sử dụng mô hình sâu PyTorch + Transformers Wav2Vec2 (~360MB). Vì **Vercel Serverless Function** giới hạn tối đa 250MB (không thể chạy PyTorch trực tiếp), cách triển khai chuẩn công nghiệp là:

### Phương án A: Deploy Full-Stack Miễn Phí với Hugging Face Spaces (Khuyên Dùng Nhất)
Hugging Face Spaces cung cấp máy chủ CPU **16GB RAM Miễn Phí**, hỗ trợ Docker và PyTorch tiếng Việt native 100%.

1. Vào [Hugging Face Spaces](https://huggingface.co/spaces) -> Chọn **Create new Space**.
2. Đặt tên Space: `luyen-noi-tieng-nghe-an`.
3. Chọn SDK: **Docker** -> **Blank**.
4. Chọn chế độ công khai (Public) -> Bấm **Create Space**.
5. Đẩy toàn bộ mã nguồn của repo này lên Hugging Face Space theo hướng dẫn của Hugging Face (hoặc link trực tiếp từ GitHub repo này).
6. Ứng dụng sẽ tự động build từ `Dockerfile` và bạn có ngay 1 link web hoàn chỉnh (cả frontend lẫn backend AI) cho mọi người truy cập!

### Phương án B: Frontend trên Vercel + Backend trên Hugging Face / Render
- **Frontend (Vercel):**
  1. Vào [Vercel](https://vercel.com) -> Import repository `Luyen-Noi-Tieng-Nghe-An`.
  2. Vercel sẽ tự nhận diện file `vercel.json` và deploy thư mục `src/frontend` lên CDN toàn cầu siêu tốc.
- **Backend AI (Hugging Face / Render / Railway):**
  1. Deploy `Dockerfile` lên Hugging Face Spaces hoặc Render để lấy link backend API (ví dụ: `https://your-backend.hf.space`).
  2. Mở file `src/frontend/app.js`, điền URL backend vào biến:
     ```javascript
     const API_BASE = "https://your-backend.hf.space";
     ```
  3. Commit lên GitHub, Vercel sẽ tự động cập nhật ngay lập tức!

---

## 📁 Cấu Trúc Thư Mục
```
Luyen-Noi-Tieng-Nghe-An/
├── dataset/
│   ├── audios/                 # Các file âm thanh mẫu chuẩn của người Nghệ An
│   └── metadata.csv            # Dữ liệu kịch bản câu thoại, người nói
├── models/
│   ├── native_reference_bank.npz    # Không gian vector 363 người Nghệ An bản địa
│   ├── native_reference_metadata.json
│   └── nghean_ae.pth                # Mô hình baseline
├── src/
│   ├── ai/
│   │   ├── feature_extractor.py     # Trích xuất Wav2Vec2 + MFCC + Prosody
│   │   ├── inference.py             # Hệ thống chấm điểm 2 tầng (Sentence-matched / General)
│   │   └── benchmark_comparison.py  # Đo lường định lượng
│   ├── backend/
│   │   └── main.py                  # FastAPI Application
│   └── frontend/
│       ├── index.html               # Giao diện "Hồn quê xứ Nghệ"
│       ├── style.css                # Thiết kế Country-style Modern UI
│       └── app.js                   # Web Audio Recorder + Chart.js F0
├── Dockerfile                  # Đóng gói container chạy trên mọi nền tảng đám mây
├── vercel.json                 # Cấu hình deploy Frontend tĩnh lên Vercel
├── requirements.txt            # Danh sách thư viện Python
└── README.md
```

---
*Dự án bảo tồn và lan tỏa nét đẹp ngữ điệu miền Trung qua công nghệ AI.*
