# CofPred

Dashboard Streamlit dự báo giá cà phê và hỗ trợ ra quyết định.

## Tính năng mới

- Trang **Dữ liệu live** cho cà phê Việt Nam, dầu diesel và thời tiết vùng cà phê.
- Nguồn cà phê có cơ chế dự phòng; lỗi 403 hoặc đổi cấu trúc sẽ không làm web dừng.
- Thời tiết hiện tại và dự báo 7 ngày từ Open-Meteo.
- Bảng xếp hạng nhiều model theo MAE, RMSE, MAPE và Directional Accuracy.
- Hỗ trợ Linear Regression, Ridge, ElasticNet, SVR, kNN, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting, XGBoost và LightGBM.
- Không tạo ảnh model tĩnh. Forecast và feature importance được lưu dạng CSV, sau đó Streamlit vẽ trực tiếp lên web.

## Cài đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dữ liệu master

Đặt file tại:

```text
data/processed/gia_cafe_master_full.csv
```

Script tự tìm cột ngày trong `Ngay`, `date`, `Date`, `ngay` và cột mục tiêu trong `Gia_target`, `Gia`, `price`, `Price`.

## Train nhiều mô hình

```powershell
python train_models.py --horizons 1 5 21 63
```

Chỉ train một số model:

```powershell
python train_models.py --horizons 1 5 --models RandomForest ExtraTrees XGBoost LightGBM
```

Kết quả:

```text
models/model_registry.csv
models/model_*.joblib
results/predictions/*.csv
results/feature_importance/*.csv
```

Các ảnh `results/**/*.png`, `jpg`, `jpeg` đã bị loại khỏi quy trình và được thêm vào `.gitignore`.

## Chạy web

```powershell
streamlit run app.py
```

Dữ liệu live chỉ phục vụ theo dõi. Không nên đưa trực tiếp vào model trước khi lưu snapshot, chuẩn hóa đơn vị, kiểm tra missing/outlier và ghép đúng thời điểm công bố.
