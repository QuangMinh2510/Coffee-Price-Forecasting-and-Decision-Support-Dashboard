# CofPred

**CofPred: Coffee Price Forecasting and Decision Support Dashboard** là web app
Streamlit dùng dữ liệu cà phê, thời tiết, nhiên liệu và các biến kinh tế để dự
báo giá cà phê theo nhiều chân trời.

Dashboard hiện có tổng quan giá cà phê theo khoảng thời gian, thống kê giá theo
các tỉnh/thị trường Việt Nam, giá Robusta London quốc tế, thống kê theo năm,
lịch sử dầu diesel 2017–2026 và thời tiết theo tỉnh/khu vực từ 2008. Người dùng
có thể chọn khoảng thời gian hoặc nhiều năm, xem biểu đồ và tải từng nguồn CSV.

Phần live lấy ảnh chụp hiện tại từ nhiều nguồn giá cà phê dự phòng, LuatVietnam
và Open-Meteo. Nếu một trang cà phê trả 403 hoặc đổi cấu trúc, web tự chuyển
nguồn và ghi rõ nguồn thực tế. Kết quả được cache 30 phút và có nút làm mới.
Dữ liệu live chỉ dùng để theo dõi; muốn đưa vào model phải cập nhật master, tạo
lại feature rồi retrain.

Trong tab Tổng quan giá cà phê, khi đổi 1 tháng, 3 tháng, 6 tháng, 1 năm hoặc
Tất cả thì giá đầu kỳ, giá cuối kỳ, mức thay đổi, xu hướng và biểu đồ đều được
tính lại theo chính khoảng ngày đã chọn. Bộ lọc này đồng thời cập nhật thống kê
theo tỉnh Việt Nam và thị trường Robusta London (USD/tấn).

## 1. Cài môi trường

Khuyến nghị Python 3.11 hoặc 3.12.

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Kiểm tra dữ liệu

File web và script train mặc định đọc:

```text
data/processed/gia_cafe_master_full.csv
```

Dữ liệu hiện tại có 1.566 dòng, 27 cột, từ 01/01/2020 đến 31/12/2025.

## 3. Train và xuất model

Chạy nhanh trước để kiểm tra toàn bộ luồng:

```bash
python train_models.py --horizons 1 5 --models LinearRegression RandomForest --rf-estimators 100
```

Chạy đầy đủ bốn model và bốn chân trời:

```bash
python train_models.py --horizons 1 5 21 63 --models LinearRegression RandomForest XGBoost LightGBM
```

Kết quả được tạo tại:

- `models/model_*.joblib`: model đã train và metadata để inference.
- `models/model_registry.csv`: MAE, RMSE, MAPE và Directional Accuracy.
- `results/preds/web/*.csv`: dự báo trên tập kiểm thử theo thời gian.

Mặc định model học `delta`, tức là dự báo mức tăng/giảm rồi cộng với giá tại
ngày gốc. Có thể thử dự báo trực tiếp mức giá bằng `--mode level`.

## 4. Chạy web

```bash
streamlit run app.py
```

Mở địa chỉ Streamlit in ra trong terminal, thường là
`http://localhost:8501`.

## 5. Đưa lên Streamlit Community Cloud

1. Push `app.py`, `train_models.py`, `requirements.txt`, thư mục `models/` và
   file master CSV lên GitHub.
2. Trên Streamlit Community Cloud, chọn repository `Baottq-dev/cofpred`.
3. Chọn branch `main`, main file `app.py`, rồi Deploy.

Không train model lại mỗi khi người dùng mở web. Hãy train trước, lưu file
`.joblib` vào repo, sau đó web chỉ nạp model để dự báo. Nếu model quá lớn, có
thể chuyển sang Git LFS hoặc lưu model ở object storage.

## 6. Trạng thái Daily Data Collection

Repo đã có script lấy thời tiết, nhiên liệu và tỷ giá, nhưng chưa có một lệnh
duy nhất cập nhật chắc chắn toàn bộ nguồn đến ngày hiện tại. Vì vậy web ghi rõ
"dữ liệu mới nhất" thay vì gọi là "giá hôm nay". Bước tiếp theo là chuẩn hóa
pipeline cập nhật dữ liệu, chạy lại `41_build_master_full.py`, sau đó lên lịch
GitHub Actions và retrain theo tuần/tháng.
