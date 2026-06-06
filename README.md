# Phân tích Văn bản & Phân loại Tự động với ML

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![License](https://img.shields.io/badge/license-MIT-green)

Xây dựng pipeline phân loại văn bản tự động sử dụng
Machine Learning trên bộ dữ liệu đánh giá sản phẩm.

## Mục tiêu

- Xây dựng pipeline NLP hoàn chỉnh từ raw text đến mô hình
- So sánh hiệu quả các thuật toán: Naive Bayes, LR, SVM, RF
- Đạt F1-Score ≥ 0.80 trên tập test

## Cấu trúc dự án

```
├── data/           # Dataset (xem data/README.md)
├── notebooks/      # Jupyter Notebooks theo từng phần
├── src/            # Module Python tái sử dụng
├── models/         # Mô hình đã train
├── reports/        # Báo cáo và biểu đồ
└── requirements.txt
```

## Bắt đầu

```bash
git clone [repo-url]
pip install -r requirements.txt
jupyter notebook notebooks/main_notebook.ipynb
```

## Kết quả

| Mô hình           | Accuracy | F1-Score |
|-------------------|----------|----------|
| Naive Bayes       | ?        | ?        |
| Logistic Regression | ?      | ?        |
| SVM (LinearSVC)   | ?        | ?        |
| Random Forest     | ?        | ?        |

## Dataset

Sử dụng: [Amazon Reviews / Tiki / IMDB]
Nguồn:

## 📜 License

MIT License
