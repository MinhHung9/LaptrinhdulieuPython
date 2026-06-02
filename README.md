[README (1).md](https://github.com/user-attachments/files/28518465/README.1.md)
# Phân Tích Cảm Xúc Đánh Giá Tiki — NLP Project

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange.svg)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Dự án phân tích và phân loại cảm xúc (Sentiment Classification) từ đánh giá sản phẩm của khách hàng trên sàn thương mại điện tử Tiki, sử dụng các kỹ thuật NLP truyền thống.

---

## Mục lục

- [Tổng quan dự án](#-tổng-quan-dự-án)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Dataset](#-dataset)
- [Hướng dẫn cài đặt](#-hướng-dẫn-cài-đặt)
- [Chạy code](#-chạy-code)
- [Phần A — Chi tiết thực hiện](#-phần-a--chi-tiết-thực-hiện)
  - [1: EDA — Khám phá dữ liệu](#a1-eda--khám-phá-dữ-liệu)
  - [2: Thống kê độ dài văn bản](#a2-thống-kê-độ-dài-văn-bản)
  - [3: Tiền xử lý văn bản](#a3-tiền-xử-lý-văn-bản)
  - [4: Trích xuất đặc trưng](#a4-trích-xuất-đặc-trưng)
  - [5: Phân tích Data Leakage](#a6-phân-tích-data-leakage)
- [Kết quả & Biểu đồ](#-kết-quả--biểu-đồ)
- [Lý thuyết NLP](#-lý-thuyết-nlp)

---

## Tổng quan dự án

Dự án thực hiện **pipeline NLP đầy đủ** từ khám phá dữ liệu đến trích xuất đặc trưng cho bài toán phân loại cảm xúc (positive / negative / neutral) từ đánh giá tiếng Việt trên Tiki.

**Công nghệ sử dụng:**
- Python 3.8+
- Pandas, NumPy — xử lý dữ liệu
- Scikit-learn — vectorization & modeling
- Matplotlib, Seaborn — visualization

---

## Cấu trúc thư mục

```
tiki-sentiment-analysis/
│
├── notebooks/
│   └── PartA_NLP_Tiki_Sentiment.ipynb   # Jupyter notebook đầy đủ Phần A
│
├── src/
│   └── part_A_analysis.py               # Script Python có thể chạy độc lập
│
├── data/
│   ├── tiki_reviews.csv                 # Dataset (từ repo gốc)
│   └── generate_sample.py               # Script tạo dữ liệu mẫu (dev only)
│
├── outputs/
│   └── figures/
│       ├── A1_label_distribution.png
│       ├── A2_text_length.png
│       └── A4_bow_vs_tfidf.png
│
├── requirements.txt
└── README.md
```

---

## Dataset

**Nguồn:** [quangvh23/tiki-dataset-sentiment-classification](https://github.com/quangvh23/tiki-dataset-sentiment-classification)

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `review` | string | Nội dung đánh giá của khách hàng |
| `label` | int | Nhãn cảm xúc: 0=Tiêu cực, 1=Tích cực, 2=Trung lập |
| `rating` | int | Sao đánh giá (1–5) |

**Tải dataset về máy:**
```bash
# Cách 1: Clone repo gốc
git clone https://github.com/quangvh23/tiki-dataset-sentiment-classification
cp tiki-dataset-sentiment-classification/1a3/tiki_reviews.csv data/

# Cách 2: Dùng dữ liệu mẫu (nếu chưa có file thật)
python data/generate_sample.py
```

---

## Hướng dẫn cài đặt

```bash
# Clone repo này
git clone https://github.com/YOUR_USERNAME/tiki-sentiment-analysis.git
cd tiki-sentiment-analysis

# Tạo virtual environment (khuyến nghị)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Cài thư viện
pip install -r requirements.txt
```

---

## Chạy code

### Cách 1: Jupyter Notebook (khuyến nghị)
```bash
jupyter notebook notebooks/PartA_NLP_Tiki_Sentiment.ipynb
```
→ Chạy từng cell theo thứ tự, có giải thích chi tiết từng bước.

### Cách 2: Script Python
```bash
cd src
python part_A_analysis.py
```
→ Chạy toàn bộ pipeline, kết quả và biểu đồ lưu vào `outputs/figures/`.

---

## Phần A — Chi tiết thực hiện

### 1: EDA — Khám phá dữ liệu

**Mục tiêu:** Tải dữ liệu, mô tả cấu trúc, phát hiện mất cân bằng nhãn.

```python
df = pd.read_csv('../data/tiki_reviews.csv', encoding='utf-8')
print(df.head(10))
print(df.dtypes)
print(df['label'].value_counts())
```

**Kết quả điển hình:**

| Nhãn | Số mẫu | Tỷ lệ |
|------|--------|--------|
| Tích cực (1) | 500 | 62.5% |
| Tiêu cực (0) | 200 | 25.0% |
| Trung lập (2) | 100 | 12.5% |

** Nhận xét — Mất cân bằng dữ liệu (Class Imbalance):**

Dataset có sự **mất cân bằng rõ rệt**, ảnh hưởng đến huấn luyện theo các cách sau:

1. **Bias về nhãn đa số:** Mô hình có xu hướng dự đoán "Tích cực" nhiều hơn vì đây là nhãn chiếm đa số trong tập train. Điều này dẫn đến **accuracy cao nhưng recall thấp** với nhãn thiểu số (Trung lập).

2. **Metrics gây hiểu nhầm:** Chỉ dự đoán "Tích cực" cho tất cả có thể đạt accuracy 62.5% — nhưng đây là mô hình vô nghĩa.

3. **Giải pháp:**
   - `class_weight='balanced'` trong sklearn
   - SMOTE (Synthetic Minority Over-sampling Technique)
   - Undersampling nhãn đa số

---

### 2: Thống kê độ dài văn bản

```python
df['num_words'] = df['review'].apply(lambda x: len(str(x).split()))
df['num_chars'] = df['review'].apply(lambda x: len(str(x)))

stats = df.groupby('label_name').agg(
    TB_từ=('num_words', 'mean'),
    Max_từ=('num_words', 'max'),
    TB_ký_tự=('num_chars', 'mean'),
)
```

** Nhận xét:**
- **Tiêu cực** thường dài hơn: người dùng giải thích chi tiết lý do không hài lòng
- **Tích cực** ngắn gọn hơn: "Tốt lắm", "Rất hài lòng"
- **Tương quan yếu** giữa độ dài và nhãn — nội dung từ ngữ quan trọng hơn độ dài

---

### 3: Tiền xử lý văn bản

**Pipeline đầy đủ:**

```python
def preprocess_text(text):
    text = text.lower()                            # 1. Lowercase
    text = re.sub(r'<[^>]+>', ' ', text)           # 2. Xóa HTML tags
    text = re.sub(r'http\S+|www\.\S+', ' ', text)  # 3. Xóa URL
    text = re.sub(r'\d+', ' ', text)               # 4. Xóa số
    text = re.sub(r'[^\w\s]', ' ', text)           # 5. Xóa ký tự đặc biệt
    tokens = text.split()                           # 6. Tokenization
    tokens = [t for t in tokens                     # 7. Xóa stopwords
              if t not in VIETNAMESE_STOPWORDS]
    tokens = [simple_stemmer_vi(t) for t in tokens] # 8. Stemming
    tokens = [t for t in tokens if len(t) >= 2]    # 9. Xóa từ quá ngắn
    return ' '.join(tokens)
```

**Lý do cần loại bỏ stopwords**

Stopwords là từ xuất hiện rất thường xuyên nhưng không mang thông tin phân biệt:

| Stopword | Lý do không phân biệt |
|----------|------------------------|
| **"và"** | Liên từ, xuất hiện đồng đều ở mọi nhãn |
| **"của"** | Giới từ sở hữu, không chứa cảm xúc |
| **"rất"** | Đi kèm cả "rất tốt" lẫn "rất tệ" |
| **"là"** | Động từ liên kết, không có nghĩa cảm xúc |
| **"có"** | Xuất hiện trong mọi ngữ cảnh tích cực và tiêu cực |

**Stemming vs Lemmatization:**

| Tiêu chí | Stemming | Lemmatization |
|----------|----------|----------------|
| Cơ chế | Cắt bỏ hậu tố | Tra từ điển, trả về dạng gốc |
| Tốc độ | Nhanh hơn | Chậm hơn |
| Độ chính xác | Thấp hơn | Cao hơn |
| Tiếng Việt |  Phù hợp (chuẩn hóa teen code) |  Ít thư viện hỗ trợ |

**→ Chọn Stemming** vì tiếng Việt là ngôn ngữ đơn lập (không biến hình từ theo ngữ pháp), và Stemming giúp chuẩn hóa teen code tiếng Việt hiệu quả.

---

### 4: Trích xuất đặc trưng

#### Bag of Words (CountVectorizer)
```python
bow_vectorizer = CountVectorizer(max_features=10000)
X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow  = bow_vectorizer.transform(X_test)    # Chỉ transform!
```

#### TF-IDF (TfidfVectorizer)
```python
tfidf_vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),      # Unigram + Bigram
    min_df=2,                # Bỏ từ quá hiếm
    max_df=0.95,             # Bỏ từ quá phổ biến
    sublinear_tf=True,       # Dùng log(tf)
)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf  = tfidf_vectorizer.transform(X_test)   # Chỉ transform!
```

**TF-IDF khác gì so với Bag of Words?**

**BoW** chỉ đếm tần suất xuất hiện, không phân biệt từ quan trọng hay không.

**TF-IDF** = TF × IDF:

$$\text{TF-IDF}(t, d) = \frac{f_{t,d}}{\sum_{t'} f_{t',d}} \times \log\left(\frac{N}{df_t}\right)$$

- **TF (Term Frequency):** Tần suất từ trong tài liệu → từ xuất hiện nhiều trong tài liệu đó
- **IDF (Inverse Document Frequency):** `log(N / số tài liệu có từ t)` → từ xuất hiện ít ở nhiều tài liệu thì đặc trưng hơn
- **Kết quả:** Từ vừa phổ biến trong tài liệu đó, vừa hiếm ở các tài liệu khác → điểm cao → **đặc trưng thực sự**

**Ví dụ:**
- "và": TF cao nhưng IDF ≈ 0 → TF-IDF thấp 
- "chất lượng tệ": TF trung bình, IDF cao (chỉ có trong review tiêu cực) → TF-IDF cao 

---

### 5: Phân tích Data Leakage

**Code lỗi của sinh viên X:**
```python
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.fit_transform(X_test)   # ← LỖI: fit_transform trên test!
```

**Vấn đề: Data Leakage (Rò rỉ dữ liệu)**

| Hậu quả | Giải thích |
|---------|-----------|
| Vocabulary không nhất quán | Train và test có từ điển riêng → chiều dữ liệu khác nhau |
| IDF bị nhiễm | IDF học từ test set → không phản ánh thực tế production |
| Đánh giá lạc quan giả | Mô hình "thấy" test data → metrics cao giả tạo |
| Không reproducible | Mỗi test set khác → kết quả khác nhau |

**Code đúng:**
```python
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)  # fit + transform
X_test_vec  = vectorizer.transform(X_test)        #  CHỈ transform!
```

**Quy tắc vàng:**
```
fit()           → CHỈ trên TRAIN set
transform()     → Áp dụng trên cả TRAIN và TEST
fit_transform() → Chỉ dùng cho TRAIN (= fit + transform)
```

**Analogy:** Đây giống như học sinh được xem đáp án trước khi thi (fit on test) — kết quả cao nhưng không phản ánh năng lực thật.

---

## Kết quả & Biểu đồ

| File | Mô tả |
|------|-------|
| `A1_label_distribution.png` | Phân phối nhãn (bar + pie chart) |
| `A2_text_length.png` | So sánh độ dài văn bản theo nhãn |
| `A4_bow_vs_tfidf.png` | Top từ theo BoW vs TF-IDF |

---

## Lý thuyết NLP

### Pipeline NLP chuẩn
```
Raw Text
   ↓
Lowercasing & Cleaning (HTML, URL, special chars)
   ↓
Tokenization
   ↓
Stopword Removal
   ↓
Stemming / Lemmatization
   ↓
Feature Extraction (BoW / TF-IDF / Word2Vec / BERT)
   ↓
Machine Learning Model
   ↓
Prediction
```
---

## 📄 License

MIT License — Xem file [LICENSE](LICENSE) để biết thêm.
