# SETUP.md — ForensicsDetective Assignment 2

## System Environment

| Item | Version |
|------|---------|
| Operating System | macOS 26.3.1 |
| Python | 3.13.5 |
| Shell | zsh |

---

## 1. Fork and Clone the Repository

Fork the original repository on GitHub, then clone your fork locally:

```bash
git clone git@github.com:Zeye430/ForensicsDetective.git
cd ForensicsDetective
```

Verify the remote is set correctly:

```bash
git remote -v
# origin  git@github.com:Zeye430/ForensicsDetective.git (fetch)
# origin  git@github.com:Zeye430/ForensicsDetective.git (push)
```

---

## 2. Create and Activate a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Key libraries installed:

| Library | Version | Purpose |
|---------|---------|---------|
| scikit-learn | 1.8.0 | SVM, SGD, and additional classifiers |
| xgboost | 3.2.0 | Gradient boosting classifier |
| lightgbm | 4.6.0 | Gradient boosting classifier |
| opencv-python | 4.13.0.92 | Image augmentation |
| pillow | 12.1.1 | Image loading and processing |
| numpy | 2.4.3 | Numerical operations |
| pandas | 3.0.1 | Results and metrics management |
| matplotlib | 3.10.8 | Visualization |
| seaborn | 0.13.2 | Visualization |

---

## 4. Verify the Dataset

The PNG images are pre-generated and located in:

```
word_pdfs_png/          # 398 Word-generated PDF images
google_docs_pdfs_png/   # 396 Google Docs-generated PDF images
python_pdfs_png/        # 100 Python/ReportLab-generated PDF images
```

Verify the counts:

```bash
ls word_pdfs_png/ | grep ".png" | wc -l         # expected: 398
ls google_docs_pdfs_png/ | grep ".png" | wc -l  # expected: 396
ls python_pdfs_png/ | grep ".png" | wc -l       # expected: 100
```


## Collaborators

The following GitHub users have been added as collaborators on this fork:

- `delveccj` (original author / instructor)
- `AnushkaTi` (TA)
