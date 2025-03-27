# 📢 Customer Review Classification

## 🌟 Project Overview
This project aims to classify customer reviews as **positive** or **negative** using **machine learning models**. It demonstrates text classification through the following steps:

✅ **Text Preprocessing**  
✅ **Feature Extraction using TF-IDF**  
✅ **Model Training** with Logistic Regression, Naive Bayes, and SVM  
✅ **Evaluation of Models**  
✅ **Saving and Loading Models**  
✅ **Predicting Sentiments of New Reviews**  

---

## 📂 Dataset Information
- **Dataset Used:** IMDB Reviews Dataset 🎬
- **Size:** 50,000 movie reviews
- **Labels:** Positive & Negative
- **Setup:** Download and place the dataset in the `data/` directory.

---

## ⚙️ Installation and Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/ma5Codes/customer_review_classification.git
cd customer_review_classification
```

### 2️⃣ Create Virtual Environment
**For Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

**For Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Required Packages
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Main Script
```bash
python main.py
```

---

## 🛠 Requirements
This project requires the following Python libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `nltk`
- `pickle-mixin`

Install them using:
```bash
pip install pandas numpy scikit-learn nltk pickle-mixin
```

---

## 🚀 How to Run the Project

### 1️⃣ Run Main Script
To classify customer reviews, run:
```bash
python main.py
```

### 2️⃣ Jupyter Notebook for Analysis
If you prefer analyzing the data in a Jupyter Notebook:
```bash
jupyter notebook
```
Open **customer_review_classification.ipynb** from the `notebooks/` folder.

---

## 🎯 Features
🔹 **Interactive UI** for sentiment classification (React + FastAPI)  
🔹 **Supports Multiple Models** (Logistic Regression, Naive Bayes, SVM)  
🔹 **Urgency Detection** for prioritizing critical reviews ⚡  
🔹 **Well-Structured Codebase** for easy modifications & improvements  

---

## 📌 Contributing
Want to contribute? 🎉 Feel free to submit a **pull request** or **report issues**!

---

## 🎯 Future Improvements
🚀 Improve UI design  
🤖 Integrate Deep Learning models (e.g., BERT)  
📊 Add more visualization for sentiment trends  

---

## 💡 Credits
Developed with ❤️ by **ma5Codes**

---

## 📜 License
This project is licensed under the **MIT License**. Feel free to use and modify! 😊
