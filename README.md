# Real-Time Cyberbullying Detection Platform

A machine learning-powered real-time chat application that automatically detects and prevents cyberbullying in **English and Hinglish languages** through intelligent content filtering. The system employs advanced NLP techniques and a Random Forest classifier optimized for **96.37% recall** to minimize false negatives in harmful content detection.


## 🎯 Project Overview

This platform addresses the critical issue of online cyberbullying by providing real-time detection and intervention. Built on rigorous primary research with **300+ respondents**, the system achieves:

- **97.1% accuracy** on **18,000+ multilingual samples**
- **96.37% recall** optimized to minimize false negatives
- Real-time chat moderation with instant feedback
- Support for **English and Hinglish** languages

### Research-Backed Statistics
According to our field survey of 300+ respondents:
- **36.5%** of people have experienced cyberbullying in their lifetime
- **60%** of teenagers have experienced some form of cyberbullying
- **61%** of users agree that women are more subject to cyberbullying than men
- **Body shaming** and **sexual orientation** are the top concerns for cyberbullying

## ✨ Key Features

### Core Functionality
- **Real-Time Chat Application**: Fully functional multi-user chat using Flask and Socket.IO
- **Automated Content Filtering**: ML-powered detection of harmful messages before they reach recipients
- **Bilingual Support**: Trained on 18,000+ samples in English and Hinglish
- **High Recall Optimization**: 96.37% recall ensures minimal missed harmful content
- **Instant User Feedback**: Senders receive warnings; receivers are protected from harmful content

### Technical Highlights
- **Research-Validated**: Feature importance and intervention thresholds validated through primary research
- **Multiple Data Sources**: Twitter, WhatsApp, and YouTube comments
- **RESTful API**: Postman-tested endpoints for easy integration
- **Service Wrapper Architecture**: Flask-based microservice for scalable deployment

## 🛠️ Technologies Used

- **Python 3.8+**
- **Flask**: Web framework and backend
- **Socket.IO**: Real-time bidirectional communication
- **Pandas**: Data manipulation and preprocessing
- **Scikit-learn**: Machine learning (Random Forest, SVM, Logistic Regression)
- **NLTK**: Natural language processing
- **Tkinter**: GUI for chat application
- **Postman**: API testing and documentation

## 📊 Model Performance

### Dataset Composition
- **Total Samples**: 18,000+ (15,307 English + 3,000 Hinglish)
- **Sources**: Twitter, WhatsApp, YouTube comments
- **Distribution**: 64% Bullying, 36% Non-Bullying
- **Languages**: English and Hinglish (Hindi + English code-mixed)
- **Features**: Real-world toxic phrases and contextual patterns

### Performance Metrics

| Algorithm | Accuracy | F1-Score | Recall | Precision |
|-----------|----------|----------|--------|-----------|
| **Random Forest** | **97.1%** | **97.2%** | 95.8% | 98.6% |
| **Linear SVC** | **96.7%** | **96.8%** | **96.37%** | 97.2% |
| Logistic Regression | 94.0% | 94.9% | 93.5% | 96.3% |
| Decision Tree | 96.2% | 96.8% | 95.6% | 98.0% |
| Bagging Classifier | 95.6% | 96.5% | 94.8% | 98.2% |

**Why High Recall?**  
In cyberbullying detection, it's better to flag a benign message for review than to miss actual harmful content that could lead to serious psychological consequences. Our 96.37% recall ensures that harmful messages are caught with minimal false negatives.

### Feature Extraction Comparison

| Method | Best Algorithm | Accuracy | Training Time |
|--------|---------------|----------|---------------|
| **TF-IDF** (Selected) | Random Forest | 97.1% | Moderate |
| Count Vectorizer | Random Forest | 96.5% | Moderate |
| TF-IDF | Linear SVC | 96.7% | **Fast** |

**Winner:** TF-IDF with Linear SVC provides the best balance of accuracy (96.7%), recall (96.37%), and speed for production deployment.

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/NihalMishra17/cyberbullying.git
cd cyberbullying
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data**
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## 💻 Usage

### Running the Chat Application

1. **Start the Flask server**
```bash
python app.py
```

2. **Launch the chat interface**
```bash
python chat_gui.py
```

3. **Use the application**
   - Enter your username
   - Create or join a room using Room ID
   - Start chatting with real-time moderation

### API Endpoints

#### POST `/predict`
Predict if a message contains cyberbullying.

**Request:**
```json
{
  "message": "Your message here"
}
```

**Response:**
```json
{
  "prediction": "bullying" | "non-bullying",
  "confidence": 0.95,
  "label": 1 | 0
}
```

### Testing with Postman

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "test message"}'
```

## 📁 Project Structure

```
cyberbullying/
│
├── app.py                      # Flask API server
├── chat_gui.py                 # Tkinter chat interface
├── model_training.py           # ML model training script
├── preprocessing.py            # Text preprocessing pipeline
├── service_wrapper.py          # Flask service wrapper
├── requirements.txt            # Python dependencies
│
├── data/
│   ├── english_dataset.csv     # English training data (15,307 samples)
│   ├── hinglish_dataset.csv    # Hinglish training data (3,000 samples)
│   └── stopwords_hinglish.txt  # Custom Hinglish stopwords
│
├── models/
│   ├── rf_classifier.pkl       # Trained Random Forest model
│   ├── tfidf_vectorizer.pkl    # TF-IDF vectorizer
│   └── linear_svc.pkl          # Linear SVC model
│
├── static/
│   └── css/
│       └── style.css           # Chat interface styles
│
├── templates/
│   ├── index.html              # Landing page
│   └── chat.html               # Chat interface
│
└── tests/
    ├── test_model.py           # Model unit tests
    └── test_api.py             # API endpoint tests
```

## 🔬 Research Methodology

### 1. Data Collection & Extraction
- **Twitter**: Real-time tweets using Twitter API
- **WhatsApp**: Extracted chat logs
- **YouTube**: Comment scraping
- **Kaggle**: Additional English datasets

### 2. Data Cleaning Pipeline
```
Raw Data → Remove URLs/Hashtags → Remove Punctuation → 
Lowercase Conversion → Remove Stopwords → Clean Data
```

Preprocessing removes:
- Hexadecimal patterns
- Numeric data
- URLs and user mentions
- Hashtags and retweet symbols
- Special characters and punctuation
- Words with less than 3 letters

### 3. NLP Preprocessing

**Tokenization** → **Lemmatization** → **Vectorization**

- **Tokenization**: Break text into individual tokens (words)
- **Lemmatization**: Reduce words to root form (e.g., "running" → "run")
- **Vectorization**: Convert text to numerical vectors using TF-IDF

### 4. Feature Engineering

**TF-IDF (Term Frequency-Inverse Document Frequency)**
- Measures word importance in document relative to corpus
- Formula: `TF-IDF = TF × IDF`
- Higher scores indicate more relevant/distinctive words
- Better than Count Vectorizer for cyberbullying detection

### 5. Machine Learning Pipeline

```
Training Data (80%) + Testing Data (20%)
           ↓
    Feature Extraction (TF-IDF)
           ↓
    Train Multiple Algorithms
           ↓
    Cross-Validation & Hyperparameter Tuning
           ↓
    Model Evaluation (Accuracy, Recall, F1-Score)
           ↓
    Select Best Model → Pickle File → Production
```

### 6. Model Selection Criteria
- **Primary**: Recall (minimize false negatives)
- **Secondary**: Accuracy and F1-Score
- **Tertiary**: Training/Prediction speed

**Winner**: Linear SVC with TF-IDF (96.7% accuracy, 96.37% recall, fastest speed)

## 🏗️ System Architecture

### Service Wrapper Flow

```
User Posts Message
       ↓
Flask Service Wrapper
       ↓
Load ML Model (Pickle)
       ↓
Preprocessing Pipeline
       ↓
Feature Extraction (TF-IDF)
       ↓
Model Prediction (0 or 1)
       ↓
   ├─ If 0 (Non-Bullying) → Display message
   └─ If 1 (Bullying) → Block message + Warning
```

### Chat Application Flow

#### Non-Bullying Flow:
1. User enters message in chat
2. Service predicts: Non-Bullying (0)
3. Message displayed to all users in room

#### Bullying Flow:
1. User enters harmful message
2. Service predicts: Bullying (1)
3. **Sender**: Receives warning - "Stop bullying people and behave decently"
4. **Receiver**: Notified - "A bullying message has been detected and hidden"
5. Message is blocked and not displayed

## 📈 Key Research Findings

### Primary Research (300+ Respondents)

**Top Cyberbullying Factors:**
1. Body shaming (highest concern)
2. Sexual orientation
3. Gender discrimination (61% agree women are targeted more)
4. Religion and social status
5. Xenophobia and hostile activity

**Most Common Bullying Indicators:**
- Threatening language
- Personal attacks
- Hate speech
- Offensive slurs
- Aggressive tone with excessive punctuation/caps

### Model Insights

**Best Performing Combinations:**
1. **TF-IDF + Linear SVC**: 96.7% accuracy, fastest (production choice)
2. **TF-IDF + Random Forest**: 97.1% accuracy, slower
3. **Count Vectorizer + Random Forest**: 96.5% accuracy

**Why TF-IDF over Count Vectorizer:**
- Considers word importance, not just frequency
- Removes less important tokens
- Reduces dimensionality
- More efficient training
- Better generalization

## 🎓 Key Learnings & Contributions

This project demonstrates:
- End-to-end ML pipeline from data collection to deployment
- Handling imbalanced datasets (64% bullying vs 36% non-bullying)
- Multilingual NLP (English + Hinglish code-mixing)
- Real-time system integration with Socket.IO
- Research-validated feature engineering
- Production-ready model deployment with Flask

## 🔮 Future Enhancements

- [ ] **Deep Learning**: Implement LSTM/BERT for improved accuracy
- [ ] **More Languages**: Add support for Gujarati, Marathi, Tamil, Telugu, Kannada
- [ ] **Multimodal Detection**: Extend to images and videos
- [ ] **User Dashboard**: Analytics for administrators
- [ ] **Sentiment Analysis**: Provide context beyond binary classification
- [ ] **Mobile App**: iOS and Android native applications
- [ ] **Cloud Deployment**: AWS/Azure hosting with auto-scaling
- [ ] **Real-time Analytics**: Monitor bullying trends and patterns

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

