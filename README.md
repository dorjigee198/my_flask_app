# 🧠 Employee Attrition Prediction Dashboard

This is a Flask-based web application developed as part of a **Data Analytics project**. It predicts whether an employee is likely to leave (attrition) based on **12 input features** provided by the user. The prediction is powered by a trained machine learning model.

---

## 🚀 Features

- 🔢 Input form to collect 12 employee attributes (e.g., Age, Monthly Income, Job Role, etc.)
- 🤖 Backend prediction using a trained XGBoost or similar model
- 📈 Real-time prediction output (Attrition: Yes/No)
- 🌐 Simple and user-friendly Flask web dashboard
- 🗃️ Organized project structure with templates and static assets

---

## 📁 Project Structure
my_flask_app/
│
├── static/ # CSS and JS files
├── templates/ # HTML templates (index.html, result.html)
├── model/ # Pretrained ML model (e.g., xgb_attrition.plk)
├── app.py # Main Flask application
├── requirements.txt # Python dependencies
└── README.md # Project documentation

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Machine Learning**: Scikit-learn / XGBoost
- **Frontend**: HTML, CSS (Jinja2 templating)
- **Deployment Ready**: Can be hosted on platforms like Heroku, Render, or Replit

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/dorjigee198/my_flask_app.git
cd my_flask_app

python -m venv venv
venv\Scripts\activate        # Windows
# Or for macOS/Linux
# source venv/bin/activate
pip install -r requirements.txt
python app.py

