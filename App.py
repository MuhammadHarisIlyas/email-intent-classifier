# =========================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# =========================================
from flask import redirect
from user import users
from database import load_users, save_users


from flask import Flask, request, jsonify   # Flask framework for API
from flask_cors import CORS                 # Allow frontend to connect
import pickle                               # Load saved ML model
import re                                   # Text cleaning
import os

# Import stopwords for cleaning
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Import Gmail functions (from your gmail_reader.py)
from gmail_reader import authenticate_gmail, get_emails, classify_emails


# =========================================
# STEP 2: CREATE FLASK APP
# =========================================

app = Flask(__name__)

# Enable CORS so frontend (HTML/JS) can call backend
CORS(app)


# =========================================
# STEP 3: LOAD MODEL & VECTORIZER
# =========================================

# Get current directory (backend folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create paths to model files
model_path = os.path.join(BASE_DIR, "..", "model", "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "..", "model", "vectorizer.pkl")

# Load trained model
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Load TF-IDF vectorizer
with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)


# =========================================
# STEP 4: TEXT CLEANING FUNCTION
# =========================================

def clean_text(text):
    """
    Clean email text before prediction
    """

    text = text.lower()

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]

    return " ".join(words)


# =========================================
# STEP 5: HOME ROUTE (TESTING)
# =========================================

@app.route('/')
def home():
    return "Email Intent Classifier API is running!"


# =========================================
# STEP 6: PREDICTION ROUTE
# =========================================

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from frontend
        data = request.get_json()

        # Extract email text
        email_text = data.get("email")

        # Check if input is empty
        if not email_text:
            return jsonify({"error": "No input provided"}), 400

        # Step 1: Clean text
        cleaned_text = clean_text(email_text)

        # Step 2: Convert text into vector
        text_vector = vectorizer.transform([cleaned_text])

        # Step 3: Predict intent
        prediction = model.predict(text_vector)[0]

        # Step 4: Return result to frontend
        return jsonify({
            "prediction": prediction
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# =========================================
# STEP 7: GMAIL ROUTE (NEW FEATURE)
# =========================================

@app.route('/gmail', methods=['GET'])
def gmail_emails():
    """
    Fetch emails from Gmail and classify them
    """
    try:
        # Step 1: Connect to Gmail
        service = authenticate_gmail()

        # Step 2: Fetch emails
        emails = get_emails(service)

        # Step 3: Classify emails using ML model
        results = classify_emails(emails)

        # Step 4: Send results to frontend
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)})
    

# -----------------------------
# STEP 9: LOGIN ROUTE
# -----------------------------
from flask import redirect

@app.route('/login')
def login():
    try:
        authenticate_gmail()
        return redirect("http://127.0.0.1:5500/email_intent_project/frontend/dashboard.html")
    except Exception as e:
        return str(e)
    
#@app.route('/dashboard')
#def dashboard():
#    return "Dashboard working"


# -----------------------------
# STEP 10: USER LOGIN (NEW)
# -----------------------------
@app.route('/user-login', methods=['POST'])
def user_login():
    try:
        data = request.get_json()

        username = data.get("username")
        password = data.get("password")

        users = load_users()

        if username in users:
            if users[username] == password:
                return jsonify({
                    "status": "success",
                    "message": "Login successful"
                })
            else:
                return jsonify({
                    "status": "fail",
                    "message": "Incorrect password"
                })
        else:
            return jsonify({
                "status": "fail",
                "message": "Please sign up first"
            })

    except Exception as e:
        return jsonify({"error": str(e)})
    # -----------------------------
# STEP 10: USER SIGNUP
# -----------------------------
@app.route('/signup', methods=['POST'])
def signup():
    try:
        data = request.get_json()

        username = data.get("username")
        password = data.get("password")

        users = load_users()

        if username in users:
            return jsonify({"status": "fail", "message": "User already exists"})

        users[username] = password
        save_users(users)

        return jsonify({"status": "success", "message": "Signup successful"})

    except Exception as e:
        return jsonify({"error": str(e)})


# =========================================
# STEP 8: RUN FLASK APP
# =========================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
    #app.run(debug=True, port=5001, use_reloader=False)

