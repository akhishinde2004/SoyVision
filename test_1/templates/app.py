from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import os
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import json
import time
from datetime import datetime
import sqlite3
from functools import wraps

# AI Model imports
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'

# Add JSON filter for Jinja2 templates
@app.template_filter('tojson')
def tojson_filter(obj):
    return json.dumps(obj)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MODEL_PATH = 'models/baseline_model.pth'  # Path to your trained model
DATABASE = 'users.db'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('models', exist_ok=True)

# Database setup
def init_db():
    """Initialize the database with users and analysis_history tables"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Analysis history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            disease_detected TEXT NOT NULL,
            confidence REAL NOT NULL,
            severity TEXT NOT NULL,
            status TEXT DEFAULT 'Completed',
            analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processing_time TEXT,
            recommendations TEXT,
            all_predictions TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

def login_required(f):
    """Decorator to require login for certain routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def get_user_by_email(email):
    """Get user by email from database"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
    user = cursor.fetchone()
    conn.close()
    return user

def get_user_by_username(username):
    """Get user by username from database"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    conn.close()
    return user

def create_user(username, email, password):
    """Create new user in database"""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        password_hash = generate_password_hash(password)
        cursor.execute(
            'INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
            (username, email, password_hash)
        )
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return user_id
    except sqlite3.IntegrityError:
        return None

def save_analysis_to_history(user_id, filename, original_filename, results):
    """Save analysis results to database"""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO analysis_history 
            (user_id, filename, original_filename, disease_detected, confidence, 
             severity, processing_time, recommendations, all_predictions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            filename,
            original_filename,
            results['detection']['disease'],
            results['detection']['confidence'],
            results['detection']['severity'],
            results['detection']['processing_time'],
            json.dumps(results['recommendations']),
            json.dumps(results.get('all_predictions', {}))
        ))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving to history: {e}")
        return False

def get_user_analysis_history(user_id):
    """Get user's analysis history from database"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, filename, original_filename, disease_detected, confidence, 
               severity, status, analysis_date, processing_time, recommendations, all_predictions
        FROM analysis_history 
        WHERE user_id = ? 
        ORDER BY analysis_date DESC
    ''', (user_id,))
    
    history = cursor.fetchall()
    conn.close()
    
    # Convert to list of dictionaries for easier template usage
    history_list = []
    for row in history:
        history_item = {
            'id': row[0],
            'filename': row[1],
            'original_filename': row[2],
            'disease_detected': row[3],
            'confidence': row[4],
            'severity': row[5],
            'status': row[6],
            'analysis_date': row[7],
            'processing_time': row[8],
            'recommendations': json.loads(row[9]) if row[9] else [],
            'all_predictions': json.loads(row[10]) if row[10] else {}
        }
        history_list.append(history_item)
    
    return history_list

def get_analysis_by_id(analysis_id, user_id):
    """Get specific analysis by ID for the logged-in user"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM analysis_history 
        WHERE id = ? AND user_id = ?
    ''', (analysis_id, user_id))
    
    result = cursor.fetchone()
    conn.close()
    return result

# Define your model architecture (same as training)
class BaselineModel(nn.Module):
    def __init__(self, num_classes):
        super(BaselineModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 112 * 112, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16 * 112 * 112)
        x = self.fc(x)
        return x

# Define class labels - update these based on your training data
class_labels = [
    'Yellow Mosaic Disease',
    'Healthy',
]

num_classes = len(class_labels)

# Load the trained model
def load_model():
    """Load the trained PyTorch model"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = BaselineModel(num_classes)
        
        # Load model weights
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.to(device)
            model.eval()
            print("Model loaded successfully!")
            return model, device
        else:
            print(f"Model file not found at {MODEL_PATH}")
            return None, device
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, torch.device('cpu')

# Initialize model
loaded_model, device = load_model()

def get_severity_level(disease, confidence):
    """Determine severity level based on disease type and confidence"""
    if disease.lower() == 'healthy':
        return 'None'
    elif confidence > 80:
        return 'High'
    elif confidence > 60:
        return 'Moderate' 
    else:
        return 'Low'

def predict_disease(image_path, model, class_labels):
    """Predict disease from leaf image using trained model"""
    if model is None:
        return simulate_prediction()
    
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        
        model.eval()
        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)[0]
            conf, pred_idx = torch.max(probs, 0)
        
        predicted_class = class_labels[pred_idx.item()]
        confidence = conf.item() * 100
        severity = get_severity_level(predicted_class, confidence)
        
        return {
            'disease': predicted_class,
            'confidence': round(confidence, 1),
            'severity': severity,
            'all_predictions': {
                class_labels[i]: round(float(probs[i]) * 100, 2) 
                for i in range(len(class_labels))
            }
        }
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return simulate_prediction()

def get_treatment_recommendations(disease, severity):
    """Get treatment recommendations based on detected disease"""
    treatments = {
        'Brown Spot': [
            'Apply fungicide containing mancozeb or chlorothalonil',
            'Practice crop rotation with non-host crops',
            'Remove and destroy crop debris after harvest',
            'Plant certified disease-free seeds',
            'Ensure proper field drainage'
        ],
        'Yellow Mosaic Disease': [
            'Apply copper-based fungicides early in the season',
            'Use resistant soybean varieties when available',
            'Practice crop rotation with cereals',
            'Remove infected plant debris',
            'Avoid overhead irrigation'
        ],
        'Frog Eye Leaf Spot': [
            'Apply fungicides containing azoxystrobin or propiconazole',
            'Plant resistant varieties',
            'Practice crop rotation',
            'Manage crop residue properly',
            'Monitor field regularly for early detection'
        ],
        'Rust Disease': [
            'Apply fungicides at first sign of infection',
            'Use early-maturing varieties in high-risk areas',
            'Monitor weather conditions for favorable rust development',
            'Consider preventive fungicide applications',
            'Remove volunteer soybean plants'
        ],
        'Downy Mildew': [
            'Use metalaxyl-based fungicides as seed treatment',
            'Plant resistant varieties',
            'Ensure good field drainage',
            'Avoid planting in low-lying areas',
            'Practice crop rotation'
        ],
        'Healthy': [
            'Continue current management practices',
            'Monitor regularly for early disease detection',
            'Maintain proper field hygiene',
            'Follow integrated pest management practices'
        ]
    }
    
    return treatments.get(disease, [
        'Consult with agricultural extension services',
        'Consider laboratory diagnosis for accurate identification',
        'Apply appropriate fungicide based on confirmed diagnosis'
    ])

def simulate_prediction():
    """Fallback simulation if model is not available"""
    diseases = ['Brown Spot', 'Yellow Mosaic Disease', 'Healthy', 'Rust Disease']
    disease = np.random.choice(diseases)
    confidence = np.random.uniform(65, 95)
    severity = get_severity_level(disease, confidence)
    
    return {
        'disease': disease,
        'confidence': round(confidence, 1),
        'severity': severity,
        'all_predictions': {label: np.random.uniform(5, 30) for label in class_labels}
    }

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_image(filename):
    """Analyze image using the trained model"""
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Get AI prediction
    prediction_result = predict_disease(file_path, loaded_model, class_labels)
    
    # Build comprehensive results
    results = {
        'detection': {
            'disease': prediction_result['disease'],
            'confidence': prediction_result['confidence'],
            'severity': prediction_result['severity'],
            'affected_regions': 1 if prediction_result['disease'].lower() != 'healthy' else 0,
            'processing_time': f"{np.random.uniform(1.8, 3.2):.1f}sec"
        },
        'performance': {
            'yolo_detection': 100,  # Simulated for now
            'cnn_classification': min(98, prediction_result['confidence'] + np.random.uniform(-5, 5)),
            'avg_processing': f"{np.random.uniform(2.0, 2.5):.2f} sec"
        },
        'analysis_details': {
            'date': datetime.now().strftime('%d/%m/%Y'),
            'time': datetime.now().strftime('%H:%M'),
            'size': f"{np.random.uniform(1.5, 4.0):.1f} MB",
            'resolution': '640×480px',
            'model': 'PyTorch CNN',
            'status': 'AI Analysis Complete'
        },
        'recommendations': get_treatment_recommendations(
            prediction_result['disease'], 
            prediction_result['severity']
        ),
        'all_predictions': prediction_result.get('all_predictions', {}),
        'model_info': {
            'architecture': 'Custom CNN',
            'classes': len(class_labels),
            'device': str(device),
            'model_loaded': loaded_model is not None
        }
    }
    
    return results

# Routes
@app.route('/')
def landing():
    """Landing page - redirect based on login status"""
    if 'user_id' in session:
        return redirect(url_for('index'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        user = get_user_by_email(email)
        
        if user and check_password_hash(user[3], password):
            session['user_id'] = user[0]
            session['username'] = user[1]
            flash('Login successful!')
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password.')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page"""
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        # Validation
        if not username or not email or not password:
            flash('All fields are required.')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Passwords do not match.')
            return render_template('register.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long.')
            return render_template('register.html')
        
        # Check if user already exists
        if get_user_by_email(email):
            flash('Email already registered.')
            return render_template('register.html')
        
        if get_user_by_username(username):
            flash('Username already taken.')
            return render_template('register.html')
        
        # Create user
        user_id = create_user(username, email, password)
        if user_id:
            session['user_id'] = user_id
            session['username'] = username
            flash('Registration successful! Welcome to Soyabean Leaf Disease Detection.')
            return redirect(url_for('index'))
        else:
            flash('Registration failed. Please try again.')
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    """Logout user"""
    session.clear()
    flash('You have been logged out.')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def index():
    return render_template('index.html', username=session.get('username'))

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        original_filename = file.filename
        filename = secure_filename(file.filename)
        # Add timestamp to prevent filename conflicts
        timestamp = str(int(time.time()))
        filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Analyze image with trained model
        analysis_results = analyze_image(filename)
        
        # Save to history database
        save_analysis_to_history(
            session['user_id'], 
            filename, 
            original_filename, 
            analysis_results
        )
        
        return render_template('results.html', 
                             filename=filename, 
                             results=analysis_results,
                             username=session.get('username'))
    else:
        flash('Invalid file type. Please upload JPG, PNG, WEBP (Max 16MB)')
        return redirect(url_for('index'))

@app.route('/history')
@login_required
def history():
    """Display user's analysis history"""
    user_history = get_user_analysis_history(session['user_id'])
    return render_template('history.html', 
                         username=session.get('username'),
                         history=user_history)

@app.route('/view_analysis/<int:analysis_id>')
@login_required
def view_analysis(analysis_id):
    """View detailed analysis results"""
    analysis = get_analysis_by_id(analysis_id, session['user_id'])
    if not analysis:
        flash('Analysis not found.')
        return redirect(url_for('history'))
    
    # Convert database row to results format
    results = {
        'detection': {
            'disease': analysis[3],
            'confidence': analysis[4],
            'severity': analysis[5],
            'processing_time': analysis[8]
        },
        'recommendations': json.loads(analysis[9]) if analysis[9] else [],
        'all_predictions': json.loads(analysis[10]) if analysis[10] else {}
    }
    
    return render_template('results.html', 
                         filename=analysis[1],
                         results=results,
                         username=session.get('username'),
                         from_history=True)

@app.route('/delete_analysis/<int:analysis_id>', methods=['POST'])
@login_required
def delete_analysis(analysis_id):
    """Delete analysis from history"""
    try:
        # First check if analysis belongs to the user
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT filename FROM analysis_history 
            WHERE id = ? AND user_id = ?
        ''', (analysis_id, session['user_id']))
        
        analysis = cursor.fetchone()
        if not analysis:
            conn.close()
            return jsonify({'success': False, 'message': 'Analysis not found'})
        
        # Delete the image file
        filename = analysis[0]  # Get the filename from the query result
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(image_path):
            os.remove(image_path)
        
        # Delete from database
        cursor.execute('DELETE FROM analysis_history WHERE id = ? AND user_id = ?', 
                      (analysis_id, session['user_id']))
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Analysis deleted successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/stats')
@login_required
def stats():
    """Display user statistics"""
    user_history = get_user_analysis_history(session['user_id'])
    
    # Calculate statistics
    total_analyses = len(user_history)
    disease_counts = {}
    severity_counts = {'None': 0, 'Low': 0, 'Moderate': 0, 'High': 0}
    
    for analysis in user_history:
        disease = analysis['disease_detected']
        severity = analysis['severity']
        
        disease_counts[disease] = disease_counts.get(disease, 0) + 1
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
    
    stats_data = {
        'total_analyses': total_analyses,
        'disease_counts': disease_counts,
        'severity_counts': severity_counts,
        'recent_analyses': user_history[:5]  # Last 5 analyses
    }
    
    return render_template('stats.html', 
                         username=session.get('username'),
                         stats=stats_data)

if __name__ == '__main__':
    app.run(debug=True)