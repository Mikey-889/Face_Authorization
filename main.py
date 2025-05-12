from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import base64
import cv2
import os
import pickle
import psycopg2
from psycopg2 import Binary

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Database connection
def get_db_connection():
    # Get connection string from environment variable
    db_url = os.environ.get('DATABASE_URL')
    
    if not db_url:
        raise ValueError("No DATABASE_URL environment variable set")
        
    return psycopg2.connect(db_url)

# Initialize database
def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create users table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        username VARCHAR(255) PRIMARY KEY,
        face_encoding BYTEA,
        timestamp BIGINT
    )
    ''')
    
    conn.commit()
    conn.close()
    print("Database initialized successfully")

# Convert base64 string to image
def base64_to_image(base64_string):
    # Convert base64 string to numpy array
    img_data = base64.b64decode(base64_string.split(',')[1])
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

# Simple face detection using Haar Cascades
def detect_face(image):
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load OpenCV's face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    if len(faces) == 0:
        return None, "No face detected"
        
    if len(faces) > 1:
        return None, "Multiple faces detected"
    
    # Extract the single face
    x, y, w, h = faces[0]
    face_roi = gray[y:y+h, x:x+w]
    
    # Check face size
    face_area = w * h
    image_area = gray.shape[0] * gray.shape[1]
    face_percentage = face_area / image_area
    
    if face_percentage < 0.05:
        return None, "Face too small, please move closer"
    
    if face_percentage > 0.65:
        return None, "Face too close to camera, please move back"
    
    # Resize to a standard size
    face_roi = cv2.resize(face_roi, (128, 128))
    
    # Extract features (flatten the resized face)
    features = face_roi.flatten()
    
    return features, "Face detected successfully"

@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    face_image_base64 = data.get('face_image')

    if not username or not face_image_base64:
        return jsonify({'success': False, 'message': 'Missing username or face image'}), 400

    try:
        # Convert base64 image to numpy array
        face_image = base64_to_image(face_image_base64)
        
        # Extract face features
        features, message = detect_face(face_image)
        
        if features is None:
            return jsonify({'success': False, 'message': message}), 400
            
        # Store in PostgreSQL database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if username already exists
        cursor.execute("SELECT username FROM users WHERE username = %s", (username,))
        if cursor.fetchone():
            conn.close()
            return jsonify({'success': False, 'message': 'Username already exists'}), 400
            
        # Insert new user
        timestamp = data.get('timestamp', 0)
        
        # Serialize face features
        features_blob = Binary(pickle.dumps(features))
        
        cursor.execute(
            "INSERT INTO users (username, face_encoding, timestamp) VALUES (%s, %s, %s)",
            (username, features_blob, timestamp)
        )
        conn.commit()
        conn.close()

        return jsonify({'success': True, 'message': 'Registration successful'})

    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    face_image_base64 = data.get('face_image')

    if not face_image_base64:
        return jsonify({'success': False, 'message': 'Missing face image'}), 400

    try:
        # Convert base64 image to numpy array
        face_image = base64_to_image(face_image_base64)
        
        # Extract face features
        features, message = detect_face(face_image)
        
        if features is None:
            return jsonify({'success': False, 'message': message}), 400
            
        # Get all users from database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT username, face_encoding FROM users")
        users = cursor.fetchall()
        conn.close()
        
        if not users:
            return jsonify({'success': False, 'message': 'No users registered yet'}), 401
            
        # Compare with all registered users
        match_results = []
        
        for username, stored_encoding_blob in users:
            # Unpickle the stored features
            stored_features = pickle.loads(bytes(stored_encoding_blob))
            
            # Calculate similarity using correlation
            correlation = np.corrcoef(features, stored_features)[0, 1]
            
            match_results.append({
                'username': username,
                'similarity': correlation
            })
            
        # Sort by highest similarity
        match_results.sort(key=lambda x: x['similarity'], reverse=True)
        best_match = match_results[0] if match_results else None
        
        # Authentication logic
        similarity_threshold = 0.8  # Adjust based on testing
        
        if best_match and best_match['similarity'] > similarity_threshold:
            # Check if the difference between best match and second best is significant
            if len(match_results) > 1:
                second_best = match_results[1]
                similarity_gap = best_match['similarity'] - second_best['similarity']
                
                if similarity_gap < 0.05:  # Require at least 0.05 gap for confidence
                    return jsonify({
                        'success': False, 
                        'message': 'Face recognition ambiguous, please try again'
                    }), 401
                    
            # Log successful login
            print(f"Login successful for {best_match['username']} with similarity {best_match['similarity']:.2f}")
            
            return jsonify({
                'success': True,
                'message': 'Authentication successful',
                'username': best_match['username'],
                'confidence': float(best_match['similarity'])
            })
        else:
            # Authentication failed
            if best_match:
                similarity = best_match['similarity']
                print(f"Login failed. Best match was {best_match['username']} with similarity {similarity:.2f}")
                
                if similarity > 0.6:  # Close but not quite
                    return jsonify({
                        'success': False,
                        'message': 'Face similar but not recognized with confidence. Please try again with better lighting.'
                    }), 401
                else:
                    return jsonify({
                        'success': False,
                        'message': 'Face not recognized'
                    }), 401
            else:
                return jsonify({
                    'success': False,
                    'message': 'No registered faces to compare with'
                }), 401
                
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500

# Serve the frontend static files
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_static(path):
    if path == "" or path == "index.html":
        return send_from_directory('.', 'index.html')
    return send_from_directory('.', path)

# Initialize database on startup
@app.before_first_request
def initialize():
    init_db()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))