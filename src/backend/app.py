# ============================================================ 

# AIVision - Artificial Intelligence Platform 

# ============================================================ 

from sklearn.neural_network import MLPClassifier
from flask import Flask, jsonify, request, render_template, send_file
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

from datetime import datetime, timedelta
from werkzeug.utils import secure_filename

import os
import io
import json
import random
import math
import requests

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    IsolationForest
)

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error


# ============================================================
# APP SETUP
# ============================================================

# Get the path of the src folder.
#
# app.py is located at:
# AIVision/src/backend/app.py
#
# dirname(__file__)          -> AIVision/src/backend
# dirname(dirname(__file__)) -> AIVision/src
#@app.route('/api/data/builtin/<dataset_name>', methods=['POST'])
app = Flask(__name__)
@app.route('/api/data/builtin/<dataset_name>', methods=['POST'])
def load_builtin_dataset(dataset_name):
    try:
        # Remove existing data
        AIData.query.delete()

        datasets = {
            'student': [
                (5, 80, 8, 65, 'PASS'),
                (8, 90, 10, 82, 'PASS'),
                (3, 60, 5, 45, 'FAIL'),
                (7, 85, 9, 76, 'PASS'),
                (2, 50, 4, 35, 'FAIL'),
                (9, 95, 10, 91, 'PASS'),
                (4, 70, 6, 55, 'FAIL'),
                (6, 88, 9, 73, 'PASS'),
                (1, 45, 3, 28, 'FAIL'),
                (10, 98, 10, 95, 'PASS')
            ],

            'customers': [
                (22, 25000, 35, 42, 'Low'),
                (25, 32000, 60, 55, 'Medium'),
                (30, 45000, 75, 70, 'High'),
                (35, 52000, 80, 82, 'High'),
                (40, 60000, 45, 65, 'Medium'),
                (28, 38000, 90, 78, 'High'),
                (45, 70000, 30, 58, 'Medium'),
                (50, 85000, 25, 45, 'Low'),
                (27, 29000, 40, 50, 'Medium'),
                (33, 48000, 70, 75, 'High')
            ],

            'houses': [
                (2, 850, 6, 180000, 'Affordable'),
                (3, 1200, 7, 250000, 'Affordable'),
                (4, 1800, 8, 380000, 'Mid-range'),
                (3, 1500, 7, 310000, 'Mid-range'),
                (5, 2500, 9, 550000, 'Expensive'),
                (4, 2200, 8, 470000, 'Expensive'),
                (2, 900, 5, 160000, 'Affordable'),
                (5, 3000, 10, 700000, 'Expensive'),
                (3, 1400, 7, 290000, 'Mid-range'),
                (4, 2000, 9, 450000, 'Expensive')
            ],

            'iris': [
                (5.1, 3.5, 1.4, 0, 'Setosa'),
                (4.9, 3.0, 1.4, 0, 'Setosa'),
                (5.4, 3.9, 1.7, 0, 'Setosa'),
                (6.4, 3.2, 4.5, 1, 'Versicolor'),
                (6.9, 3.1, 4.9, 1, 'Versicolor'),
                (5.8, 2.7, 4.1, 1, 'Versicolor'),
                (6.3, 3.3, 6.0, 2, 'Virginica'),
                (6.5, 3.0, 5.8, 2, 'Virginica'),
                (7.1, 3.0, 5.9, 2, 'Virginica'),
                (6.7, 3.1, 5.6, 2, 'Virginica')
            ]
        }

        if dataset_name not in datasets:
            return jsonify({
                'success': False,
                'error': 'Dataset not found.'
            }), 404

        for row in datasets[dataset_name]:

            record = AIData(
                feature1=float(row[0]),
                feature2=float(row[1]),
                feature3=float(row[2]),
                target=float(row[3]),
                category=row[4],
                source=f'Built-in: {dataset_name}'
            )

            db.session.add(record)

        db.session.commit()

        return jsonify({
            'success': True,
            'dataset': dataset_name,
            'records_loaded': len(datasets[dataset_name]),
            'message': f'{dataset_name.title()} dataset loaded successfully.'
        })

    except Exception as e:
        db.session.rollback()

        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
 BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


# ============================================================
# FOLDERS
# ============================================================

# Folder where uploaded files will be stored
UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    'data',
    'uploads'
)

# Create the upload folder if it does not already exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================================================
# CREATE FLASK APPLICATION
# ============================================================

# IMPORTANT:
# app.py is inside:
#
# AIVision/src/backend/
#
# Therefore:
#
# template_folder='templates'
#
# tells Flask to look here:
#
# AIVision/src/backend/templates/
#
# This is where analytics.html is located.

app = Flask(
    __name__,
    template_folder='templates',

    # Your frontend files are located in:
    # AIVision/src/frontend/
    static_folder=os.path.join(BASE_DIR, 'frontend'),

    # Allows Flask to serve static files from /css/... etc.
    static_url_path=''
)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder='templates',
    static_folder='static',
    static_url_path='/static'
)


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app.config['SECRET_KEY'] = 'aivision-2026'

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///aivision.db'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024


# ============================================================
# ENABLE CORS
# ============================================================

CORS(app)


# ============================================================
# DATABASE
# ============================================================

db = SQLAlchemy(app)


# ============================================================
# PAGE ROUTES
# ============================================================

# Home page
#
# When you open:
# http://localhost:5000/
#
# Flask will render:
# src/backend/templates/analytics.html

#@app.route('/')
#def index():
 #   return render_template('index.html')


# Analytics page
#
# When you open:
# http://localhost:5000/analytics
#
# Flask will render the same page.
@app.route('/api/data/samples', methods=['GET'])
def get_samples():
    try:
        samples = AIData.query.order_by(AIData.id.desc()).limit(20).all()

        return jsonify({
            'success': True,
            'samples': [
                {
                    'id': item.id,
                    'timestamp': item.timestamp.strftime('%Y-%m-%d %H:%M:%S')
                    if item.timestamp else None,
                    'feature1': item.feature1,
                    'feature2': item.feature2,
                    'feature3': item.feature3,
                    'target': item.target,
                    'category': item.category,
                    'source': item.source
                }
                for item in samples
            ]
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

    
@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

@app.route('/')
def index():
    return render_template('index.html')


# Dashboard page
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


# Machine Learning page
@app.route('/ml')
def ml_page():
    return render_template('ml.html')

@app.route('/api/deep-learning/train', methods=['POST'])
def train_deep_learning():
    try:
        # Get all data from the database
        data = AIData.query.all()

        # Make sure we have enough data
        if len(data) < 10:
            return jsonify({
                'success': False,
                'error': 'Not enough data. Please generate data from the Dashboard first.'
            }), 400

        # Use the three features as inputs
        X = np.array([
            [item.feature1, item.feature2, item.feature3]
            for item in data
        ])

        # Get the categories
        categories = [item.category for item in data]

        # Convert categories into numbers
        unique_categories = list(set(categories))

        category_map = {
            category: index
            for index, category in enumerate(unique_categories)
        }

        y = np.array([
            category_map[category]
            for category in categories
        ])

        # Split the data into training and testing data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # Scale the input features
        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Create the neural network
        model = MLPClassifier(
            hidden_layer_sizes=(8,),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42
        )

        # Train the neural network
        model.fit(X_train, y_train)

        # Test the trained model
        predictions = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, predictions)

        # Get the final training loss
        loss = model.loss_

        return jsonify({
            'success': True,
            'model': 'Neural Network',
            'architecture': '3 → 8 → 1',
            'input_neurons': 3,
            'hidden_neurons': 8,
            'output_neurons': 1,
            'training_samples': len(X_train),
            'testing_samples': len(X_test),
            'accuracy': round(float(accuracy * 100), 2),
            'loss': round(float(loss), 4),
            'message': 'Neural network trained successfully.'
        })

    except Exception as e:

        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
# Deep Learning page
@app.route('/deep-learning')
def deep_learning():
    return render_template('deep_learning.html')


# ============================================================
# MATHEMATICS (PM-01)
# ============================================================


class AIMath:

    @staticmethod
    def mean(values):
        return sum(values) / len(values) if values else 0


    @staticmethod
    def std_dev(values):

        if len(values) < 2:
            return 0

        mean = AIMath.mean(values)

        variance = sum(
            (x - mean) ** 2
            for x in values
        ) / len(values)

        return math.sqrt(variance)


    @staticmethod
    def sigmoid(x):

        return (
            1 / (1 + math.exp(-x))
            if x > -500
            else 0
        )


    @staticmethod
    def relu(x):

        return max(0, x)


    @staticmethod
    def softmax(values):

        exp_vals = np.exp(
            values - np.max(values)
        )

        return (
            exp_vals / exp_vals.sum()
        ).tolist()


    @staticmethod
    def bayes(prior, likelihood, evidence):

        return (
            (likelihood * prior) / evidence
            if evidence != 0
            else 0
        )


    @staticmethod
    def correlation(x, y):

        if len(x) != len(y) or len(x) < 2:
            return 0

        mx = AIMath.mean(x)
        my = AIMath.mean(y)

        cov = sum(
            (x[i] - mx) * (y[i] - my)
            for i in range(len(x))
        ) / (len(x) - 1)

        sx = AIMath.std_dev(x)
        sy = AIMath.std_dev(y)

        return (
            cov / (sx * sy)
            if sx != 0 and sy != 0
            else 0
        )


# ============================================================
# MACHINE LEARNING (PM-07)
# ============================================================


# ============================================================
# MACHINE LEARNING APIs
# ============================================================

# ------------------------------------------------------------
# 1. CLASSIFICATION
# ------------------------------------------------------------
@app.route('/api/ml/classify', methods=['POST'])
def classify():
    try:
        # Get all records from the database
        data = AIData.query.all()

        # Check that enough data exists
        if len(data) < 5:
            return jsonify({
                'success': False,
                'error': 'Not enough data. Please generate data from the Dashboard first.'
            }), 400

        # Create input features
        X = np.array([
            [item.feature1, item.feature2, item.feature3]
            for item in data
        ])

        # Create classification target
        # Convert categories into numbers
        categories = [item.category for item in data]

        unique_categories = list(set(categories))

        category_map = {
            category: index
            for index, category in enumerate(unique_categories)
        }

        y = np.array([
            category_map[category]
            for category in categories
        ])

        # Split data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # Create Random Forest classifier
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, predictions)

        return jsonify({
            'success': True,
            'model': 'Random Forest Classification',
            'training_samples': len(X_train),
            'testing_samples': len(X_test),
            'accuracy': round(float(accuracy * 100), 2),
            'classes': unique_categories,
            'message': 'Classification model trained successfully.'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ------------------------------------------------------------
# 2. REGRESSION
# ------------------------------------------------------------
@app.route('/api/ml/regression', methods=['POST'])
def regression():
    try:
        # Get all records from the database
        data = AIData.query.all()

        # Check that enough data exists
        if len(data) < 5:
            return jsonify({
                'success': False,
                'error': 'Not enough data. Please generate data from the Dashboard first.'
            }), 400

        # Create input features
        X = np.array([
            [item.feature1, item.feature2, item.feature3]
            for item in data
        ])

        # Create target values
        y = np.array([
            item.target
            for item in data
        ])

        # Split data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # Create Random Forest regression model
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Calculate Mean Squared Error
        mse = mean_squared_error(
            y_test,
            predictions
        )

        # Calculate Root Mean Squared Error
        rmse = np.sqrt(mse)

        return jsonify({
            'success': True,
            'model': 'Random Forest Regression',
            'training_samples': len(X_train),
            'testing_samples': len(X_test),
            'mean_squared_error': round(float(mse), 4),
            'root_mean_squared_error': round(float(rmse), 4),
            'message': 'Regression model trained successfully.'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ------------------------------------------------------------
# 3. ANOMALY DETECTION
# ------------------------------------------------------------
@app.route('/api/ml/anomaly', methods=['POST'])
def anomaly_detection():
    try:
        # Get all records from the database
        data = AIData.query.all()

        # Check that enough data exists
        if len(data) < 5:
            return jsonify({
                'success': False,
                'error': 'Not enough data. Please generate data from the Dashboard first.'
            }), 400

        # Create feature matrix
        X = np.array([
            [item.feature1, item.feature2, item.feature3]
            for item in data
        ])

        # Create Isolation Forest model
        model = IsolationForest(
            contamination=0.05,
            random_state=42
        )

        # Detect anomalies
        predictions = model.fit_predict(X)

        # Isolation Forest returns:
        #  1  = normal
        # -1  = anomaly

        anomalies = int(np.sum(predictions == -1))
        normal = int(np.sum(predictions == 1))

        return jsonify({
            'success': True,
            'model': 'Isolation Forest',
            'total_samples': len(X),
            'normal_samples': normal,
            'anomalies': anomalies,
            'anomaly_percentage': round(
                (anomalies / len(X)) * 100,
                2
            ),
            'message': 'Anomaly detection completed successfully.'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ------------------------------------------------------------
# 4. CLUSTERING
# ------------------------------------------------------------
@app.route('/api/ml/clustering', methods=['POST'])
def clustering():
    try:
        # Get all records from the database
        data = AIData.query.all()

        # Check that enough data exists
        if len(data) < 5:
            return jsonify({
                'success': False,
                'error': 'Not enough data. Please generate data from the Dashboard first.'
            }), 400

        # Create feature matrix
        X = np.array([
            [item.feature1, item.feature2, item.feature3]
            for item in data
        ])

        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Create K-Means clustering model
        model = KMeans(
            n_clusters=3,
            random_state=42,
            n_init=10
        )

        # Assign each record to a cluster
        clusters = model.fit_predict(X_scaled)

        # Count records in each cluster
        cluster_counts = {}

        for cluster in clusters:
            cluster_number = int(cluster)

            if cluster_number not in cluster_counts:
                cluster_counts[cluster_number] = 0

            cluster_counts[cluster_number] += 1

        return jsonify({
            'success': True,
            'model': 'K-Means Clustering',
            'total_samples': len(X),
            'number_of_clusters': 3,
            'cluster_counts': cluster_counts,
            'message': 'Clustering completed successfully.'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
@app.route('/api/data/upload/csv', methods=['POST'])
def upload_csv():
    try:

        # Check that a file was uploaded
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No CSV file was uploaded.'
            }), 400

        file = request.files['file']

        # Check that a file was selected
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file was selected.'
            }), 400

        # Check file extension
        if not file.filename.lower().endswith('.csv'):
            return jsonify({
                'success': False,
                'error': 'Please upload a CSV file.'
            }), 400

        # Read the CSV
        df = pd.read_csv(file)

        # Check if the CSV is empty
        if df.empty:
            return jsonify({
                'success': False,
                'error': 'The CSV file is empty.'
            }), 400

        # Required Spotify columns
        required_columns = [
            'wks',
            't10',
            'pk',
            'PkStreams',
            'total',
            'artist name',
            'song name'
        ]

        # Check that the required columns exist
        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:

            return jsonify({
                'success': False,
                'error': 'The CSV is missing required columns.',
                'missing_columns': missing_columns,
                'available_columns': df.columns.tolist()
            }), 400

        # Remove existing AIVision data
        AIData.query.delete()

        records_added = 0

        # Process every row
        for _, row in df.iterrows():

            # Convert numeric values safely
            feature1 = pd.to_numeric(
                row['wks'],
                errors='coerce'
            )

            feature2 = pd.to_numeric(
                row['t10'],
                errors='coerce'
            )

            feature3 = pd.to_numeric(
                row['pk'],
                errors='coerce'
            )

            target = pd.to_numeric(
                row['total'],
                errors='coerce'
            )

            # Skip rows with missing numeric values
            if pd.isna(feature1) or \
               pd.isna(feature2) or \
               pd.isna(feature3) or \
               pd.isna(target):

                continue

            # Artist becomes the category
            category = str(row['artist name'])

            # Create database record
            record = AIData(
                feature1=float(feature1),
                feature2=float(feature2),
                feature3=float(feature3),
                target=float(target),
                category=category,
                source='Spotify CSV'
            )

            db.session.add(record)

            records_added += 1

        # Save records to database
        db.session.commit()

        return jsonify({
            'success': True,
            'message': 'Spotify CSV uploaded successfully.',
            'filename': file.filename,
            'total_rows': len(df),
            'records_added': records_added,
            'numeric_features': [
                'wks',
                't10',
                'pk'
            ],
            'target': 'total',
            'category': 'artist name',
            'text_columns': [
                'artist and title',
                'x?',
                'artist name',
                'song name',
                'lyrics'
            ]
        })

    except Exception as e:

        # Undo incomplete database changes
        db.session.rollback()

        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============================================================
# NEURAL NETWORK (PM-08)
# ============================================================


class NeuralNetwork:

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size
    ):

        self.weights1 = (
            np.random.randn(
                input_size,
                hidden_size
            ) * 0.01
        )

        self.bias1 = np.zeros(
            (1, hidden_size)
        )

        self.weights2 = (
            np.random.randn(
                hidden_size,
                output_size
            ) * 0.01
        )

        self.bias2 = np.zeros(
            (1, output_size)
        )


    def forward(self, X):

        self.z1 = np.dot(
            X,
            self.weights1
        ) + self.bias1

        self.a1 = np.maximum(
            0,
            self.z1
        )

        self.z2 = np.dot(
            self.a1,
            self.weights2
        ) + self.bias2

        self.a2 = 1 / (
            1 + np.exp(
                -np.clip(
                    self.z2,
                    -500,
                    500
                )
            )
        )

        return self.a2


    def train(
        self,
        X,
        y,
        epochs=100,
        lr=0.01
    ):

        losses = []

        for _ in range(epochs):

            out = self.forward(X)

            err = out - y

            losses.append(
                float(np.mean(err ** 2))
            )

            d_out = (
                err *
                out *
                (1 - out)
            )

            d_w2 = np.dot(
                self.a1.T,
                d_out
            )

            d_b2 = np.sum(
                d_out,
                axis=0,
                keepdims=True
            )

            d_h = np.dot(
                d_out,
                self.weights2.T
            )

            d_h[self.z1 <= 0] = 0

            d_w1 = np.dot(
                X.T,
                d_h
            )

            d_b1 = np.sum(
                d_h,
                axis=0,
                keepdims=True
            )

            self.weights2 -= (
                lr * d_w2
            )

            self.bias2 -= (
                lr * d_b2
            )

            self.weights1 -= (
                lr * d_w1
            )

            self.bias1 -= (
                lr * d_b1
            )

        return losses


    def predict(self, X):

        return self.forward(X).flatten().tolist()


# ============================================================
# DATABASE MODELS
# ============================================================


class AIData(db.Model):

    __tablename__ = 'ai_data'

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    timestamp = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )

    feature1 = db.Column(
        db.Float
    )

    feature2 = db.Column(
        db.Float
    )

    feature3 = db.Column(
        db.Float
    )

    target = db.Column(
        db.Float
    )

    category = db.Column(
        db.String(50)
    )

    source = db.Column(
        db.String(100)
    )


    def to_dict(self):

        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'feature1': self.feature1,
            'feature2': self.feature2,
            'feature3': self.feature3,
            'target': self.target,
            'category': self.category,
            'source': self.source
        }


# ============================================================
# API: GENERATE DATA
# ============================================================


@app.route(
    '/api/data/generate',
    methods=['POST']
)
def generate_data():

    try:

        AIData.query.delete()

        db.session.commit()

        categories = [
            'Type_A',
            'Type_B',
            'Type_C'
        ]

        for _ in range(500):

            base = random.uniform(
                0,
                100
            )

            db.session.add(
                AIData(
                    timestamp=(
                        datetime.utcnow()
                        -
                        timedelta(
                            hours=random.randint(
                                0,
                                720
                            )
                        )
                    ),

                    feature1=round(
                        base +
                        random.uniform(
                            -10,
                            10
                        ),
                        2
                    ),

                    feature2=round(
                        base * 0.8 +
                        random.uniform(
                            -15,
                            15
                        ),
                        2
                    ),

                    feature3=round(
                        base * 1.2 +
                        random.uniform(
                            -5,
                            5
                        ),
                        2
                    ),

                    target=round(
                        base * 0.5 +
                        random.uniform(
                            -20,
                            20
                        ),
                        2
                    ),

                    category=random.choice(
                        categories
                    ),

                    source='Generated'
                )
            )

        db.session.commit()

        return jsonify({
            'success': True,
            'message': 'Generated 500 samples'
        })

    except Exception as e:

        db.session.rollback()

        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================
# API: STATISTICS
# ============================================================


@app.route(
    '/api/math/statistics',
    methods=['GET']
)
def get_statistics():

    data = AIData.query.limit(200).all()

    if not data:

        return jsonify({
            'success': False,
            'error': 'No data available'
        }), 404


    f1 = [
        d.feature1
        for d in data
    ]

    f2 = [
        d.feature2
        for d in data
    ]

    f3 = [
        d.feature3
        for d in data
    ]

    tg = [
        d.target
        for d in data
    ]


    return jsonify({

        'success': True,

        'statistics': {

            'feature1': {
                'mean': AIMath.mean(f1),
                'std_dev': AIMath.std_dev(f1),
                'min': min(f1),
                'max': max(f1)
            },

            'feature2': {
                'mean': AIMath.mean(f2),
                'std_dev': AIMath.std_dev(f2),
                'min': min(f2),
                'max': max(f2)
            },

            'feature3': {
                'mean': AIMath.mean(f3),
                'std_dev': AIMath.std_dev(f3),
                'min': min(f3),
                'max': max(f3)
            },

            'target': {
                'mean': AIMath.mean(tg),
                'std_dev': AIMath.std_dev(tg)
            },

            'correlations': {

                'f1_f2': AIMath.correlation(
                    f1,
                    f2
                ),

                'f1_f3': AIMath.correlation(
                    f1,
                    f3
                ),

                'f1_target': AIMath.correlation(
                    f1,
                    tg
                )
            },

            'activation_functions': {

                'sigmoid_demo': AIMath.sigmoid(
                    0.5
                ),

                'relu_demo': AIMath.relu(
                    -1.5
                ),

                'softmax_demo': AIMath.softmax(
                    [1.0, 2.0, 3.0]
                ),

                'bayes_demo': AIMath.bayes(
                    0.3,
                    0.8,
                    0.5
                )
            }
        }
    })


# ============================================================
# RUN APPLICATION
# ============================================================


with app.app_context():

    db.create_all()

    print(
        "Database created successfully"
    )


if __name__ == '__main__':

    print(
        "\nAIVision running at "
        "http://localhost:5000\n"
    )

    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000
    )