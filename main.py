# main.py - FastAPI Backend with Database Integration

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import shutil
import os
import cv2
import numpy as np
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import base64
from io import BytesIO
from typing import Optional, List
from ultralytics import YOLO
import requests
import sqlalchemy
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from typing import Dict, Any

# Initialize SQLAlchemy
Base = declarative_base()
DATABASE_URL = "sqlite:///./olive_disease_detection.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Database models
class AnalysisResult(Base):
    __tablename__ = "analysis_results"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.now)
    filename = Column(String)
    region = Column(String)
    region_name = Column(String)
    detection_image_path = Column(String)
    charts_image_path = Column(String)
    pdf_report_path = Column(String)
    
    # Analysis summary
    total_leaves = Column(Integer)
    total_fruits = Column(Integer)
    leaf_infection_rate = Column(String)
    fruit_infection_rate = Column(String)
    
    # Weather data
    temperature = Column(Float)
    humidity = Column(Float)
    temperature_status = Column(String)
    humidity_status = Column(String)
    
    # JSON data for detailed information
    detailed_results = Column(Text)  # Store as JSON string
    
    # Recommendations
    recommendations = relationship("Recommendation", back_populates="analysis")

class Recommendation(Base):
    __tablename__ = "recommendations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    analysis_id = Column(String, ForeignKey("analysis_results.id"))
    recommendation_text = Column(Text)
    
    analysis = relationship("AnalysisResult", back_populates="recommendations")

# Define common olive diseases for leaf classification
LEAF_DISEASES = {
    0: "Healthy",
    1: "aculus_olearius",  # Aculus olearius
    2: "olive_peacock_spot"
}

FRUIT_DISEASES = {
    0: "healthy",
    1: "olive_fruit_fly",
    2: "olive_anthracnose"
}

# Define standard optimal growing conditions for olives globally
OLIVE_GROWING_CONDITIONS = {
    "standard": {
        "temperature": {"min": 15, "max": 30, "unit": "°C"},
        "humidity": {"min": 40, "max": 65, "unit": "%"},
        "rainfall": {"min": 400, "max": 700, "unit": "mm/year"}
    }
}

# Predefined locations for olive growing regions
OLIVE_GROWING_REGIONS = {
    "karachi_pakistan": {"latitude": 24.8607, "longitude": 67.0011, "name": "Karachi, Pakistan"},
    "islamabad_pakistan": {"latitude": 33.6844, "longitude": 73.0479, "name": "Islamabad, Pakistan"},
    "quetta_pakistan": {"latitude": 30.1798, "longitude": 66.9750, "name": "Quetta, Pakistan"},
    "peshawar_pakistan": {"latitude": 34.0151, "longitude": 71.5249, "name": "Peshawar, Pakistan"},
    "mumbai_india": {"latitude": 19.0760, "longitude": 72.8777, "name": "Mumbai, India"},
    "rajasthan_india": {"latitude": 27.0238, "longitude": 74.2179, "name": "Rajasthan, India"},
    "athens_greece": {"latitude": 37.9838, "longitude": 23.7275, "name": "Athens, Greece"},
    "seville_spain": {"latitude": 37.3891, "longitude": -5.9845, "name": "Seville, Spain"},
    "tuscany_italy": {"latitude": 43.7711, "longitude": 11.2486, "name": "Tuscany, Italy"}
}

CONF_THRESHOLD = 0.5

class OliveDiseaseDetection:
    def __init__(self, leaf_model_path, leaf_classifier_path, fruit_model_path):
        self.leaf_model = YOLO(leaf_model_path)
        self.fruit_model = YOLO(fruit_model_path)
        self.leaf_classifier = YOLO(leaf_classifier_path)
        
    def classify_leaf_disease(self, image, x1, y1, x2, y2):
        leaf_roi = image[y1:y2, x1:x2]
        if leaf_roi.size == 0:
            return 0, 0.0
            
        # Save temporary ROI
        temp_path = "temp_leaf.jpg"
        cv2.imwrite(temp_path, leaf_roi)
        
        # Perform classification
        results = self.leaf_classifier(temp_path, conf=0.5)
        
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        class_id = results[0].probs.top1
        confidence = results[0].probs.top1conf.item()
        
        return class_id, confidence

    def classify_fruit_disease(self, image, x1, y1, x2, y2):
        fruit_roi = image[y1:y2, x1:x2]
        if fruit_roi.size == 0:
            return 0, 0.0
        hsv_roi = cv2.cvtColor(fruit_roi, cv2.COLOR_BGR2HSV)
        dark_spots = cv2.inRange(hsv_roi, (0, 0, 0), (180, 255, 100))
        spot_ratio = np.sum(dark_spots) / dark_spots.size / 255
        brown_orange_areas = cv2.inRange(hsv_roi, (5, 50, 50), (30, 255, 255))
        brown_orange_ratio = np.sum(brown_orange_areas) / brown_orange_areas.size / 255
        if spot_ratio > 0.3:
            return 1, min(spot_ratio * 3, 0.99)  # Olive fruit fly damage
        elif brown_orange_ratio > 0.1:
            return 2, min(brown_orange_ratio * 3, 0.99)  # Anthracnose
        health_score = 1.0 - (spot_ratio + brown_orange_ratio) * 3
        return 0, max(0.5, min(health_score, 0.99))

    def analyze_diseases(self, image, leaf_results, fruit_results):
        leaf_disease_counts = {
            "Healthy": 0,
            "aculus_olearius": 0,
            "olive_peacock_spot": 0
        }
        fruit_disease_counts = {
            "healthy": 0,
            "olive_fruit_fly": 0,
            "olive_anthracnose": 0
        }
        
        # Store detection details for visualization
        leaf_detections = []
        fruit_detections = []
        
        # Process detected leaves
        filtered_leaf_boxes = [box for box in leaf_results.boxes if box.conf[0] > CONF_THRESHOLD]
        filtered_fruit_boxes = [box for box in fruit_results.boxes if box.conf[0] > CONF_THRESHOLD]

        for box in filtered_leaf_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            disease_id, disease_conf = self.classify_leaf_disease(image, x1, y1, x2, y2)
            disease_name = LEAF_DISEASES[disease_id]
            leaf_disease_counts[disease_name] += 1
            leaf_detections.append({
                'bbox': (x1, y1, x2, y2),
                'disease': disease_name,
                'confidence': disease_conf
            })
        
        for box in filtered_fruit_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            disease_id, disease_conf = self.classify_fruit_disease(image, x1, y1, x2, y2)
            disease_name = FRUIT_DISEASES[disease_id]
            fruit_disease_counts[disease_name] += 1
            fruit_detections.append({
                'bbox': (x1, y1, x2, y2),
                'disease': disease_name,
                'confidence': disease_conf
            })
        
        # Calculate percentages
        total_leaves = sum(leaf_disease_counts.values())
        total_fruits = sum(fruit_disease_counts.values())
        
        leaf_percentages = {}
        for disease, count in leaf_disease_counts.items():
            leaf_percentages[disease] = {
                'count': count,
                'percentage': (count / total_leaves * 100) if total_leaves > 0 else 0
            }
            
        fruit_percentages = {}
        for disease, count in fruit_disease_counts.items():
            fruit_percentages[disease] = {
                'count': count,
                'percentage': (count / total_fruits * 100) if total_fruits > 0 else 0
            }
        
        return {
            "leaves": {
                "counts": leaf_disease_counts,
                "percentages": leaf_percentages,
                "total": total_leaves,
                "detections": leaf_detections
            },
            "fruits": {
                "counts": fruit_disease_counts,
                "percentages": fruit_percentages,
                "total": total_fruits,
                "detections": fruit_detections
            }
        }

class Visualizer:
    @staticmethod
    def draw_detection_boxes(image, disease_analysis):
        """Draw detection boxes on the image with disease labels"""
        img_with_boxes = image.copy()
        
        # Define colors for different diseases
        colors = {
            "Healthy": (0, 255, 0),       # Green
            "healthy": (0, 255, 0),       # Green
            "aculus_olearius": (0, 165, 255),  # Orange
            "olive_peacock_spot": (0, 0, 255),  # Red
            "olive_fruit_fly": (255, 0, 0),     # Blue
            "olive_anthracnose": (255, 0, 255)  # Purple
        }
        
        # Draw leaf detections
        for detection in disease_analysis['leaves']['detections']:
            x1, y1, x2, y2 = detection['bbox']
            disease = detection['disease']
            confidence = detection['confidence']
            color = colors.get(disease, (255, 255, 255))
            
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
            label = f"{disease} ({confidence:.2f})"
            cv2.putText(img_with_boxes, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw fruit detections
        for detection in disease_analysis['fruits']['detections']:
            x1, y1, x2, y2 = detection['bbox']
            disease = detection['disease']
            confidence = detection['confidence']
            color = colors.get(disease, (255, 255, 255))
            
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
            label = f"{disease} ({confidence:.2f})"
            cv2.putText(img_with_boxes, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return img_with_boxes
    
    @staticmethod
    def create_distribution_charts(disease_analysis):
        """Create pie charts showing disease distribution"""
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2, figure=fig)
        
        # Leaf diseases pie chart
        ax1 = fig.add_subplot(gs[0, 0])
        leaf_data = []
        leaf_labels = []
        for disease, info in disease_analysis['leaves']['percentages'].items():
            if info['count'] > 0:
                leaf_data.append(info['percentage'])
                leaf_labels.append(f"{disease}\n({info['count']}, {info['percentage']:.1f}%)")
        
        if leaf_data:
            ax1.pie(leaf_data, labels=leaf_labels, autopct='', 
                   colors=['lightgreen', 'orange', 'red'][:len(leaf_data)])
            ax1.set_title('Leaf Health Distribution', fontsize=14, fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'No leaves detected', ha='center', va='center')
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
        
        # Fruit diseases pie chart
        ax2 = fig.add_subplot(gs[0, 1])
        fruit_data = []
        fruit_labels = []
        for disease, info in disease_analysis['fruits']['percentages'].items():
            if info['count'] > 0:
                fruit_data.append(info['percentage'])
                fruit_labels.append(f"{disease}\n({info['count']}, {info['percentage']:.1f}%)")
        
        if fruit_data:
            ax2.pie(fruit_data, labels=fruit_labels, autopct='', 
                   colors=['lightgreen', 'lightblue', 'purple'][:len(fruit_data)])
            ax2.set_title('Fruit Health Distribution', fontsize=14, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No fruits detected', ha='center', va='center')
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
        
        # Overall health bar chart
        ax3 = fig.add_subplot(gs[1, :])
        categories = ['Healthy Leaves', 'Diseased Leaves', 'Healthy Fruits', 'Diseased Fruits']
        
        healthy_leaves = disease_analysis['leaves']['percentages'].get('Healthy', {}).get('count', 0)
        diseased_leaves = disease_analysis['leaves']['total'] - healthy_leaves
        healthy_fruits = disease_analysis['fruits']['percentages'].get('healthy', {}).get('count', 0)
        diseased_fruits = disease_analysis['fruits']['total'] - healthy_fruits
        
        values = [healthy_leaves, diseased_leaves, healthy_fruits, diseased_fruits]
        colors = ['green', 'red', 'green', 'red']
        
        bars = ax3.bar(categories, values, color=colors)
        ax3.set_ylabel('Count', fontsize=12)
        ax3.set_title('Overall Health Summary', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(value)}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig

class WeatherAnalysis:
    @staticmethod
    def get_weather_data(region_key):
        """Get weather data from Open-Meteo API based on predefined region coordinates"""
        if region_key not in OLIVE_GROWING_REGIONS:
            # Default to Karachi if region not found
            region_key = "karachi_pakistan"
            
        region = OLIVE_GROWING_REGIONS[region_key]
        
        try:
            # Make API request to Open-Meteo
            url = f"https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": region["latitude"],
                "longitude": region["longitude"],
                "current": "temperature_2m,relative_humidity_2m",
                "timezone": "auto"
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "temperature": data["current"]["temperature_2m"],
                    "humidity": data["current"]["relative_humidity_2m"],
                    "city": region["name"].split(",")[0],
                    "country": region["name"].split(",")[1].strip() if "," in region["name"] else "",
                    "latitude": region["latitude"],
                    "longitude": region["longitude"]
                }
            else:
                # Fallback values if API fails
                return {
                    "temperature": 25, 
                    "humidity": 50,
                    "city": region["name"].split(",")[0],
                    "country": region["name"].split(",")[1].strip() if "," in region["name"] else "",
                    "latitude": region["latitude"],
                    "longitude": region["longitude"]
                }
                
        except Exception as e:
            print(f"Error fetching weather data: {e}")
            # Fallback values if API fails
            return {
                "temperature": 25, 
                "humidity": 50,
                "city": region["name"].split(",")[0],
                "country": region["name"].split(",")[1].strip() if "," in region["name"] else "",
                "latitude": region["latitude"],
                "longitude": region["longitude"]
            }

    @staticmethod
    def evaluate_weather_conditions(weather_data):
        """Evaluate weather conditions against standard olive growing conditions"""
        conditions = OLIVE_GROWING_CONDITIONS["standard"]
        
        temp_min = conditions.get("temperature", {}).get("min", 0)
        temp_max = conditions.get("temperature", {}).get("max", 100)
        humid_min = conditions.get("humidity", {}).get("min", 0)
        humid_max = conditions.get("humidity", {}).get("max", 100)
        
        temp = weather_data.get("temperature", 25)
        humidity = weather_data.get("humidity", 50)
        
        if temp < temp_min:
            temperature_status = "too_cold"
        elif temp > temp_max:
            temperature_status = "too_hot"
        else:
            temperature_status = "optimal"
            
        if humidity < humid_min:
            humidity_status = "too_dry"
        elif humidity > humid_max:
            humidity_status = "too_humid"
        else:
            humidity_status = "optimal"
            
        return temperature_status, humidity_status

class OliveFarmRecommendation:
    @staticmethod
    def generate_recommendations(disease_analysis, weather_analysis):
        recommendations = []
        
        # Calculate infection rates
        leaf_total = disease_analysis['leaves']['total']
        fruit_total = disease_analysis['fruits']['total']
        
        if leaf_total > 0:
            leaf_infection_rate = (1 - disease_analysis['leaves']['percentages']['Healthy']['percentage'] / 100)
            recommendations.append(f"Leaf infection rate: {leaf_infection_rate * 100:.1f}%")
        
        if fruit_total > 0:
            fruit_infection_rate = (1 - disease_analysis['fruits']['percentages']['healthy']['percentage'] / 100)
            recommendations.append(f"Fruit infection rate: {fruit_infection_rate * 100:.1f}%")

        # Temperature-based recommendations
        if weather_analysis['temperature_status'] == 'too_hot':
            recommendations.append("Temperature is higher than optimal. Ensure adequate irrigation and consider providing shade for young trees.")
        elif weather_analysis['temperature_status'] == 'too_cold':
            recommendations.append("Temperature is lower than optimal. Consider using frost protection methods or apply Engro's CuGuard (Copper Oxychloride) as anti-frost spray.")
        elif weather_analysis['temperature_status'] == 'optimal':
            recommendations.append("Temperature conditions are optimal for olive trees.")
        elif weather_analysis['temperature_status'] == 'unknown':
            recommendations.append("Unable to assess temperature conditions due to missing data.")

        # Humidity-based recommendations
        if weather_analysis['humidity_status'] == 'too_dry':
            recommendations.append("Humidity is lower than optimal. Increase irrigation frequency and consider applying Engro's Zarkhez mulch to retain soil moisture.")
        elif weather_analysis['humidity_status'] == 'too_humid':
            recommendations.append("Humidity is higher than optimal. Ensure good air circulation around trees and apply Engro's CuGuard (Copper Oxychloride) preventively.")
        elif weather_analysis['humidity_status'] == 'optimal':
            recommendations.append("Humidity conditions are optimal for olive trees.")
        elif weather_analysis['humidity_status'] == 'unknown':
            recommendations.append("Unable to assess humidity conditions due to missing data.")
        
        # Disease-specific recommendations
        if disease_analysis['leaves']['counts']['aculus_olearius'] > 0:
            percentage = disease_analysis['leaves']['percentages']['aculus_olearius']['percentage']
            recommendations.append(f"{percentage:.1f}% of leaves show symptoms of Aculus Olearius (tiny mites). Apply Engro's Abamite (abamectin 1.8% EC) at 0.5ml/L water and remove infected leaves.")
        
        if disease_analysis['leaves']['counts']['olive_peacock_spot'] > 0:
            percentage = disease_analysis['leaves']['percentages']['olive_peacock_spot']['percentage']
            recommendations.append(f"{percentage:.1f}% of leaves show symptoms of Olive Peacock Spot. Apply Engro's CuGuard (Copper Oxychloride 50% WP) at 2-3g/L water.")

        if disease_analysis['fruits']['counts']['olive_fruit_fly'] > 0:
            percentage = disease_analysis['fruits']['percentages']['olive_fruit_fly']['percentage']
            recommendations.append(f"{percentage:.1f}% of fruits are affected by Olive Fruit Fly. Set up pheromone traps and apply Engro's Lambda Super (lambda-cyhalothrin 2.5% EC) at 1ml/L water.")
        
        if disease_analysis['fruits']['counts']['olive_anthracnose'] > 0:
            percentage = disease_analysis['fruits']['percentages']['olive_anthracnose']['percentage']
            recommendations.append(f"{percentage:.1f}% of fruits show signs of Anthracnose. Apply Engro's Curzate (cymoxanil + mancozeb) at 2g/L water and remove infected fruits.")
        
        # Overall health status
        if leaf_total > 0 and fruit_total > 0:
            if (disease_analysis['leaves']['percentages']['Healthy']['percentage'] == 100 and 
                disease_analysis['fruits']['percentages']['healthy']['percentage'] == 100):
                recommendations.append("All olive trees and fruits are healthy. Apply preventive Engro CuGuard (Copper Oxychloride) at 2g/L water twice per year.")
        
        return recommendations
    
class OliveFarmAnalysis:
    def __init__(self, leaf_model_path, leaf_classifier_path, fruit_model_path, region="karachi_pakistan"):
        self.disease_detection = OliveDiseaseDetection(leaf_model_path, leaf_classifier_path, fruit_model_path)
        self.weather_analysis = WeatherAnalysis()
        self.recommendation_system = OliveFarmRecommendation()
        self.visualizer = Visualizer()
        self.region = region

    def run_analysis(self, image, leaf_results, fruit_results):
        # Get disease analysis
        disease_analysis = self.disease_detection.analyze_diseases(image, leaf_results, fruit_results)
        
        # Get weather data and analysis
        weather_data = self.weather_analysis.get_weather_data(self.region)
        temp_status, humidity_status = self.weather_analysis.evaluate_weather_conditions(weather_data)
        
        # Generate recommendations
        weather_status = {
            "temperature_status": temp_status, 
            "humidity_status": humidity_status
        }
        recommendations = self.recommendation_system.generate_recommendations(disease_analysis, weather_status)
        
        # Create visualizations
        img_with_boxes = self.visualizer.draw_detection_boxes(image, disease_analysis)
        cv2.imwrite('detection_result.jpg', img_with_boxes)
        
        distribution_fig = self.visualizer.create_distribution_charts(disease_analysis)
        distribution_fig.savefig('disease_distribution.png', dpi=300, bbox_inches='tight')
        plt.close(distribution_fig)
        
        # Compile enriched result data
        leaf_infection_rate = "0.0%"
        if disease_analysis['leaves']['total'] > 0:
            leaf_infection_rate = f"{(1 - disease_analysis['leaves']['percentages']['Healthy']['percentage'] / 100) * 100:.1f}%"
            
        fruit_infection_rate = "0.0%"
        if disease_analysis['fruits']['total'] > 0:
            fruit_infection_rate = f"{(1 - disease_analysis['fruits']['percentages']['healthy']['percentage'] / 100) * 100:.1f}%"
        
        result_data = {
            "summary": {
                "total_leaves_analyzed": disease_analysis['leaves']['total'],
                "total_fruits_analyzed": disease_analysis['fruits']['total'],
                "overall_leaf_infection_rate": leaf_infection_rate,
                "overall_fruit_infection_rate": fruit_infection_rate
            },
            "disease_analysis": {
                "leaves": {
                    "raw_counts": disease_analysis['leaves']['counts'],
                    "percentage_distribution": {
                        disease: {
                            "count": info['count'],
                            "percentage": f"{info['percentage']:.1f}%"
                        }
                        for disease, info in disease_analysis['leaves']['percentages'].items()
                    },
                    "total_detected": disease_analysis['leaves']['total']
                },
                "fruits": {
                    "raw_counts": disease_analysis['fruits']['counts'],
                    "percentage_distribution": {
                        disease: {
                            "count": info['count'],
                            "percentage": f"{info['percentage']:.1f}%"
                        }
                        for disease, info in disease_analysis['fruits']['percentages'].items()
                    },
                    "total_detected": disease_analysis['fruits']['total']
                }
            },
            "weather_conditions": {
                "weather_data": weather_data,
                "temperature_status": temp_status,
                "humidity_status": humidity_status
            },
            "recommendations": recommendations,
            "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "visualizations": {
                "detection_image": "detection_result.jpg",
                "distribution_charts": "disease_distribution.png"
            }
        }
        
        return result_data

# Route Models and Endpoints setup
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory user store
USERS_DB = {
    "admin": {
        "username": "admin",
        "password": "admin"
    }
}

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Token expiration time
ACCESS_TOKEN_EXPIRE_MINUTES = 120

class Token(BaseModel):
    access_token: str
    token_type: str

class User(BaseModel):
    username: str

class RegionSelect(BaseModel):
    region: str

class AnalysisResultResponse(BaseModel):
    id: str
    timestamp: str
    region_name: str
    total_leaves: int
    total_fruits: int
    leaf_infection_rate: str
    fruit_infection_rate: str

# Simple token generation (no JWT for simplicity)
def create_access_token(data: dict):
    return data["username"]

def verify_token(token: str):
    if token == "admin":
        return token
    raise HTTPException(status_code=401, detail="Invalid authentication credentials")

@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = USERS_DB.get(form_data.username)
    if not user or user["password"] != form_data.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token({"username": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}

# Protected route decorator
async def get_current_user(token: str = Depends(oauth2_scheme)):
    return verify_token(token)

# Initialize analysis with model paths
analysis = None

@app.on_event("startup")
async def startup_event():
    global analysis
    # Initialize models
    leaf_model_path = 'weights/leave_detection.pt'
    fruit_model_path = 'weights/fruit_detection.pt'
    leaf_classifier_path = 'weights/leave_classification.pt'
    
    analysis = OliveFarmAnalysis(leaf_model_path, leaf_classifier_path, fruit_model_path)
    
    # Create necessary directories
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Create database tables
    Base.metadata.create_all(bind=engine)

@app.get("/regions")
async def get_regions():
    """Return list of available olive growing regions"""
    return {
        "regions": [
            {"key": key, "name": region["name"]} 
            for key, region in OLIVE_GROWING_REGIONS.items()
        ]
    }

@app.post("/set_region/")
async def set_region(
    region_data: RegionSelect,
    current_user: str = Depends(get_current_user)
):
    """Set the region for analysis"""
    global analysis
    
    if region_data.region in OLIVE_GROWING_REGIONS:
        analysis.region = region_data.region
        return {"status": "success", "message": f"Region set to {OLIVE_GROWING_REGIONS[region_data.region]['name']}"}
    else:
        return {"status": "error", "message": "Invalid region selected"}

@app.post("/upload/")
async def upload_image(
    file: UploadFile = File(...),
    current_user: str = Depends(get_current_user)
):
    try:
        # Save uploaded file
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {"status": "success", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/")
async def analyze_image(
    filename: str,
    current_user: str = Depends(get_current_user),
    db: sqlalchemy.orm.Session = Depends(get_db)
):
    try:
        file_path = f"uploads/{filename}"
        
        # Load image
        image = cv2.imread(file_path)
        if image is None:
            raise HTTPException(status_code=400, detail="Could not read image")
        
        # Run analysis
        leaf_results = analysis.disease_detection.leaf_model(image)[0]
        fruit_results = analysis.disease_detection.fruit_model(image)[0]
        
        result = analysis.run_analysis(image, leaf_results, fruit_results)
        
        # Save results
        result_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = f"results/{result_id}"
        os.makedirs(result_dir, exist_ok=True)
        
        # Save detection image
        detection_img_path = f"{result_dir}/detection_result.jpg"
        shutil.copy("detection_result.jpg", detection_img_path)
        
        # Save charts
        charts_path = f"{result_dir}/disease_distribution.png"
        shutil.copy("disease_distribution.png", charts_path)
        
        # Save JSON results
        json_path = f"{result_dir}/analysis_results.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=4)
        
        # Convert images to base64 for frontend display
        with open(detection_img_path, "rb") as img_file:
            detection_img_b64 = base64.b64encode(img_file.read()).decode()
        
        with open(charts_path, "rb") as img_file:
            charts_img_b64 = base64.b64encode(img_file.read()).decode()
        
        # Store results in database
        db_result = AnalysisResult(
            id=result_id,
            user_id=current_user,
            filename=filename,
            region=analysis.region,
            region_name=OLIVE_GROWING_REGIONS[analysis.region]['name'],
            detection_image_path=detection_img_path,
            charts_image_path=charts_path,
            pdf_report_path=f"{result_dir}/olive_analysis_report.pdf",
            total_leaves=result['summary']['total_leaves_analyzed'],
            total_fruits=result['summary']['total_fruits_analyzed'],
            leaf_infection_rate=result['summary']['overall_leaf_infection_rate'],
            fruit_infection_rate=result['summary']['overall_fruit_infection_rate'],
            temperature=result['weather_conditions']['weather_data']['temperature'],
            humidity=result['weather_conditions']['weather_data']['humidity'],
            temperature_status=result['weather_conditions']['temperature_status'],
            humidity_status=result['weather_conditions']['humidity_status'],
            detailed_results=json.dumps(result)
        )
        db.add(db_result)
        
        # Add recommendations to database
        for rec_text in result['recommendations']:
            recommendation = Recommendation(
                analysis_id=result_id,
                recommendation_text=rec_text
            )
            db.add(recommendation)
        
        db.commit()
        
        return {
            "status": "success",
            "result_id": result_id,
            "analysis": result,
            "detection_image": detection_img_b64,
            "charts_image": charts_img_b64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/generate_pdf/{result_id}")
async def generate_pdf(
    result_id: str,
    current_user: str = Depends(get_current_user),
    db: sqlalchemy.orm.Session = Depends(get_db)
):
    try:
        # First check if it exists in the database
        db_result = db.query(AnalysisResult).filter(AnalysisResult.id == result_id).first()
        if not db_result:
            raise HTTPException(status_code=404, detail="Analysis result not found")
        
        result_dir = f"results/{result_id}"
        json_path = f"{result_dir}/analysis_results.json"
        
        # If JSON file exists, use it; otherwise use data from DB
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                result = json.load(f)
        else:
            result = json.loads(db_result.detailed_results)
        
        pdf_path = f"{result_dir}/olive_analysis_report.pdf"
        
        # Create PDF
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        story.append(Paragraph("Olive Disease Analysis Report", title_style))
        story.append(Spacer(1, 12))
        
        # Timestamp
        story.append(Paragraph(f"Analysis Date: {result['analysis_timestamp']}", styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Location
        if 'weather_conditions' in result and 'weather_data' in result['weather_conditions']:
            weather_data = result['weather_conditions']['weather_data']
            if 'city' in weather_data and 'country' in weather_data:
                location = f"Location: {weather_data['city']}, {weather_data['country']}"
                story.append(Paragraph(location, styles['Normal']))
                story.append(Spacer(1, 12))
        
        # Summary section
        summary_style = ParagraphStyle(
            'SummaryStyle',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12
        )
        story.append(Paragraph("Summary", summary_style))
        
        summary_data = [
            ["Metric", "Value"],
            ["Total Leaves Analyzed", str(result['summary']['total_leaves_analyzed'])],
            ["Total Fruits Analyzed", str(result['summary']['total_fruits_analyzed'])],
            ["Leaf Infection Rate", result['summary']['overall_leaf_infection_rate']],
            ["Fruit Infection Rate", result['summary']['overall_fruit_infection_rate']]
        ]
        
        summary_table = Table(summary_data, colWidths=[200, 200])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        # Weather conditions section
        story.append(Paragraph("Weather Conditions", summary_style))
        
        if 'weather_conditions' in result and 'weather_data' in result['weather_conditions']:
            weather_data = result['weather_conditions']['weather_data']
            temp_status = result['weather_conditions']['temperature_status']
            humidity_status = result['weather_conditions']['humidity_status']
            
            weather_data_table = [
                ["Condition", "Value", "Status"],
                ["Temperature", f"{weather_data.get('temperature', 'N/A')}°C", temp_status.replace('_', ' ').title()],
                ["Humidity", f"{weather_data.get('humidity', 'N/A')}%", humidity_status.replace('_', ' ').title()]
            ]
            
            weather_table = Table(weather_data_table, colWidths=[133, 133, 133])
            weather_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(weather_table)
        
        story.append(Spacer(1, 20))
        
        # Detection image
        detection_img_path = f"{result_dir}/detection_result.jpg"
        if os.path.exists(detection_img_path):
            story.append(Paragraph("Detection Results", summary_style))
            story.append(Image(detection_img_path, width=500, height=300))
            story.append(Spacer(1, 20))
        
        # Charts
        charts_path = f"{result_dir}/disease_distribution.png"
        if os.path.exists(charts_path):
            story.append(Paragraph("Disease Distribution", summary_style))
            story.append(Image(charts_path, width=500, height=350))
            story.append(Spacer(1, 20))
        
        # Recommendations
        story.append(Paragraph("Recommendations", summary_style))
        for rec in result['recommendations']:
            story.append(Paragraph(f"• {rec}", styles['Normal']))
            story.append(Spacer(1, 6))
        
        doc.build(story)
        
        # Update database with PDF path
        db_result.pdf_report_path = pdf_path
        db.commit()
        
        return FileResponse(pdf_path, filename=f"olive_analysis_report_{result_id}.pdf")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def get_analysis_history(
    current_user: str = Depends(get_current_user),
    db: sqlalchemy.orm.Session = Depends(get_db)
):
    """Get the history of analysis results for the current user"""
    try:
        results = db.query(AnalysisResult).filter(
            AnalysisResult.user_id == current_user
        ).order_by(AnalysisResult.timestamp.desc()).all()
        
        response_data = []
        for result in results:
            response_data.append({
                "id": result.id,
                "timestamp": result.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "region_name": result.region_name,
                "total_leaves": result.total_leaves,
                "total_fruits": result.total_fruits,
                "leaf_infection_rate": result.leaf_infection_rate,
                "fruit_infection_rate": result.fruit_infection_rate
            })
        
        return {"status": "success", "results": response_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analysis/{result_id}")
async def get_analysis_detail(
    result_id: str,
    current_user: str = Depends(get_current_user),
    db: sqlalchemy.orm.Session = Depends(get_db)
):
    """Get detailed information for a specific analysis result"""
    try:
        result = db.query(AnalysisResult).filter(
            AnalysisResult.id == result_id,
            AnalysisResult.user_id == current_user
        ).first()
        
        if not result:
            raise HTTPException(status_code=404, detail="Analysis result not found")
        
        # Get recommendations
        recommendations = db.query(Recommendation).filter(
            Recommendation.analysis_id == result_id
        ).all()
        
        # Convert detection image to base64
        detection_img_b64 = ""
        if os.path.exists(result.detection_image_path):
            with open(result.detection_image_path, "rb") as img_file:
                detection_img_b64 = base64.b64encode(img_file.read()).decode()
        
        # Convert charts image to base64
        charts_img_b64 = ""
        if os.path.exists(result.charts_image_path):
            with open(result.charts_image_path, "rb") as img_file:
                charts_img_b64 = base64.b64encode(img_file.read()).decode()
        
        # Parse detailed results if available
        detailed_results = {}
        if result.detailed_results:
            detailed_results = json.loads(result.detailed_results)
        
        response_data = {
            "id": result.id,
            "timestamp": result.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "region_name": result.region_name,
            "summary": {
                "total_leaves_analyzed": result.total_leaves,
                "total_fruits_analyzed": result.total_fruits,
                "overall_leaf_infection_rate": result.leaf_infection_rate,
                "overall_fruit_infection_rate": result.fruit_infection_rate
            },
            "weather_conditions": {
                "weather_data": {
                    "temperature": result.temperature,
                    "humidity": result.humidity,
                    "city": result.region_name.split(',')[0] if ',' in result.region_name else result.region_name,
                    "country": result.region_name.split(',')[1].strip() if ',' in result.region_name else ""
                },
                "temperature_status": result.temperature_status,
                "humidity_status": result.humidity_status
            },
            "recommendations": [rec.recommendation_text for rec in recommendations],
            "detection_image": detection_img_b64,
            "charts_image": charts_img_b64,
            "detailed_results": detailed_results
        }
        
        return {"status": "success", "analysis": response_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)