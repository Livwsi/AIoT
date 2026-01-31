# AIoT Workshop - Complete Beginner's Guide
## Bus Arrival Time Prediction Using Machine Learning

---

## 📚 Table of Contents

1. [Introduction](#introduction)
2. [What You'll Learn](#what-youll-learn)
3. [Pre-Workshop Setup](#pre-workshop-setup)
4. [Understanding the Technologies](#understanding-the-technologies)
5. [The Mathematics Behind Machine Learning](#the-mathematics-behind-machine-learning)
6. [Step-by-Step Tutorial](#step-by-step-tutorial)
7. [Workshop Day Instructions](#workshop-day-instructions)
8. [Troubleshooting](#troubleshooting)

---

## Introduction

Welcome to the AIoT (Artificial Intelligence of Things) Workshop! 

**The Challenge:** You will build a machine learning model that predicts exactly when a bus will arrive at its destination using real-time GPS data.

**The Competition:** 10 rounds of bus journeys. Your model predicts arrival time. Most accurate predictor wins!

**Why This Matters:** This is how real-world systems work:
- Google Maps predicting your arrival time
- Uber estimating pickup times
- Package delivery tracking
- Traffic management systems

You'll build a complete AIoT system from scratch!

---

## What You'll Learn

### Technical Skills:
- **Machine Learning:** Build neural networks using TensorFlow
- **IoT Communication:** Work with MQTT protocol (industry standard)
- **Real-Time Systems:** Process streaming data
- **Python Programming:** pandas, numpy, data processing
- **Model Evaluation:** Measure and improve accuracy

### Concepts:
- **Supervised Learning:** Teaching computers to predict from examples
- **Neural Networks:** How artificial neurons learn patterns
- **Feature Engineering:** Transforming raw data for ML
- **Regression:** Predicting continuous values (time)
- **Optimization:** Gradient descent and backpropagation

### Practical Experience:
- Training ML models on large datasets
- Making real-time predictions
- Competing in data science challenge
- Iterative model improvement

---

## Pre-Workshop Setup

### 📥 Downloads

#### 1. Install Anaconda (Python Distribution)

**What is Anaconda?**
- A package of Python and 250+ scientific libraries
- Includes Jupyter Notebook (interactive coding environment)
- Manages dependencies automatically

**Download:** https://www.anaconda.com/download

**Steps:**
1. Go to link above
2. Click "Download" (choose Windows 64-bit)
3. File size: ~600 MB
4. Run installer (takes 10-15 minutes)
5. **IMPORTANT:** During installation, check "Add Anaconda to PATH" (even if not recommended)
6. Restart computer after installation

**Verify Installation:**
Open Command Prompt and type:
```cmd
python --version
```
Should show: `Python 3.x.x`

```cmd
jupyter --version
```
Should show version numbers

---

#### 2. Install Required Python Libraries

**Open Anaconda Prompt** (search in Start menu)

**Install TensorFlow:**
```cmd
pip install tensorflow
```
**What is TensorFlow?**
- Google's machine learning framework
- Used for building neural networks
- Industry standard (used by Google, Uber, Airbnb)
- Download: ~500 MB, takes 5-10 minutes

**Install MQTT Library:**
```cmd
pip install paho-mqtt
```
**What is MQTT?**
- Message Queuing Telemetry Transport
- Lightweight protocol for IoT devices
- Used for real-time data streaming
- Like WhatsApp for sensors!

**Install Additional Libraries:**
```cmd
pip install requests flask flask-cors
```
- **requests:** For HTTP communication (sending predictions)
- **flask:** Web framework (for leaderboard server)
- **flask-cors:** Cross-origin resource sharing

**Verify All Installations:**
```cmd
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import paho.mqtt.client as mqtt; print('MQTT: OK')"
python -c "import pandas; print('pandas:', pandas.__version__)"
python -c "import numpy; print('numpy:', numpy.__version__)"
```
All should print without errors!

---

#### 3. Download Workshop Files

Your instructor will send you:
1. **training_dataset.csv** (~10 MB)
   - 100 bus trajectories with GPS data
   - 100,000 data points
   - This is what you'll train your model on

2. **student_template.ipynb** (Jupyter notebook)
   - Complete code template
   - You'll run this step-by-step

**Save both files in:**
```
C:\Users\YourName\Desktop\AIoT_Workshop\
```

---

#### 4. Test Your Setup

**Open Jupyter Notebook:**
```cmd
cd C:\Users\YourName\Desktop\AIoT_Workshop
jupyter notebook
```

Browser will open automatically showing your files.

**Click on:** `student_template.ipynb`

**Run first cell:** (Press Shift+Enter)
```python
import pandas as pd
import numpy as np
import tensorflow as tf
```

**If no errors → You're ready!** ✅

**If errors → Screenshot and email instructor immediately!** ⚠️

---

## Understanding the Technologies

### 🐍 Python
**What:** Programming language  
**Why:** Easy to learn, powerful for data science  
**Used for:** Everything in this workshop!

### 📊 pandas
**What:** Data manipulation library  
**Why:** Excel-like operations in Python  
**Example:** Loading CSV files, filtering data, calculations  
**Website:** https://pandas.pydata.org/

### 🔢 NumPy
**What:** Numerical computing library  
**Why:** Fast mathematical operations on arrays  
**Example:** Matrix multiplication, statistical functions  
**Website:** https://numpy.org/

### 🧠 TensorFlow
**What:** Machine learning framework  
**Why:** Build and train neural networks  
**Created by:** Google  
**Website:** https://www.tensorflow.org/

**Key Concepts:**
- **Tensor:** Multi-dimensional array (like a matrix)
- **Model:** The neural network architecture
- **Training:** Teaching the model from data
- **Prediction:** Using trained model on new data

### 📡 MQTT (Message Queuing Telemetry Transport)
**What:** IoT communication protocol  
**Why:** Real-time data streaming from sensors  
**How it works:**

```
Sensor/Device (Publisher)
        ↓
    MQTT Broker (Server)
        ↓
Your Computer (Subscriber)
```

**Analogy:** Like YouTube
- Publisher = Content creator
- Broker = YouTube servers
- Subscriber = You watching videos
- Topic = Channel name

**In our workshop:**
- Publisher: Instructor's laptop (simulates bus GPS)
- Broker: Central server (forwards data)
- Subscriber: Your laptop (receives GPS data)
- Topic: "agadir/workshop/gps"

### 📓 Jupyter Notebook
**What:** Interactive coding environment  
**Why:** Write code, see results immediately  
**Features:**
- Code cells (write and run Python)
- Markdown cells (write explanations)
- Outputs (graphs, tables, results)

**How to use:**
- **Run cell:** Shift+Enter
- **Add cell:** Press "+"
- **Save:** Ctrl+S

---

## The Mathematics Behind Machine Learning

### 🎯 The Problem: Regression

**Goal:** Predict a continuous number (arrival time in seconds)

**Mathematical formulation:**

Given features **x** = [speed, distance, hour, traffic, ...]

Find function **f** such that:

**ŷ = f(x)**

Where:
- **x** = input features (what we know)
- **ŷ** = predicted output (arrival time)
- **f** = our model (neural network)

---

### 🧮 Neural Networks: The Mathematics

#### Single Neuron

A neuron performs this calculation:

**z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b**

**y = σ(z)**

Where:
- **xᵢ** = input features
- **wᵢ** = weights (learned parameters)
- **b** = bias (intercept)
- **z** = weighted sum
- **σ** = activation function
- **y** = output

**In matrix form:**

**z = Wx + b**

**y = σ(z)**

---

#### Activation Functions

**Why needed?** Without activation, neural network = linear regression (limited!)

**Common activations:**

**1. ReLU (Rectified Linear Unit):**

σ(z) = max(0, z) = {
  z   if z > 0
  0   if z ≤ 0
}

**Why use it?**
- Simple, fast to compute
- Helps with "vanishing gradient" problem
- Most popular for hidden layers

**2. Linear (for output layer):**

σ(z) = z

**Why use it?**
- For regression (predicting any real number)
- No constraints on output range

---

#### Multi-Layer Network

**Our architecture:**

```
Input Layer (8 neurons)
    ↓
Hidden Layer 1 (64 neurons, ReLU)
    ↓
Dropout (20% - prevents overfitting)
    ↓
Hidden Layer 2 (32 neurons, ReLU)
    ↓
Dropout (20%)
    ↓
Hidden Layer 3 (16 neurons, ReLU)
    ↓
Output Layer (1 neuron, Linear)
```

**Mathematical representation:**

**h₁ = ReLU(W₁x + b₁)**  
**h₂ = ReLU(W₂h₁ + b₂)**  
**h₃ = ReLU(W₃h₂ + b₃)**  
**ŷ = W₄h₃ + b₄**

Where:
- **x** = input (8 features)
- **h₁, h₂, h₃** = hidden layer activations
- **W₁, W₂, W₃, W₄** = weight matrices
- **b₁, b₂, b₃, b₄** = bias vectors
- **ŷ** = predicted time remaining (seconds)

---

#### Loss Function: Mean Squared Error (MSE)

**Goal:** Measure how wrong our predictions are

**Formula:**

**MSE = (1/n) Σᵢ₌₁ⁿ (yᵢ - ŷᵢ)²**

Where:
- **n** = number of samples
- **yᵢ** = actual value (true arrival time)
- **ŷᵢ** = predicted value (our prediction)

**Why square the difference?**
- Makes errors positive (10 sec early = 10 sec late in penalty)
- Punishes large errors more (100 sec error = 10,000 penalty)
- Mathematically convenient for calculus

**Mean Absolute Error (MAE):** (also used)

**MAE = (1/n) Σᵢ₌₁ⁿ |yᵢ - ŷᵢ|**

- More interpretable (average seconds off)
- Less sensitive to outliers

---

#### Training: Gradient Descent

**Goal:** Find weights **W** and biases **b** that minimize loss

**Algorithm:**

1. Initialize weights randomly
2. Repeat until converged:
   - Compute predictions: **ŷ = f(x; W, b)**
   - Compute loss: **L = MSE(y, ŷ)**
   - Compute gradients: **∂L/∂W, ∂L/∂b**
   - Update weights: **W = W - α(∂L/∂W)**
   - Update biases: **b = b - α(∂L/∂b)**

Where:
- **α** = learning rate (step size, typically 0.001)
- **∂L/∂W** = how much to change W to reduce L
- **∂L/∂b** = how much to change b to reduce L

**Gradient:** Direction of steepest increase

**Negative gradient:** Direction of steepest decrease (we want this!)

**Intuition:**
- Imagine you're in a hilly landscape (loss surface)
- You want to reach the lowest valley (minimum loss)
- Gradient tells you which direction is downhill
- Learning rate tells you how big a step to take

---

#### Backpropagation

**What:** Algorithm to compute gradients efficiently

**Chain rule:** (from calculus)

If **y = f(g(x))**, then:

**dy/dx = (df/dg) × (dg/dx)**

**In neural networks:**

For output layer:
**∂L/∂W₄ = ∂L/∂ŷ × ∂ŷ/∂W₄**

For hidden layers (chain rule):
**∂L/∂W₃ = ∂L/∂ŷ × ∂ŷ/∂h₃ × ∂h₃/∂W₃**

TensorFlow does this automatically! 🎉

---

#### Dropout: Regularization

**Problem:** Model memorizes training data (overfitting)

**Solution:** Randomly "drop" 20% of neurons during training

**Mathematics:**

During training:
**h̃ = h ⊙ m**

Where:
- **m** = random binary mask (0 or 1 for each neuron)
- **P(mᵢ = 1) = 0.8** (keep 80%, drop 20%)
- **⊙** = element-wise multiplication

**Why it works:**
- Forces network to not rely on any single neuron
- Creates ensemble of sub-networks
- Reduces overfitting

---

### 📊 Feature Scaling: Normalization

**Problem:** Features have different scales
- Speed: 30-65 km/h
- Distance: 0-31 km
- Hour: 0-23

**Solution:** Standardization (Z-score normalization)

**Formula:**

**x' = (x - μ) / σ**

Where:
- **x** = original feature value
- **μ** = mean of feature
- **σ** = standard deviation of feature
- **x'** = normalized value (mean=0, std=1)

**Why important?**
- Gradient descent converges faster
- All features contribute equally
- Prevents numerical instability

**Example:**

Original speeds: [30, 40, 50, 60] km/h

Mean: μ = 45
Std: σ = 12.9

Normalized: [-1.16, -0.39, 0.39, 1.16]

---

### 🎲 Train/Test Split

**Goal:** Evaluate model on unseen data

**Method:**
- **Training set (80%):** Learn patterns
- **Test set (20%):** Evaluate performance

**Why?**
- Prevents overfitting detection
- Simulates real-world scenario
- Honest performance estimate

**Random split:** Ensures both sets represent population

---

## Step-by-Step Tutorial

### Part 1: Understanding the Data

#### What's in training_dataset.csv?

**100 trajectories:**
- 50 trips: Faculty of Science → Taghazout Village
- 50 trips: Taghazout Village → Faculty of Science

**Each trajectory has ~1000 GPS points:**

| Column | Description | Example | Type |
|--------|-------------|---------|------|
| `trajectory_id` | Unique trip identifier | "T042" | Categorical |
| `direction` | Trip direction | "to_taghazout" | Categorical |
| `point_index` | Point number in trip | 450 | Integer |
| `timestamp` | When GPS recorded | "2026-01-24 14:25:30" | DateTime |
| `lat` | Latitude (GPS) | 30.4250 | Float |
| `lon` | Longitude (GPS) | -9.5850 | Float |
| `speed_kmh` | Current speed | 55 | Integer |
| `distance_covered_km` | Distance from start | 12.5 | Float |
| `distance_remaining_km` | Distance to destination | 19.1 | Float |
| `progress_percent` | Journey completion | 39.5 | Float |
| `passengers` | People on bus | 23 | Integer |
| `hour_of_day` | Hour (0-23) | 14 | Integer |
| `is_rush_hour` | Rush hour flag | 0 or 1 | Binary |
| `is_weekend` | Weekend flag | 0 or 1 | Binary |
| `weather` | Weather condition | "sunny" | Categorical |
| `traffic_level` | Traffic condition | "moderate" | Categorical |

**Total:** ~100,000 rows of data!

---

#### Feature Types

**1. Continuous Features:**
- Speed, distances, progress
- Can take any value in a range
- **Use directly** in model

**2. Categorical Features:**
- Weather: sunny, cloudy, rainy, windy
- Traffic: light, moderate, heavy
- **Must encode** as numbers

**Encoding method: Label Encoding**
- sunny = 0, cloudy = 1, rainy = 2, windy = 3
- light = 0, moderate = 1, heavy = 2

**3. Binary Features:**
- is_rush_hour: 0 or 1
- is_weekend: 0 or 1
- Already numbers!

---

#### Target Variable: What We're Predicting

**We want to predict:** Time remaining until arrival (seconds)

**How to calculate it?**

For each GPS point, we need:
- **Time at this point:** tₙₒw
- **Time at destination:** tₑₙd
- **Time remaining:** tᵣₑₘₐᵢₙ = tₑₙd - tₙₒw

**From trajectory timestamps:**

If trajectory has points at:
- t₀ = 14:00:00 (start)
- t₅₀₀ = 14:15:30 (current point)
- t₁₀₀₀ = 14:32:15 (end)

Then at point 500:
**time_remaining = t₁₀₀₀ - t₅₀₀ = 14:32:15 - 14:15:30 = 16 min 45 sec = 1005 seconds**

This is our **y** (target)!

---

### Part 2: The Code - Line by Line

#### Cell 1: Import Libraries

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json

# TensorFlow
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# MQTT
import paho.mqtt.client as mqtt

# HTTP for submissions
import requests

print("✅ All libraries imported successfully!")
```

**What each library does:**

- **pandas (pd):** Data manipulation (like Excel in Python)
- **numpy (np):** Fast math on arrays
- **datetime, timedelta:** Work with dates and times
- **time:** Pause execution, timing
- **json:** Handle JSON data format
- **tensorflow, keras:** Build neural networks
- **train_test_split:** Split data into train/test
- **StandardScaler:** Normalize features
- **mqtt:** Connect to MQTT broker
- **requests:** Send HTTP requests

---

#### Cell 2: Configuration

```python
# ============ CONFIGURATION ============
STUDENT_NAME = "Ahmed"  # 🚨 CHANGE THIS TO YOUR NAME!

# MQTT Settings
MQTT_BROKER = "192.168.137.1"  # Instructor's laptop IP
MQTT_PORT = 1883
MQTT_TOPIC_GPS = "agadir/workshop/gps"
MQTT_TOPIC_CONTROL = "agadir/workshop/control"

# Leaderboard Server
LEADERBOARD_URL = "http://192.168.137.1:5000/submit_prediction"

# Submission interval
SUBMIT_INTERVAL = 30  # seconds

print(f"✅ Configuration set for: {STUDENT_NAME}")
```

**What this does:**
- Sets your name (for leaderboard)
- Configures network addresses
- Sets prediction submission frequency

**⚠️ IMPORTANT:** Update IP address from whiteboard on workshop day!

---

#### Cell 3: Load Training Dataset

```python
# Load dataset
df = pd.read_csv('training_dataset.csv')

print(f"📊 Dataset loaded: {len(df)} rows")
print(f"📊 Trajectories: {df['trajectory_id'].nunique()}")
print("\nFirst few rows:")
df.head()
```

**What `pd.read_csv()` does:**
- Reads CSV file into DataFrame (pandas table)
- Automatically detects column types
- Returns DataFrame object

**What `.head()` shows:**
- First 5 rows of data
- Lets you verify data loaded correctly

**Expected output:**
```
📊 Dataset loaded: 98600 rows
📊 Trajectories: 100
```

Plus a table showing first 5 rows.

---

#### Cell 4: Explore Data

```python
# Dataset info
print("Dataset columns:")
print(df.columns.tolist())
print("\nDataset statistics:")
df.describe()
```

**What `.describe()` does:**
Statistical summary of numeric columns:
- **count:** Number of non-null values
- **mean:** Average value
- **std:** Standard deviation (spread)
- **min:** Minimum value
- **25%:** First quartile
- **50%:** Median
- **75%:** Third quartile
- **max:** Maximum value

**Why useful:**
- Understand data ranges
- Detect outliers
- Check for missing values

---

#### Cell 5: Feature Engineering - Create Target

```python
def calculate_remaining_time(group):
    """Calculate remaining time for each point in trajectory"""
    # Sort by point index
    group = group.sort_values('point_index')
    
    # Parse timestamps
    group['timestamp'] = pd.to_datetime(group['timestamp'])
    
    # Calculate total trajectory time
    start_time = group['timestamp'].iloc[0]
    end_time = group['timestamp'].iloc[-1]
    total_seconds = (end_time - start_time).total_seconds()
    
    # Calculate remaining time for each point
    remaining_times = []
    for idx, row in group.iterrows():
        current_time = row['timestamp']
        time_elapsed = (current_time - start_time).total_seconds()
        time_remaining = total_seconds - time_elapsed
        remaining_times.append(time_remaining)
    
    group['time_remaining_seconds'] = remaining_times
    return group

# Apply to each trajectory
df = df.groupby('trajectory_id').apply(calculate_remaining_time).reset_index(drop=True)

print("✅ Target variable created: time_remaining_seconds")
print(f"Range: {df['time_remaining_seconds'].min():.0f}s to {df['time_remaining_seconds'].max():.0f}s")
```

**Mathematics:**

For trajectory with n points:
- **t₀** = start time
- **tₙ** = end time
- **T** = total time = tₙ - t₀

For point i at time tᵢ:
- **Elapsed:** eᵢ = tᵢ - t₀
- **Remaining:** rᵢ = T - eᵢ = tₙ - tᵢ

**Why `.groupby()`?**
- Processes each trajectory separately
- Maintains temporal order
- Applies function to each group

---

#### Cell 6: Prepare Features

```python
# Select features for model
feature_columns = [
    'speed_kmh',              # How fast moving now
    'distance_remaining_km',   # How far to go
    'progress_percent',        # How much completed
    'hour_of_day',            # Time of day
    'is_rush_hour',           # Rush hour indicator
    'is_weekend'              # Weekend indicator
]

# Encode categorical features
df['weather_encoded'] = df['weather'].astype('category').cat.codes
df['traffic_encoded'] = df['traffic_level'].astype('category').cat.codes

# Add encoded features
feature_columns.extend(['weather_encoded', 'traffic_encoded'])

# Create feature matrix X and target vector y
X = df[feature_columns].values
y = df['time_remaining_seconds'].values

print(f"✅ Features: {len(feature_columns)}")
print(f"✅ Samples: {len(X)}")
print(f"\nFeatures used: {feature_columns}")
```

**What `.astype('category').cat.codes` does:**

Original:
```
weather: ['sunny', 'cloudy', 'rainy', 'sunny', 'windy']
```

Encoded:
```
weather_encoded: [0, 1, 2, 0, 3]
```

Mapping:
- sunny → 0
- cloudy → 1
- rainy → 2
- windy → 3

**Why `.values`?**
- Converts pandas DataFrame to numpy array
- TensorFlow works with numpy arrays
- Faster computation

**Result:**
- **X:** (98600, 8) array - 98600 samples, 8 features
- **y:** (98600,) array - 98600 target values

---

#### Cell 7: Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,    # 20% for testing
    random_state=42   # Reproducible results
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
```

**What `train_test_split()` does:**

Original dataset (100%):
```
[sample1, sample2, ..., sample98600]
```

After split:
- **Training (80%):** [sample1, sample5, sample9, ...] → 78,880 samples
- **Testing (20%):** [sample2, sample3, sample6, ...] → 19,720 samples

**random_state=42:**
- Sets random seed
- Same split every time you run
- Reproducible results

**Why 80/20 split?**
- Standard practice
- Enough data for training
- Enough data for reliable testing

---

#### Cell 8: Feature Normalization

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Features normalized")
```

**What `StandardScaler` does:**

**Training phase:** (`.fit_transform()`)
1. Calculate mean μ and std σ from X_train
2. Transform: **x' = (x - μ) / σ**

**Testing phase:** (`.transform()`)
1. Use same μ and σ from training
2. Transform: **x' = (x - μ) / σ**

**⚠️ CRITICAL:** Never fit on test data!
- Would leak information
- Overly optimistic results
- Not representative of real-world

**Example:**

Before scaling:
```
speed: [30, 45, 60, 35, 50]
```

After scaling (μ=44, σ=11.4):
```
speed_scaled: [-1.23, 0.09, 1.40, -0.79, 0.53]
```

All features now have mean ≈ 0, std ≈ 1

---

#### Cell 9: Build Neural Network

```python
# Build neural network
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(len(feature_columns),)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1)  # Output: time remaining in seconds
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

print("✅ Model created")
model.summary()
```

**Layer-by-layer explanation:**

**Layer 1: Dense(64, activation='relu', input_shape=(8,))**
- **64 neurons:** Each computes z = Wx + b, then y = ReLU(z)
- **Input shape:** 8 features
- **Parameters:** (8 × 64) + 64 = 576
  - Weights: 8 × 64 = 512
  - Biases: 64

**Layer 2: Dropout(0.2)**
- **Randomly drops 20% of neurons** during training
- **No parameters** to learn
- **Prevents overfitting**

**Layer 3: Dense(32, activation='relu')**
- **32 neurons**
- **Parameters:** (64 × 32) + 32 = 2,080
- Takes output from Layer 1 (64 values)

**Layer 4: Dropout(0.2)**
- Same as Layer 2

**Layer 5: Dense(16, activation='relu')**
- **16 neurons**
- **Parameters:** (32 × 16) + 16 = 528

**Layer 6: Dense(1)**
- **1 neuron:** Final prediction
- **No activation** (linear output)
- **Parameters:** (16 × 1) + 1 = 17
- **Output:** Time remaining in seconds (can be any value)

**Total parameters:** 576 + 2,080 + 528 + 17 = **3,201 parameters to learn!**

---

**Optimizer: Adam**

**What it does:** Smart gradient descent
- Adaptive learning rate
- Momentum (considers past gradients)
- Fast convergence

**Update rule:**

**mₜ = β₁mₜ₋₁ + (1-β₁)gₜ**  
**vₜ = β₂vₜ₋₁ + (1-β₂)gₜ²**  
**m̂ₜ = mₜ/(1-β₁ᵗ)**  
**v̂ₜ = vₜ/(1-β₂ᵗ)**  
**θₜ = θₜ₋₁ - α·m̂ₜ/(√v̂ₜ + ε)**

Where:
- **gₜ** = gradient
- **mₜ** = first moment (mean)
- **vₜ** = second moment (variance)
- **β₁, β₂** = decay rates (default: 0.9, 0.999)
- **α** = learning rate (default: 0.001)

Don't worry about details - Adam just works! ✨

---

**Loss: MSE (Mean Squared Error)**

**L = (1/n) Σ(yᵢ - ŷᵢ)²**

**Why MSE?**
- Differentiable (needed for gradient descent)
- Penalizes large errors heavily
- Standard for regression

---

**Metric: MAE (Mean Absolute Error)**

**MAE = (1/n) Σ|yᵢ - ŷᵢ|**

**Why MAE?**
- Interpretable (average seconds off)
- What we care about in competition!
- Not used for training, only monitoring

---

#### Cell 10: Train Model

```python
print("🔄 Training model...")

history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,    # Use 20% of training for validation
    epochs=50,                # Train for 50 iterations
    batch_size=32,           # Process 32 samples at a time
    verbose=1                 # Show progress
)

print("\n✅ Training complete!")
```

**What happens during training:**

**Epoch 1:**
1. **Forward pass:** Compute predictions for all training data
2. **Compute loss:** MSE(y_train, predictions)
3. **Backward pass:** Compute gradients using backpropagation
4. **Update weights:** W = W - α·∇W (gradient descent)
5. **Validation:** Test on validation set (no weight updates)
6. **Print:** Training loss, validation loss, MAE

**Epochs 2-50:** Repeat above

**Batch size = 32:**
- Don't process all 78,880 samples at once (too slow)
- Process in batches of 32
- Update weights after each batch
- Number of updates per epoch: 78,880/32 ≈ 2,465

**Validation split = 0.2:**
- Use 80% of training (63,104 samples) for actual training
- Use 20% of training (15,776 samples) for validation
- Validation monitors overfitting
- If validation loss increases → overfitting!

**Expected output:**
```
Epoch 1/50
2465/2465 [======] - 8s - loss: 15234.5 - mae: 98.3 - val_loss: 12456.2 - val_mae: 89.1
Epoch 2/50
2465/2465 [======] - 7s - loss: 10123.4 - mae: 80.5 - val_loss: 9876.3 - val_mae: 79.2
...
Epoch 50/50
2465/2465 [======] - 7s - loss: 2345.6 - mae: 38.7 - val_loss: 2456.8 - val_mae: 39.2
```

**What to look for:**
- **Loss decreasing:** Model learning! ✅
- **MAE decreasing:** Getting more accurate! ✅
- **Val_loss close to loss:** Not overfitting! ✅

---

#### Cell 11: Evaluate Model

```python
test_loss, test_mae = model.evaluate(X_test_scaled, y_test)

print(f"\n📊 Test MAE: {test_mae:.2f} seconds")
print(f"📊 This means your predictions are off by ~{test_mae:.0f} seconds on average")
```

**What `.evaluate()` does:**
- Runs model on test set (never seen during training!)
- Computes loss and MAE
- Shows real-world performance

**Interpreting results:**

If **test_mae = 42.5 seconds:**
- On average, predictions are 42.5 seconds from true arrival
- For 30 minute journey: 42.5/1800 ≈ 2.4% error
- Pretty good! 🎉

If **test_mae = 150 seconds:**
- Average error: 2.5 minutes
- For 30 minute journey: 8.3% error
- Room for improvement! 🔧

**Goal:** Minimize this number!

---

#### Cell 12: Real-Time Prediction System

```python
# Global variables
current_trajectory_id = None
last_gps_data = None
last_submission_time = None

def predict_arrival_time(gps_data):
    """Predict arrival time from GPS data"""
    try:
        # Extract features (same order as training!)
        features = [
            gps_data['speed_kmh'],
            gps_data['distance_remaining_km'],
            gps_data['progress_percent'],
            datetime.now().hour,  # Current hour
            0,  # is_rush_hour (you can improve this)
            0,  # is_weekend (you can improve this)
            0,  # weather_encoded (default to sunny)
            1   # traffic_encoded (default to moderate)
        ]
        
        # Scale features (using same scaler from training!)
        features_scaled = scaler.transform([features])
        
        # Predict remaining seconds
        remaining_seconds = model.predict(features_scaled, verbose=0)[0][0]
        
        # Calculate arrival time
        arrival_time = datetime.now() + timedelta(seconds=float(remaining_seconds))
        
        return arrival_time
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None
```

**Step-by-step:**

**1. Extract features from GPS data:**
```python
features = [55, 19.1, 39.5, 14, 0, 0, 0, 1]
```
Must be **same order** as training!

**2. Scale features:**
```python
features_scaled = scaler.transform([features])
```
Uses μ and σ from training

**3. Predict:**
```python
remaining_seconds = model.predict(features_scaled)[0][0]
# Example: 1245.6 seconds
```

**4. Calculate arrival time:**
```python
now = 14:25:30
remaining = 1245.6 seconds ≈ 20 minutes 45 seconds
arrival = 14:25:30 + 20:45 = 14:46:15
```

**Return:** 14:46:15 (ISO format)

---

#### Cell 12 (continued): Submit Prediction

```python
def submit_prediction(trajectory_id, arrival_time):
    """Submit prediction to leaderboard"""
    try:
        # Prepare data
        data = {
            'student_name': STUDENT_NAME,
            'trajectory_id': trajectory_id,
            'predicted_arrival_time': arrival_time.isoformat(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Send HTTP POST request
        response = requests.post(LEADERBOARD_URL, json=data, timeout=5)
        
        if response.status_code == 200:
            print(f"✅ Prediction submitted: {arrival_time.strftime('%H:%M:%S')}")
        else:
            print(f"⚠️ Submission failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Submission error: {e}")
```

**What happens:**

**1. Create JSON data:**
```json
{
  "student_name": "Ahmed",
  "trajectory_id": "R001",
  "predicted_arrival_time": "2026-01-24T14:46:15",
  "timestamp": "2026-01-24T14:25:30"
}
```

**2. Send HTTP POST:**
```
POST http://192.168.137.1:5000/submit_prediction
Content-Type: application/json
Body: <JSON data above>
```

**3. Leaderboard receives:**
- Stores your prediction
- Waits for bus to arrive
- Calculates error when round ends
- Updates rankings

---

#### Cell 12 (continued): MQTT Callbacks

```python
def on_connect(client, userdata, flags, rc):
    """Called when connected to MQTT broker"""
    if rc == 0:
        print("✅ Connected to MQTT broker")
        client.subscribe(MQTT_TOPIC_GPS)
        client.subscribe(MQTT_TOPIC_CONTROL)
    else:
        print(f"❌ Connection failed: {rc}")

def on_message(client, userdata, msg):
    """Called when a new message arrives"""
    global current_trajectory_id, last_gps_data, last_submission_time
    
    try:
        # Parse JSON data
        data = json.loads(msg.payload.decode())
        
        # Handle control messages
        if msg.topic == MQTT_TOPIC_CONTROL:
            if data.get('type') == 'ROUND_START':
                current_trajectory_id = data['trajectory_id']
                last_submission_time = None
                print(f"\n🚌 NEW ROUND: {current_trajectory_id}")
                print("="*50)
            elif data.get('type') == 'ARRIVED':
                print(f"\n🏁 ROUND COMPLETE!")
                print(f"Actual arrival: {data['actual_arrival_time']}")
                print("="*50)
            return
        
        # Handle GPS messages
        if msg.topic == MQTT_TOPIC_GPS:
            last_gps_data = data
            
            # Check if we should submit (every 30 seconds)
            now = time.time()
            if last_submission_time is None or (now - last_submission_time) >= SUBMIT_INTERVAL:
                
                # Make prediction
                arrival_time = predict_arrival_time(data)
                
                if arrival_time:
                    # Submit to leaderboard
                    submit_prediction(current_trajectory_id, arrival_time)
                    last_submission_time = now
                    
                    # Print status
                    print(f"📍 Position: {data['progress_percent']:.1f}% | "
                          f"Speed: {data['speed_kmh']} km/h | "
                          f"Remaining: {data['distance_remaining_km']:.1f} km")
                    
    except Exception as e:
        print(f"❌ Error: {e}")

print("✅ Prediction system ready!")
```

**How MQTT callbacks work:**

**on_connect:** Called once when connection established
- Subscribe to topics
- Topics are like "channels"
- We subscribe to 2 topics:
  - `agadir/workshop/gps` (GPS data)
  - `agadir/workshop/control` (round start/end)

**on_message:** Called every time message received
- Parses JSON data
- Checks message type:
  - Control: Round start/end
  - GPS: Real-time position
- Every 30 seconds: Make prediction and submit

---

#### Cell 13: Start Competition!

```python
print("="*70)
print("🏆 STARTING COMPETITION!")
print("="*70)
print(f"Student: {STUDENT_NAME}")
print(f"MQTT Broker: {MQTT_BROKER}")
print(f"Leaderboard: {LEADERBOARD_URL}")
print(f"Submission interval: {SUBMIT_INTERVAL} seconds")
print("="*70)
print("\nWaiting for instructor to start rounds...")
print("Press STOP button to disconnect\n")

# Create MQTT client
client = mqtt.Client(client_id=f"student_{STUDENT_NAME}")
client.on_connect = on_connect
client.on_message = on_message

try:
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_forever()  # Keep listening
except KeyboardInterrupt:
    print("\n👋 Disconnecting...")
    client.disconnect()
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\n💡 Make sure:")
    print("  1. MQTT_BROKER is correct")
    print("  2. Instructor has started streaming")
    print("  3. You have internet connection")
```

**What `.loop_forever()` does:**
- Keeps program running
- Listens for MQTT messages
- Calls on_message() when data arrives
- Runs until you press STOP

**Flow:**
1. Connect to MQTT broker
2. Subscribe to topics
3. Wait for messages
4. When GPS arrives → Process → Predict → Submit
5. Repeat every 30 seconds
6. Continue until round ends

---

### Part 3: Improving Your Model

**If your predictions aren't accurate enough, try:**

#### 1. Add More Features

```python
# Add speed change (acceleration)
df['speed_change'] = df.groupby('trajectory_id')['speed_kmh'].diff()

# Add distance per minute
df['distance_per_min'] = df['distance_covered_km'] / (df['point_index'] + 1)
```

#### 2. Engineer Time Features

```python
# Is it morning/afternoon/evening?
df['time_of_day'] = pd.cut(df['hour_of_day'], 
                           bins=[0, 6, 12, 18, 24],
                           labels=['night', 'morning', 'afternoon', 'evening'])

# Encode
df['time_of_day_encoded'] = df['time_of_day'].astype('category').cat.codes
```

#### 3. Try Different Architecture

```python
# Deeper network
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(n_features,)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1)
])
```

#### 4. Train Longer

```python
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,  # More epochs
    batch_size=64,  # Larger batches
    verbose=1
)
```

#### 5. Use Different Activation

```python
# Try LeakyReLU instead of ReLU
model = keras.Sequential([
    keras.layers.Dense(64, activation='leaky_relu'),
    ...
])
```

#### 6. Ensemble Methods

Train multiple models and average predictions:

```python
pred1 = model1.predict(features)
pred2 = model2.predict(features)
pred3 = model3.predict(features)

final_pred = (pred1 + pred2 + pred3) / 3
```

---

## Workshop Day Instructions

### 1. Connect to WiFi (5 minutes)

**Network:** AIoT_Workshop  
**Password:** workshop2024

Check whiteboard for exact network name!

---

### 2. Update Your Notebook (5 minutes)

Open `student_template.ipynb`

Find Cell 2 (Configuration), update:

```python
STUDENT_NAME = "YourActualName"  # Not "Ahmed"!
MQTT_BROKER = "192.168.137.1"    # Check whiteboard!
LEADERBOARD_URL = "http://192.168.137.1:5000/submit_prediction"  # Check whiteboard!
```

**Save:** Ctrl+S

---

### 3. Run Training Cells (20 minutes)

**Run cells 1-11 sequentially** (Shift+Enter for each)

**Cell 1:** Imports (should be instant)  
**Cell 2:** Config (instant)  
**Cell 3:** Load data (5 seconds)  
**Cell 4:** Explore (instant)  
**Cell 5:** Feature engineering (10 seconds)  
**Cell 6:** Prepare features (instant)  
**Cell 7:** Split data (instant)  
**Cell 8:** Normalize (instant)  
**Cell 9:** Build model (instant)  
**Cell 10:** Train model **(5-10 minutes)** ⏳  
**Cell 11:** Evaluate (instant)

**During Cell 10 (training):**
- Watch loss decrease
- Check validation loss
- If train loss << val loss → overfitting!

---

### 4. Start Competition (Cell 13)

**When instructor says "Start":**

Run Cell 13 (last cell)

**You'll see:**
```
✅ Connected to MQTT broker
📡 Subscribed to agadir/workshop/gps
📡 Subscribed to agadir/workshop/control

Waiting for instructor to start rounds...
```

**When Round 1 starts:**
```
🚌 NEW ROUND: R001
==================================================
📍 Position: 5.2% | Speed: 35 km/h | Remaining: 29.8 km
✅ Prediction submitted: 14:35:22
📍 Position: 8.4% | Speed: 42 km/h | Remaining: 28.5 km
✅ Prediction submitted: 14:34:18
...
```

**Keep this cell running!** Don't stop it.

**When round ends:**
```
🏁 ROUND COMPLETE!
Actual arrival: 14:32:05
==================================================
```

Check projected screen to see your ranking!

---

### 5. Between Rounds (2 minutes)

**Option A:** Keep same model (no changes)

**Option B:** Improve model
1. **Stop Cell 13** (Stop button)
2. Modify architecture (Cell 9)
3. Retrain (Cell 10)
4. **Restart Cell 13**

**Pro tip:** Try Option B only if you're losing badly! Otherwise, stick with working model.

---

### 6. Final Round

After Round 10:
- Check final leaderboard
- See who won!
- Congratulate winner 🎉

---

## Troubleshooting

### Problem: "Module not found" errors

**Solution:**
```cmd
pip install <module_name>
```

Example:
```cmd
pip install tensorflow
```

---

### Problem: "Can't connect to MQTT broker"

**Check:**
1. Are you connected to AIoT_Workshop WiFi?
2. Is MQTT_BROKER IP correct? (check whiteboard)
3. Did instructor start streaming?

**Test connection:**
```python
import socket
socket.create_connection(("192.168.137.1", 1883), timeout=5)
```

If fails → Network problem

---

### Problem: "Predictions not appearing on leaderboard"

**Check:**
1. Is your name unique? (no duplicates)
2. Is LEADERBOARD_URL correct?
3. Is Cell 13 running?

**Test manually:**
```python
import requests
response = requests.get("http://192.168.137.1:5000")
print(response.status_code)  # Should be 200
```

---

### Problem: Training very slow

**Solutions:**
1. Reduce epochs: `epochs=20`
2. Increase batch size: `batch_size=128`
3. Use fewer layers
4. Use fewer neurons per layer

---

### Problem: Model predicts same time for everything

**Causes:**
- Not enough training
- Features not normalized
- Model too simple
- Learning rate too high

**Solutions:**
- Train more epochs
- Check scaler applied
- Add more layers
- Use default optimizer

---

### Problem: High training accuracy, low test accuracy

**Diagnosis:** Overfitting!

**Solutions:**
- Increase dropout: `Dropout(0.4)`
- Add more dropout layers
- Reduce model size (fewer neurons)
- Get more training data (use all)
- Early stopping

---

## Additional Resources

### Learn More About:

**Python:**
- https://www.python.org/about/gettingstarted/
- https://docs.python.org/3/tutorial/

**pandas:**
- https://pandas.pydata.org/docs/getting_started/index.html
- https://www.kaggle.com/learn/pandas

**NumPy:**
- https://numpy.org/doc/stable/user/quickstart.html

**TensorFlow:**
- https://www.tensorflow.org/tutorials
- https://www.tensorflow.org/guide/keras

**Neural Networks:**
- http://neuralnetworksanddeeplearning.com/
- https://www.3blue1brown.com/topics/neural-networks

**MQTT:**
- https://mqtt.org/
- https://www.hivemq.com/mqtt-essentials/

### Books:

- "Hands-On Machine Learning" by Aurélien Géron
- "Deep Learning" by Ian Goodfellow
- "Python for Data Analysis" by Wes McKinney

---

## Glossary

**Activation Function:** Non-linear function applied to neuron output

**Backpropagation:** Algorithm to compute gradients for neural networks

**Batch:** Subset of training data processed together

**Bias:** Intercept term in linear equation (b in y = Wx + b)

**DataFrame:** pandas table structure (rows and columns)

**Dropout:** Regularization technique that randomly disables neurons

**Epoch:** One complete pass through training data

**Feature:** Input variable (column in dataset)

**Gradient:** Derivative showing direction of steepest increase

**Gradient Descent:** Optimization algorithm for finding minimum loss

**IoT:** Internet of Things (connected devices with sensors)

**Loss Function:** Measures how wrong predictions are

**MQTT:** Lightweight messaging protocol for IoT

**Neural Network:** Computational model inspired by biological neurons

**Normalization:** Scaling features to standard range

**Overfitting:** Model memorizes training data, poor on test data

**Regression:** Predicting continuous values (not categories)

**Training:** Process of learning optimal weights from data

**Validation Set:** Data used to tune model (not for training or testing)

**Weight:** Learned parameter in neural network (W in y = Wx + b)

---

## Good Luck! 🚀

You now have everything you need:
- ✅ Understanding of concepts
- ✅ Mathematical foundations
- ✅ Complete code with explanations
- ✅ Workshop day instructions
- ✅ Troubleshooting guide

**Remember:**
- Come prepared (software installed)
- Start simple (basic model first)
- Iterate (improve between rounds)
- Have fun! (it's a learning experience)

**May the best predictor win!** 🏆

---

*For questions during workshop: Ask your instructor or neighbors!*

*After workshop: Email instructor or check additional resources.*
