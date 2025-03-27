# **LSTM Sensor Prediction 🚀**  
A **deep learning-based motion prediction system** using **Long Short-Term Memory (LSTM) networks** to predict future sensor readings from **treadmill walking data**.  

---

## 📌 **Project Overview**  
This project aims to develop an **LSTM-based time series prediction model** that forecasts **future motion sensor readings** (gyroscope & accelerometer) based on past sensor values from volunteers walking on a treadmill at different speeds.  

### **Key Highlights**  
✅ **Uses LSTM networks** to model sequential dependencies in motion sensor data  
✅ **Predicts future sensor readings** (accelerometer & gyroscope in x, y, z directions)  
✅ **Handles real-time walking motion** for possible biomechanical analysis  
✅ **Uses TensorFlow/Keras for deep learning** & supports **GPU acceleration**  

### **Why Use LSTM?**  
LSTM networks are ideal for time-series forecasting because they can:  
- Remember long-term dependencies in sequential data  
- Handle complex patterns in motion sensor readings  
- Predict **future motion states** based on historical sensor data  

---

## 📂 **Dataset Description**  
The dataset (`Final_data_cleaned.csv`) consists of **motion sensor readings** collected from **10 volunteers walking at different speeds (1, 2, 3, and 4 km/hr) on a treadmill**.  

### **📊 Input Features (Independent Variables)**  
These are the values used to train the model:  
- **Speed_km_hr**: Walking speed of the volunteer  
- **Volunteer ID**: Identifies each participant  
- **Thigh & Ankle Sensor Readings**:  
  - Accelerometer (`ax, ay, az`): Measures linear acceleration in x, y, z axes  
  - Gyroscope (`wx, wy, wz`): Measures rotational velocity in x, y, z axes  

### **🎯 Output Targets (Dependent Variables)**  
The model predicts **future sensor values** for:  
- **Thigh Sensor (Accelerometer & Gyroscope):** `Thigh_ax, Thigh_ay, Thigh_az, Thigh_wx, Thigh_wy, Thigh_wz`  
- **Ankle Sensor (Accelerometer & Gyroscope):** `Ankle_ax, Ankle_ay, Ankle_az, Ankle_wx, Ankle_wy, Ankle_wz`  

---

## ⚙️ **Installation & Setup**  
### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/your-username/LSTM-Sensor-Prediction.git
cd LSTM-Sensor-Prediction
```

### **2️⃣ Install Dependencies**  
Ensure you have Python **3.8+** and install required libraries:  
```bash
pip install -r requirements.txt
```

### **3️⃣ Check GPU Availability** (Optional)  
If using a **GPU**, verify TensorFlow detects it:  
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```
If no GPU is found, install **CUDA & cuDNN** compatible with your TensorFlow version.

---

## 🚀 **Model Architecture**  
The model uses **three stacked LSTM layers** with **Dropout regularization** to prevent overfitting.  

### **🔍 LSTM Network Structure**  
| Layer | Type | Neurons | Activation | Additional Info |
|-------|------|---------|------------|----------------|
| 1 | LSTM | 256 | tanh | return sequences = True |
| 2 | Dropout | - | - | 30% dropout |
| 3 | LSTM | 128 | tanh | return sequences = True |
| 4 | Dropout | - | - | 30% dropout |
| 5 | LSTM | 64 | tanh | return sequences = False |
| 6 | Dropout | - | - | 30% dropout |
| 7 | Dense | 64 | relu | Fully connected layer |
| 8 | Dense | 12 | Linear | Output layer (12 target values) |

The **sequence length** is set to **30 time steps**, meaning the model uses the last **30 frames of sensor data** to predict the next frame.

---

## 🎯 **Training the Model**  
Run the Jupyter Notebook:  
```bash
jupyter notebook lstm_training.ipynb
```
- Uses **Adam optimizer** and **MSE loss**  
- **Early Stopping & ReduceLROnPlateau** help avoid overfitting  
- Saves the trained model as **`lstm_sensor_model.h5`**  

### **Hyperparameters Used**  
- **Batch Size**: `32`  
- **Epochs**: `100` (early stopping applied)  
- **Learning Rate Adjustment**: Reduce LR if validation loss stagnates  
- **Validation Split**: 20%  

---

## 📊 **Model Evaluation Metrics**  
After training, the model is evaluated using:  
1️⃣ **Mean Squared Error (MSE)** – Measures average squared difference between predicted & actual values  
2️⃣ **Mean Absolute Error (MAE)** – Measures absolute difference between predicted & actual values  
3️⃣ **R² Score (R-Squared)** – Measures goodness of fit  

**Example results after training:**  
```
Mean Squared Error (MSE): 2.2369
Mean Absolute Error (MAE): 0.8122
R² Score: 0.7688
```

---

## 📈 **Results Visualization**  
The model's **actual vs predicted** values are plotted for **all 12 target variables**.  

```python
plt.figure(figsize=(15, 12))
for i in range(len(targets)):
    plt.subplot(4, 3, i + 1)
    plt.plot(Y_test_actual[:100, i], label=f"Actual {targets[i]}", color='blue')
    plt.plot(Y_pred_actual[:100, i], label=f"Predicted {targets[i]}", color='red', linestyle='dashed')
    plt.xlabel("Sample Index")
    plt.ylabel(f"{targets[i]} Value")
    plt.legend()
    plt.title(f"Actual vs Predicted {targets[i]}")
plt.tight_layout()
plt.show()
```

---

## 🔥 **Using the Pretrained Model**  
Once trained, the model can be used to **predict future motion states** in real-time.  

### **Load and Use the Model for Predictions**
```python
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained model
model = load_model("lstm_sensor_model.h5")

# Predict on new data
new_input = np.array(X_test[:1])  # Example input
predicted_output = model.predict(new_input)
print(predicted_output)
```

---

## 🎯 **Real-World Applications**  
This project has potential applications in:  
✅ **Biomechanics Research** – Analyzing human walking patterns  
✅ **Sports Science** – Predicting athlete movement dynamics  
✅ **Rehabilitation & Injury Prevention** – Detecting gait abnormalities  
✅ **Robotics & Wearables** – Enhancing sensor-based motion tracking  

---

## 🛠 **Future Improvements**  
🔹 Improve prediction accuracy using **Transformer models** (e.g., Time Series Transformers)  
🔹 Use **additional sensor features** like **foot pressure** and **joint angles**  
🔹 Implement **real-time streaming** for live sensor predictions  

---

