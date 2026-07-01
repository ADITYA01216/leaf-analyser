import os
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import time

# --- MODEL CLASSES (Extracted from your notebook) ---

class ConvLayer:
    def __init__(self, in_channels, out_channels, kernel_size=3):
        self.K = kernel_size
        self.W = np.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.b = np.zeros(out_channels)

    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        F, _, K, _ = self.W.shape
        Ho, Wo = H-K+1, W-K+1
        out = np.zeros((N, F, Ho, Wo))
        for f in range(F):
            for i in range(Ho):
                for j in range(Wo):
                    out[:,f,i,j] = np.tensordot(
                        x[:,:,i:i+K,j:j+K], self.W[f], axes=([1,2,3],[0,1,2])) + self.b[f]
        return out

class ReLU:
    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask

class MaxPool2x2:
    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        H2, W2 = H//2, W//2
        out = np.zeros((N, C, H2, W2))
        for i in range(H2):
            for j in range(W2):
                out[:,:,i,j] = x[:,:,i*2:i*2+2,j*2:j*2+2].max(axis=(2,3))
        return out

class Flatten:
    def forward(self, x): 
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

class DenseLayer:
    def __init__(self, in_dim, out_dim):
        self.W = np.zeros((in_dim, out_dim))
        self.b = np.zeros(out_dim)

    def forward(self, x): 
        return x @ self.W + self.b

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

class CNN:
    def __init__(self, num_classes=38, img_size=32):
        self.num_classes = num_classes
        self.img_size = img_size
        self.conv1 = ConvLayer(3, 16)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2x2()
        self.conv2 = ConvLayer(16, 32)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2x2()
        self.flatten = Flatten()
        h = ((img_size - 2) // 2 - 2) // 2
        flat_dim = 32 * h * h
        self.dense1 = DenseLayer(flat_dim, 512)
        self.relu3 = ReLU()
        self.dense2 = DenseLayer(512, num_classes)

    def forward(self, x):
        x = self.conv1.forward(x); x = self.relu1.forward(x); x = self.pool1.forward(x)
        x = self.conv2.forward(x); x = self.relu2.forward(x); x = self.pool2.forward(x)
        x = self.flatten.forward(x)
        x = self.dense1.forward(x); x = self.relu3.forward(x)
        return self.dense2.forward(x)

    def predict(self, x):
        return np.argmax(softmax(self.forward(x)), axis=1)

    def load(self, weights_file):
        w = np.load(weights_file)
        self.conv1.W = w['conv1_W']; self.conv1.b = w['conv1_b']
        self.conv2.W = w['conv2_W']; self.conv2.b = w['conv2_b']
        self.dense1.W = w['dense1_W']; self.dense1.b = w['dense1_b']
        self.dense2.W = w['dense2_W']; self.dense2.b = w['dense2_b']
        print(f"Loaded weights from {weights_file}")

# --- FLASK APP ---

app = Flask(__name__)
CORS(app)

# Settings
IMG_SIZE = 32
WEIGHTS_PATH = "plant_cnn_weights.npz"
LABEL_MAP_PATH = "plant_label_map.json"

cnn = None
label_map = None
index_to_label = None

def init_model():
    global cnn, label_map, index_to_label
    if not os.path.exists(WEIGHTS_PATH) or not os.path.exists(LABEL_MAP_PATH):
        print("--------------------------------------------------")
        print("💡 MODEL FILES NOT FOUND - STARTING IN DEMO MODE")
        print("   (Website will show sample results for testing)")
        print("--------------------------------------------------")
        return "DEMO"
    
    try:
        with open(LABEL_MAP_PATH, 'r') as f:
            label_map = json.load(f)
            index_to_label = {int(v): k for k, v in label_map.items()}
        
        cnn = CNN(num_classes=len(label_map), img_size=IMG_SIZE)
        cnn.load(WEIGHTS_PATH)
        return "REAL"
    except Exception as e:
        print(f"Error loading model: {e}")
        return "DEMO"

MODE = "DEMO"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    # --- DEMO MODE RESPONSE ---
    if MODE == "DEMO":
        time.sleep(1.5) # Simulate processing
        demo_results = [
            "Apple___Apple_scab", "Corn___Common_rust", 
            "Tomato___Early_blight", "Potato___Late_blight",
            "Grape___Black_rot", "Pepper__bell___healthy"
        ]
        import random
        return jsonify({'prediction': random.choice(demo_results)})
    
    # --- REAL MODE RESPONSE ---
    file = request.files['file']
    try:
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)
        image_batch = np.expand_dims(arr, axis=0)
        
        pred_idx = cnn.predict(image_batch)[0]
        label = index_to_label.get(pred_idx, "Unknown")
        
        return jsonify({'prediction': label})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    MODE = init_model()
    print(f"Backend server starting on http://127.0.0.1:5000 (Mode: {MODE})")
    app.run(port=5000, debug=False)
