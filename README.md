# 🌿 Leaf Analyser

A plant leaf disease classifier built with a **Convolutional Neural Network written from scratch in NumPy** (no PyTorch/TensorFlow), served through a Flask API and a simple HTML/CSS/JS frontend.

Upload a photo of a plant leaf, and the app predicts the plant species and whether it's healthy or affected by a specific disease (e.g. `Tomato___Early_blight`, `Apple___Apple_scab`).

## Features

- Custom CNN implementation (convolution, ReLU, max-pooling, dense layers, softmax) built entirely with NumPy — see `Untitled.ipynb` for the training code.
- Flask backend (`backend.py`) that loads trained weights and serves predictions via a `/predict` REST endpoint.
- Lightweight frontend (`index.html`, `style.css`, `app.js`) for uploading an image and viewing the prediction.
- **Demo mode**: if trained weight files aren't present, the backend automatically falls back to returning sample predictions, so the app is runnable out of the box.

## Project Structure

```
leaf_analyser/
├── backend.py          # Flask server + from-scratch CNN model
├── index.html           # Frontend UI
├── style.css             # Styling
├── app.js                 # Frontend logic (calls the /predict API)
├── Untitled.ipynb    # Model training notebook
├── plant_cnn_weights.npz     # (not included) trained weights
└── plant_label_map.json      # (not included) class label mapping
```

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/ADITYA01216/leaf-analyser.git
cd leaf-analyser
```

### 2. Install backend dependencies

```bash
pip install flask flask-cors numpy pillow
```

### 3. Run the backend

```bash
python backend.py
```

The server starts at `http://127.0.0.1:5000`.

> If `plant_cnn_weights.npz` and `plant_label_map.json` aren't present in the folder, the app runs in **DEMO mode** and returns random sample predictions — useful for testing the UI without a trained model.

### 4. Open the frontend

Simply open `index.html` in your browser (or serve it with a local server), then upload a leaf image to see the prediction.

## Model

The CNN architecture (defined in `backend.py` and trained in `Untitled.ipynb`):

```
Input (32x32x3)
→ Conv Layer → ReLU → MaxPool 2x2
→ Conv Layer → ReLU → MaxPool 2x2
→ Flatten
→ Dense (512) → ReLU
→ Dense (num_classes) → Softmax
```

All layers are implemented manually in NumPy for educational purposes — no deep learning frameworks used.

## To-Do

- [ ] Add trained weights + label map to enable real predictions
- [ ] Improve model accuracy / add more disease classes
- [ ] Add deployment instructions (e.g. Render, Railway, Docker)

## License

This project currently has no license specified — all rights reserved by default. Add a `LICENSE` file if you'd like to open it up for reuse.
