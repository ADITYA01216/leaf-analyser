let API_URL = localStorage.getItem('leaf_ai_url') || "http://127.0.0.1:5000/predict"; 

// DOM Elements
const configBtn = document.getElementById('configBtn');
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const selectBtn = document.getElementById('selectBtn');
const uploadContent = document.getElementById('uploadContent');
const previewContent = document.getElementById('previewContent');
const imagePreview = document.getElementById('imagePreview');
const retakeBtn = document.getElementById('retakeBtn');
const resultsZone = document.getElementById('resultsZone');
const loader = document.getElementById('loader');
const resultCard = document.getElementById('resultCard');
const predictionResult = document.getElementById('predictionResult');
const confidenceScore = document.getElementById('confidenceScore');
const resetBtn = document.getElementById('resetBtn');

let selectedFile = null;

// Event Listeners
selectBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
});

dropZone.addEventListener('click', () => {
    if (!selectedFile) fileInput.click();
});

fileInput.addEventListener('change', handleFileSelect);

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
});

retakeBtn.addEventListener('click', resetUploader);

resetBtn.addEventListener('click', () => {
    resultsZone.classList.add('hidden');
    dropZone.classList.remove('hidden');
    resetUploader();
});

// Functions
function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

function processFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file.');
        return;
    }
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        uploadContent.classList.add('hidden');
        previewContent.classList.remove('hidden');
        
        // AUTOMATIC CONNECTION: Trigger analysis instantly after preview is set
        setTimeout(performAnalysis, 500); 
    };
    reader.readAsDataURL(file);
}

function resetUploader() {
    selectedFile = null;
    fileInput.value = '';
    imagePreview.src = '';
    previewContent.classList.add('hidden');
    uploadContent.classList.remove('hidden');
}

async function performAnalysis() {
    if (!selectedFile) return;

    // Show loading state
    dropZone.classList.add('hidden');
    resultsZone.classList.remove('hidden');
    loader.classList.remove('hidden');
    resultCard.classList.add('hidden');

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server responded with ${response.status}`);
        }

        const data = await response.json();
        displayResults(data);
    } catch (error) {
        console.error('Error during analysis:', error);
        alert('Failed to connect to AI backend. Please ensure the backend is running and the URL is correct.');
        
        // Go back to preview if failed
        loader.classList.add('hidden');
        resultsZone.classList.add('hidden');
        dropZone.classList.remove('hidden');
    }
}

function displayResults(data) {
    loader.classList.add('hidden');
    resultCard.classList.remove('hidden');

    const prediction = data.prediction || "Unknown Disease";
    
    // Format the prediction text (e.g., Tomato___Early_blight -> Tomato: Early Blight)
    const formattedPrediction = prediction.replace(/___/g, ': ').replace(/_/g, ' ');
    
    predictionResult.innerText = formattedPrediction;
    
    // The current backend doesn't return confidence, but we can simulate or handle if added
    if (data.confidence) {
        confidenceScore.innerText = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
    } else {
        confidenceScore.innerText = `Diagnosis confirmed by LeafGuard AI Model.`;
    }
}

configBtn.addEventListener('click', () => {
    const newUrl = prompt("Enter your Backend API URL (e.g., https://.../predict):", API_URL);
    if (newUrl) {
        API_URL = newUrl;
        localStorage.setItem('leaf_ai_url', newUrl);
        alert("Backend URL updated successfully!");
    }
});

