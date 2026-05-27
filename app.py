import torch
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import io
import torch.nn.functional as F

app = Flask(__name__)
CORS(app)

# --- 1. CẤU HÌNH MODEL ---
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2) 

try:
    model.load_state_dict(torch.load('cat_dog_model.pth', map_location='cpu'))
    model.eval()
    print(">>> Model đã được load thành công!")
except Exception as e:
    print(f">>> Lỗi khi load model: {e}")

# --- 2. TIỀN XỬ LÝ ẢNH ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 3. CÁC ĐƯỜNG DẪN (ROUTES) ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Không có file ảnh'}), 400
    
    file = request.files['file']
    
    try:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img_t = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            output = model(img_t)
            
            # Sử dụng Softmax để tính xác suất
            probabilities = F.softmax(output, dim=1)
            
            # Lấy giá trị cao nhất
            prob, predicted = torch.max(probabilities, 1)
            
            label = "Mèo" if predicted.item() == 0 else "Chó"
            confidence = round(prob.item() * 100, 2)
            
        return jsonify({'result': label, 'confidence': confidence})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)