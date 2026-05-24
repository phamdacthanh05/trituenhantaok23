import torch
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import io

app = Flask(__name__)
CORS(app)

# --- 1. CẤU HÌNH MODEL ---
# Phải trùng khớp 100% với cấu trúc khi huấn luyện
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2) 

# Load trọng số đã train
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

# Route phục vụ giao diện Web (index.html trong thư mục templates)
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# API Dự đoán (được gọi từ JavaScript)
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Không có file ảnh'}), 400
    
    file = request.files['file']
    
    try:
        # Chuyển đổi dữ liệu ảnh
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img_t = transform(img).unsqueeze(0)
        
        # Chạy dự đoán
        with torch.no_grad():
            output = model(img_t)
            _, predicted = torch.max(output, 1)
            
            # 0: Mèo, 1: Chó (dựa trên thứ tự alphabet của thư mục)
            label = "Mèo" if predicted.item() == 0 else "Chó"
            
        return jsonify({'result': label})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- 4. CHẠY SERVER ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)