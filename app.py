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

# 1. CẤU HÌNH
MODEL_PATH   = 'cat_dog_model.pth' 
NUM_CLASSES  = 2
LABELS       = ["Mèo", "Chó"]

# 2. LOAD MODEL
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    print(">>> Model 2 nhãn đã được load thành công!")
except Exception as e:
    print(f">>> Lỗi khi load model: {e}")

# 3. TIỀN XỬ LÝ ẢNH
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 5. TRANG CHỦ
@app.route('/')
def index():
    return render_template('index.html')

# 6. ENDPOINT DỰ ĐOÁN (2 NHÃN)
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Không có file ảnh'}), 400

    file = request.files['file']
    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_t = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img_t)
            probs  = F.softmax(output, dim=1)[0]

        cat_prob = probs[0].item()
        dog_prob = probs[1].item()

        top_prob, top_idx = torch.max(probs, dim=0)
        
        label = LABELS[top_idx.item()]
        confidence = round(top_prob.item() * 100, 2)

        print(f">>> Kết quả: {label} ({confidence}%) | Mèo: {cat_prob:.2f} | Chó: {dog_prob:.2f}")

        return jsonify({
            'result':     label,
            'confidence': confidence,
            'details': {
                'Mèo': round(cat_prob * 100, 2),
                'Chó': round(dog_prob * 100, 2),
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 7. ENDPOINT KIỂM TRA SERVER
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_type': 'binary_classification'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)