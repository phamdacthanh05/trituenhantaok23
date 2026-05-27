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
MODEL_PATH   = 'cat_dog_others_model.pth'
NUM_CLASSES  = 3
LABELS       = ["Mèo", "Chó", "Khác"]   # thứ tự phải khớp lúc train

# Ngưỡng lọc — chỉnh tại đây nếu cần
HIGH_THRESHOLD  = 0.80   # độ tin cậy tối thiểu để nhận Mèo/Chó
OTHER_FLOOR     = 0.15   # nếu xác suất nhãn "Khác" >= mức này → trả về Khác
MIN_ANIMAL_PROB = 0.45   # cả Mèo lẫn Chó đều yếu hơn ngưỡng này → trả về Khác

# 2. LOAD MODEL
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    print(">>> Model đã được load thành công!")
except FileNotFoundError:
    print(f">>> CẢNH BÁO: Không tìm thấy file '{MODEL_PATH}'. Server vẫn chạy nhưng predict sẽ lỗi.")
except Exception as e:
    print(f">>> Lỗi khi load model: {e}")

# 3. TIỀN XỬ LÝ ẢNH
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 4. HÀM LỌC NHÃN
def smart_filter(cat_prob, dog_prob, other_prob, top_prob, top_idx):

    confidence = round(top_prob * 100, 2)

    conditions = [
        top_prob < HIGH_THRESHOLD,
        other_prob >= OTHER_FLOOR,
        cat_prob < MIN_ANIMAL_PROB and dog_prob < MIN_ANIMAL_PROB,
    ]

    if any(conditions):
        return "Khác", confidence

    return LABELS[top_idx], confidence

# 5. TRANG CHỦ — serve file index.html từ thư mục templates/
@app.route('/')
def index():
    return render_template('index.html')

# 6. ENDPOINT DỰ ĐOÁN
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Không có file ảnh trong request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Tên file rỗng'}), 400

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_t = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img_t)
            probs  = F.softmax(output, dim=1)[0]

        cat_prob   = probs[0].item()
        dog_prob   = probs[1].item()
        other_prob = probs[2].item()

        top_prob, top_idx = torch.max(probs, dim=0)
        top_prob = top_prob.item()
        top_idx  = top_idx.item()

        print(f">>> Mèo: {cat_prob:.4f} | Chó: {dog_prob:.4f} | Khác: {other_prob:.4f}")
        print(f">>> Top: {LABELS[top_idx]} ({top_prob*100:.2f}%)")

        label, confidence = smart_filter(cat_prob, dog_prob, other_prob, top_prob, top_idx)

        print(f">>> Kết quả sau lọc: {label} ({confidence}%)\n")

        return jsonify({
            'result':     label,
            'confidence': confidence,
            'details': {
                'Mèo':  round(cat_prob   * 100, 2),
                'Chó':  round(dog_prob   * 100, 2),
                'Khác': round(other_prob * 100, 2),
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'labels': LABELS})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)