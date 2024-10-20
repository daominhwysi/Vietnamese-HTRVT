import torch
import os
import re
from PIL import Image
from utils import utils
from data import dataset
from model import HTR_VT
from collections import OrderedDict
import skimage
import time
import numpy as np
from tqdm import tqdm
import editdistance
from utils import option

def load_image(image_path, img_size):
    """Load và xử lý ảnh đầu vào."""
    with Image.open(image_path).convert('L') as image:
        image = skimage.img_as_float32(image)  # Chuyển đổi thành float32
        image = np.array(image)
        image = dataset.npThum(image, img_size[0], img_size[1])

        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)

        image = image.transpose((2, 0, 1))  # Chuyển thành (1, H, W)
        image = torch.from_numpy(image).unsqueeze(0).float()
    
    return image


def load_model(device, model_path, nb_cls, img_size):
    """Load mô hình HTR-VT và checkpoint."""
    model = HTR_VT.create_model(nb_cls=nb_cls, img_size=img_size[::-1])
    
    print(f'Loading model checkpoint from {model_path}')
    ckpt = torch.load(model_path, map_location='cpu')
    model_dict = {re.sub('module.', '', k): v for k, v in ckpt['state_dict_ema'].items()}

    model.load_state_dict(model_dict, strict=True)
    model = model.to(device).eval()  # Đưa vào chế độ eval
    return model


def infer_image(image_path, model, device):
    """Infer một ảnh với mô hình đã load."""
    image = load_image(image_path, [512,64]).to(device)

    with torch.no_grad():
        start_time = time.time()
        preds = model(image).float()
        preds_size = torch.IntTensor([preds.size(1)])
        preds = preds.permute(1, 0, 2).log_softmax(2)

        _, preds_index = preds.max(2)
        preds_index = preds_index.transpose(1, 0).contiguous().view(-1)

        # Sử dụng danh sách ký tự thực tế
        converter = utils.CTCLabelConverter(
            [' ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 
             'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 
             'y', 'à', 'á', 'â', 'ã', 'è', 'é', 'ê', 'ì', 'í', 'ò', 'ó', 'ô', 'õ', 'ù', 'ú', 
             'ă', 'đ', 'ĩ', 'ũ', 'ơ', 'ư', 'ạ', 'ả', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'ắ', 'ằ', 'ẳ', 
             'ẵ', 'ặ', 'ẹ', 'ẻ', 'ẽ', 'ế', 'ề', 'ể', 'ễ', 'ệ', 'ỉ', 'ị', 'ọ', 'ỏ', 'ố', 'ồ', 
             'ổ', 'ỗ', 'ộ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ', 'ụ', 'ủ', 'ứ', 'ừ', 'ử', 'ữ', 'ự', 'ỳ', 
             'ỷ', 'ỹ']
        )

        pred_str = converter.decode(preds_index.data, preds_size.data)

        inference_time = time.time() - start_time
        print(f"Prediction: {pred_str}")
        print(f"Thời gian infer: {inference_time:.4f} giây")

    return pred_str

def clean_string(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()  # Xóa khoảng trắng thừa

def compute_cer(reference: str, hypothesis: str) -> float:
    # Kiểm tra kiểu dữ liệu
    if not isinstance(reference, str):
        print(f"Expected string for reference but got {type(reference)}: {reference}")
        reference = str(reference)  # Chuyển đổi về chuỗi
    if not isinstance(hypothesis, str):
        print(f"Expected string for hypothesis but got {type(hypothesis)}: {hypothesis}")
        hypothesis = str(hypothesis)  # Chuyển đổi về chuỗi

    reference = clean_string(reference)
    hypothesis = clean_string(hypothesis)

    # Tính toán khoảng cách chỉnh sửa
    errors = editdistance.eval(reference, hypothesis)

    cer = errors / len(reference) if reference else 0
    return cer

if __name__ == '__main__':
    args = option.get_args_parser()
    folder_path = 'resized_2'  # Đường dẫn đến thư mục chứa ảnh
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # Load mô hình 1 lần duy nhất
    model = load_model(device, '/content/output/Evnondb/best_CER.pth', args.nb_cls, args.img_size)

    # Tạo danh sách đường dẫn ảnh và tệp tin văn bản
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpeg')]
    results = []

    # Duyệt qua tất cả các tệp tin trong thư mục với tqdm
    for filename in tqdm(image_files, desc="Processing Images", unit="image"):
        image_path = os.path.join(folder_path, filename)
        text_path = os.path.join(folder_path, filename.replace('.jpeg', '.txt'))

        # Infer ảnh
        prediction = infer_image(image_path, model, device)

        # Nếu prediction là một danh sách, chuyển đổi thành chuỗi
        if isinstance(prediction, list) and len(prediction) > 0:
            prediction = prediction[0]  # Lấy phần tử đầu tiên từ danh sách

        # Đọc ground truth từ file txt
        ground_truth = ''
        if os.path.exists(text_path):
            with open(text_path, 'r', encoding='utf-8') as f:
                ground_truth = f.read().strip()

        # Tính CER
        cer = compute_cer(ground_truth, prediction)  # Lưu ý thứ tự tham số
        results.append({
            'filename': filename,
            'prediction': prediction,
            'ground_truth': ground_truth,
            'cer': cer
        })

    # In ra kết quả
    for result in results:
        print(f"Filename: {result['filename']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Ground Truth: {result['ground_truth']}")
        print(f"CER: {result['cer']:.4f}\n")

    # Sắp xếp kết quả theo CER giảm dần
    sorted_results = sorted(results, key=lambda x: x['cer'], reverse=True)

    # Lưu 100 ảnh có CER cao nhất vào file txt
    top_n = 100
    output_file = 'top_cer_images.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in sorted_results[:top_n]:
            f.write(f"{result['filename']} - CER: {result['cer']:.4f}\n")

    print(f"\nĐã lưu {top_n} ảnh có CER cao nhất vào file {output_file}")
