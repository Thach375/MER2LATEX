"""
Module để tải và xử lý dataset CROHME
Chuyển đổi file InkML thành ảnh và tạo file CSV dataset hoàn chỉnh
"""

import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image, ImageDraw
import pandas as pd
from tqdm import tqdm


def download_datasets(target_folder="data"):
    """
    Tải datasets từ Kaggle
    
    Args:
        target_folder: Thư mục đích để lưu datasets
        
    Returns:
        dict: Thông tin về các datasets đã tải
    """
    try:
        import kagglehub
    except ImportError:
        print("❌ Chưa cài đặt kagglehub. Cài đặt bằng: pip install kagglehub")
        return None
    
    # Định nghĩa datasets cần tải
    datasets = [
        {
            "name": "rtatman/handwritten-mathematical-expressions",
            "target": os.path.join(target_folder, "CROHME")
        },
        {
            "name": "shahrukhkhan/im2latex100k",
            "target": os.path.join(target_folder, "IM2LATEX")
        }
    ]
    
    downloaded = []
    
    # Tải từng dataset nếu cần
    for dataset in datasets:
        dataset_name = dataset["name"]
        target_path = dataset["target"]
        
        # Kiểm tra xem thư mục có tồn tại và không trống
        should_download = False
        
        if not os.path.exists(target_path):
            should_download = True
        elif not os.listdir(target_path):
            should_download = True
        else:
            print(f"✓ Dataset {os.path.basename(target_path)} đã tồn tại")
        
        if should_download:
            print(f"⬇️  Đang tải {dataset_name}...")
            
            try:
                # Tải dataset
                cache_path = kagglehub.dataset_download(dataset_name)
                
                # Tạo thư mục đích nếu chưa có
                os.makedirs(target_path, exist_ok=True)
                
                # Move tất cả nội dung từ cache vào thư mục đích
                for item in os.listdir(cache_path):
                    src = os.path.join(cache_path, item)
                    dst = os.path.join(target_path, item)
                    
                    if os.path.exists(dst):
                        if os.path.isdir(dst):
                            shutil.rmtree(dst)
                        else:
                            os.remove(dst)
                    
                    shutil.move(src, dst)
                
                downloaded.append(target_path)
                print(f"✓ Đã tải {os.path.basename(target_path)}")
            except Exception as e:
                print(f"❌ Lỗi khi tải {dataset_name}: {e}")
    
    return {"downloaded": downloaded, "datasets": datasets}


def parse_inkml_traces(inkml_file):
    """
    Parse InkML file và trích xuất traces
    
    Args:
        inkml_file: Đường dẫn đến file InkML
        
    Returns:
        list: Danh sách các traces, mỗi trace là list các điểm (x, y)
    """
    try:
        tree = ET.parse(inkml_file)
        root = tree.getroot()
        ns = {'ink': 'http://www.w3.org/2003/InkML'}
        
        traces = []
        for trace in root.findall('.//ink:trace', ns):
            points = []
            trace_data = trace.text.strip() if trace.text else ""
            for point in trace_data.split(','):
                coords = point.strip().split()
                if len(coords) == 2:
                    try:
                        x, y = float(coords[0]), float(coords[1])
                        points.append((x, y))
                    except ValueError:
                        continue
            if points:
                traces.append(points)
        
        return traces
    except Exception:
        return []


def get_ground_truth(inkml_file):
    """
    Lấy ground truth (công thức LaTeX) từ file InkML
    
    Args:
        inkml_file: Đường dẫn đến file InkML
        
    Returns:
        str: Ground truth hoặc None nếu không có
    """
    try:
        tree = ET.parse(inkml_file)
        root = tree.getroot()
        ns = {'ink': 'http://www.w3.org/2003/InkML'}
        
        for annotation in root.findall('.//ink:annotation', ns):
            if annotation.get('type') == 'truth':
                truth = annotation.text.strip() if annotation.text else ""
                # Loại bỏ $ ở đầu và cuối nếu có
                truth = truth.strip('$').strip()
                return truth
        
        return None
    except Exception:
        return None


def traces_to_image(traces, img_size=(400, 400), padding=20, line_width=2):
    """
    Chuyển đổi traces thành ảnh
    
    Args:
        traces: List các traces từ InkML
        img_size: Kích thước ảnh đầu ra (width, height)
        padding: Khoảng cách padding
        line_width: Độ dày của nét vẽ
        
    Returns:
        PIL.Image: Ảnh đã render hoặc None nếu có lỗi
    """
    if not traces:
        return None
    
    try:
        # Tìm bounding box
        all_points = [p for trace in traces for p in trace]
        if not all_points:
            return None
        
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # Tránh chia cho 0
        width = max_x - min_x
        height = max_y - min_y
        
        if width == 0 or height == 0:
            return None
        
        # Tính scale để fit vào ảnh
        scale = min((img_size[0] - 2*padding) / width, 
                    (img_size[1] - 2*padding) / height)
        
        # Tạo ảnh trắng
        img = Image.new('RGB', img_size, 'white')
        draw = ImageDraw.Draw(img)
        
        # Vẽ từng trace
        for trace in traces:
            if len(trace) < 2:
                continue
            
            # Scale và translate points
            scaled_trace = [
                (
                    (x - min_x) * scale + padding,
                    (y - min_y) * scale + padding
                )
                for x, y in trace
            ]
            
            # Vẽ đường nối
            for i in range(len(scaled_trace) - 1):
                draw.line([scaled_trace[i], scaled_trace[i+1]], 
                         fill='black', width=line_width)
        
        return img
    except Exception:
        return None


def process_crohme_dataset(crohme_path, output_dir="processed_data"):
    """
    Xử lý dataset CROHME và tạo cấu trúc dataset hoàn chỉnh
    
    Args:
        crohme_path: Đường dẫn đến thư mục CROHME
        output_dir: Thư mục đầu ra cho dataset đã xử lý
        
    Returns:
        dict: Thống kê về quá trình xử lý
    """
    crohme_path = Path(crohme_path)
    output_path = Path(output_dir)
    
    # Tạo thư mục đích
    ground_truth_dir = output_path / "ground_truth"
    non_ground_truth_dir = output_path / "non_ground_truth"
    
    gt_images_dir = ground_truth_dir / "images"
    ngt_images_dir = non_ground_truth_dir / "images"
    
    # Tạo thư mục nếu chưa có
    gt_images_dir.mkdir(parents=True, exist_ok=True)
    ngt_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Danh sách để lưu vào CSV
    gt_data = []  # [(formula, image_name), ...]
    ngt_data = []  # [image_name, ...]
    
    # Tìm tất cả file InkML
    inkml_files = list(crohme_path.rglob("*.inkml"))
    
    # Lọc bỏ các file trong processed_data để tránh đệ quy
    inkml_files = [f for f in inkml_files 
                   if "processed_data" not in str(f) and 
                      "ground_truth" not in str(f) and 
                      "non_ground_truth" not in str(f)]
    
    print(f"🔍 Tìm thấy {len(inkml_files)} file InkML")
    
    # Đếm số lượng
    gt_count = 0
    ngt_count = 0
    error_count = 0
    
    # Xử lý từng file với progress bar
    for inkml_file in tqdm(inkml_files, desc="Xử lý files"):
        # Parse traces
        traces = parse_inkml_traces(inkml_file)
        if not traces:
            error_count += 1
            continue
        
        # Tạo tên file ảnh (dùng tên file gốc)
        image_name = inkml_file.stem + ".png"
        
        # Tạo ảnh
        img = traces_to_image(traces)
        if img is None:
            error_count += 1
            continue
        
        # Kiểm tra có ground truth không
        truth = get_ground_truth(inkml_file)
        
        if truth:
            # Có ground truth
            img_path = gt_images_dir / image_name
            img.save(img_path)
            gt_data.append({
                'formula': truth,
                'image': image_name
            })
            gt_count += 1
        else:
            # Không có ground truth
            img_path = ngt_images_dir / image_name
            img.save(img_path)
            ngt_data.append({
                'image': image_name
            })
            ngt_count += 1
    
    # Lưu CSV cho ground truth
    if gt_data:
        gt_df = pd.DataFrame(gt_data)
        gt_csv_path = ground_truth_dir / "dataset.csv"
        gt_df.to_csv(gt_csv_path, index=False, encoding='utf-8')
    
    # Lưu CSV cho non ground truth
    if ngt_data:
        ngt_df = pd.DataFrame(ngt_data)
        ngt_csv_path = non_ground_truth_dir / "dataset.csv"
        ngt_df.to_csv(ngt_csv_path, index=False, encoding='utf-8')
    
    # Báo cáo tổng hợp
    print(f"\n✓ Xử lý xong: {gt_count} có ground truth | {ngt_count} không có | {error_count} lỗi")
    
    return {
        'ground_truth': gt_count,
        'non_ground_truth': ngt_count,
        'errors': error_count,
        'total': len(inkml_files),
        'output_dir': str(output_path)
    }


def main():
    """
    Hàm chính để chạy toàn bộ pipeline: tải dataset và xử lý
    """
    print("🚀 Tải và xử lý dataset CROHME\n")
    
    # Bước 1: Tải datasets
    print("📥 Bước 1: Tải datasets")
    download_result = download_datasets(target_folder="data")
    
    if download_result is None:
        print("❌ Không thể tải datasets")
        return
    
    # Bước 2: Xử lý CROHME dataset
    print("\n⚙️  Bước 2: Xử lý CROHME")
    
    crohme_path = "data/CROHME"
    if not os.path.exists(crohme_path):
        print(f"❌ Không tìm thấy thư mục CROHME tại {crohme_path}")
        return
    
    process_result = process_crohme_dataset(
        crohme_path=crohme_path,
        output_dir="processed_data"
    )
    
    print(f"\n✅ Hoàn tất! Dataset lưu tại: {process_result['output_dir']}")


if __name__ == "__main__":
    main()
