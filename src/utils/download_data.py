"""
Module de tai va xu ly dataset CROHME va IM2LATEX
Chuyen doi file InkML thanh anh va tao file CSV dataset hoan chinh

HUONG DAN SETUP KAGGLE CREDENTIALS:
1. Dang nhap vao https://www.kaggle.com
2. Vao Account Settings -> API -> Create New Token
3. Download file kaggle.json
4. Dat file vao ~/.kaggle/kaggle.json hoac set environment variables:
   export KAGGLE_USERNAME="your_username"
   export KAGGLE_KEY="your_api_key"
"""

import os
import shutil
import subprocess
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image, ImageDraw
import pandas as pd
from tqdm import tqdm


def setup_kaggle_credentials(username=None, key=None, required=True):
    """
    Setup Kaggle credentials tu environment variables hoac tham so
    
    Args:
        username: Kaggle username (optional, lay tu env neu khong co)
        key: Kaggle API key (optional, lay tu env neu khong co)
        required: Neu True, raise exception khi khong co credentials
        
    Returns:
        bool: True neu setup thanh cong
        
    Raises:
        SystemExit: Neu required=True va khong co credentials
    """
    # Kiem tra credentials da co chua
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
    
    if os.path.exists(kaggle_json):
        print("[OK] Kaggle credentials da ton tai")
        return True
    
    # Lay tu environment variables
    username = username or os.environ.get("KAGGLE_USERNAME")
    key = key or os.environ.get("KAGGLE_KEY")
    
    if not username or not key:
        print("")
        print("=" * 70)
        print("[ERROR] KAGGLE CREDENTIALS REQUIRED!")
        print("=" * 70)
        print("")
        print("De download datasets, ban can setup Kaggle credentials:")
        print("")
        print("  BUOC 1: Lay API key")
        print("    - Dang nhap: https://www.kaggle.com")
        print("    - Vao: Settings -> API -> Create New Token")
        print("    - Download file kaggle.json")
        print("")
        print("  BUOC 2: Setup credentials (chon 1 trong 2 cach)")
        print("")
        print("    Cach A - Dung script:")
        print("      ./setup_kaggle.sh YOUR_USERNAME YOUR_API_KEY")
        print("")
        print("    Cach B - Set environment variables:")
        print("      export KAGGLE_USERNAME='your_username'")
        print("      export KAGGLE_KEY='your_api_key'")
        print("")
        print("  BUOC 3: Chay lai pipeline")
        print("      ./pipeline.sh")
        print("")
        print("=" * 70)
        
        if required:
            raise SystemExit(1)
        return False
    
    # Tao thu muc va file
    os.makedirs(kaggle_dir, exist_ok=True)
    
    import json
    with open(kaggle_json, "w") as f:
        json.dump({"username": username, "key": key}, f)
    
    os.chmod(kaggle_json, 0o600)
    print(f"[OK] Da tao Kaggle credentials tai {kaggle_json}")
    return True


def download_with_kaggle_cli(dataset_name, target_path):
    """
    Download dataset bang Kaggle CLI
    
    Args:
        dataset_name: Ten dataset tren Kaggle (owner/dataset)
        target_path: Thu muc dich
        
    Returns:
        bool: True neu thanh cong
    """
    try:
        # Tao thu muc dich
        os.makedirs(target_path, exist_ok=True)
        
        # Download bang kaggle CLI
        cmd = [
            "kaggle", "datasets", "download",
            "-d", dataset_name,
            "-p", target_path,
            "--unzip"
        ]
        
        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return True
        else:
            print(f"  CLI Error: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("  [WARNING] Kaggle CLI chua duoc cai dat. Cai bang: pip install kaggle")
        return False
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def download_with_kagglehub(dataset_name, target_path):
    """
    Download dataset bang kagglehub library
    
    Args:
        dataset_name: Ten dataset tren Kaggle
        target_path: Thu muc dich
        
    Returns:
        bool: True neu thanh cong
    """
    try:
        import kagglehub
        
        cache_path = kagglehub.dataset_download(dataset_name)
        
        # Kiem tra cache co noi dung khong
        if not os.path.exists(cache_path) or not os.listdir(cache_path):
            print(f"  [WARNING] kagglehub tra ve cache rong")
            return False
        
        # Tao thu muc dich
        os.makedirs(target_path, exist_ok=True)
        
        # Copy noi dung tu cache vao target
        for item in os.listdir(cache_path):
            src = os.path.join(cache_path, item)
            dst = os.path.join(target_path, item)
            
            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            
            shutil.copytree(src, dst) if os.path.isdir(src) else shutil.copy2(src, dst)
        
        return True
        
    except ImportError:
        print("  [WARNING] kagglehub chua duoc cai dat")
        return False
    except Exception as e:
        print(f"  [ERROR] kagglehub: {e}")
        return False


def download_single_dataset(dataset_name, target_path):
    """
    Download mot dataset, thu nhieu phuong thuc
    
    Args:
        dataset_name: Ten dataset (owner/dataset)
        target_path: Thu muc dich
        
    Returns:
        bool: True neu thanh cong
    """
    print(f"[DOWNLOAD] {dataset_name} -> {target_path}")
    
    # Thu kaggle CLI truoc (tin cay hon)
    if setup_kaggle_credentials():
        print("  Trying Kaggle CLI...")
        if download_with_kaggle_cli(dataset_name, target_path):
            return True
    
    # Thu kagglehub
    print("  Trying kagglehub...")
    if download_with_kagglehub(dataset_name, target_path):
        return True
    
    print(f"[FAILED] Khong the tai {dataset_name}")
    print("  Vui long download thu cong tu:")
    print(f"  https://www.kaggle.com/datasets/{dataset_name}")
    print(f"  Giai nen vao: {target_path}")
    return False


def download_datasets(target_folder="data"):
    """
    Tai datasets tu Kaggle (CROHME va IM2LATEX)
    
    BAT BUOC phai co Kaggle credentials truoc khi chay.
    
    Args:
        target_folder: Thu muc dich de luu datasets
        
    Returns:
        dict: Thong tin ve cac datasets da tai
        
    Raises:
        SystemExit: Neu khong co Kaggle credentials
    """
    # BAT BUOC kiem tra credentials truoc
    print("[CHECK] Kiem tra Kaggle credentials...")
    setup_kaggle_credentials(required=True)  # Se exit neu khong co
    
    # Dinh nghia datasets can tai
    datasets = [
        {
            "name": "rtatman/handwritten-mathematical-expressions",
            "target": os.path.join(target_folder, "CROHME"),
            "required_check": lambda p: any(f.endswith('.inkml') for r, d, files in os.walk(p) for f in files)
        },
        {
            "name": "shahrukhkhan/im2latex100k", 
            "target": os.path.join(target_folder, "IM2LATEX"),
            "required_check": lambda p: any(f.endswith('.csv') for f in os.listdir(p)) if os.path.exists(p) else False
        }
    ]
    
    downloaded = []
    failed = []
    
    for dataset in datasets:
        dataset_name = dataset["name"]
        target_path = dataset["target"]
        check_func = dataset.get("required_check", lambda p: bool(os.listdir(p)))
        
        # Kiem tra da co data hop le chua
        if os.path.exists(target_path) and check_func(target_path):
            print(f"[OK] Dataset {os.path.basename(target_path)} da ton tai va hop le")
            continue
        
        # Xoa folder rong neu co
        if os.path.exists(target_path) and not os.listdir(target_path):
            os.rmdir(target_path)
        
        # Download
        if download_single_dataset(dataset_name, target_path):
            downloaded.append(target_path)
            print(f"[OK] Da tai {os.path.basename(target_path)}")
        else:
            failed.append(dataset_name)
    
    # Kiem tra tat ca datasets da download thanh cong chua
    if failed:
        print("")
        print("=" * 70)
        print("[ERROR] DOWNLOAD THAT BAI!")
        print("=" * 70)
        for ds in failed:
            print(f"  - {ds}")
        print("")
        print("Vui long kiem tra:")
        print("  1. Kaggle credentials hop le")
        print("  2. Ket noi internet")
        print("  3. Dataset van con tren Kaggle")
        print("")
        raise SystemExit(1)
    
    return {"downloaded": downloaded, "failed": failed, "datasets": datasets}


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


def process_crohme_dataset(crohme_path, output_dir="data/preprocessed/crohme"):
    """
    Xu ly dataset CROHME va tao cau truc dataset hoan chinh
    
    Args:
        crohme_path: Duong dan den thu muc CROHME
        output_dir: Thu muc dau ra cho dataset da xu ly
        
    Returns:
        dict: Thong ke ve qua trinh xu ly
    """
    crohme_path = Path(crohme_path)
    output_path = Path(output_dir)
    
    # Tao thu muc dich
    ground_truth_dir = output_path / "ground_truth"
    non_ground_truth_dir = output_path / "non_ground_truth"
    
    gt_images_dir = ground_truth_dir / "images"
    ngt_images_dir = non_ground_truth_dir / "images"
    
    # Tao thu muc neu chua co
    gt_images_dir.mkdir(parents=True, exist_ok=True)
    ngt_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Danh sach de luu vao CSV
    gt_data = []  # [(formula, image_name), ...]
    ngt_data = []  # [image_name, ...]
    
    # Tim tat ca file InkML
    inkml_files = list(crohme_path.rglob("*.inkml"))
    
    # Loc bo cac file trong processed_data de tranh de quy
    inkml_files = [f for f in inkml_files 
                   if "processed_data" not in str(f) and 
                      "ground_truth" not in str(f) and 
                      "non_ground_truth" not in str(f)]
    
    print(f"[INFO] Tim thay {len(inkml_files)} file InkML")
    
    # Dem so luong
    gt_count = 0
    ngt_count = 0
    error_count = 0
    
    # Xu ly tung file voi progress bar
    for inkml_file in tqdm(inkml_files, desc="Xu ly files"):
        # Parse traces
        traces = parse_inkml_traces(inkml_file)
        if not traces:
            error_count += 1
            continue
        
        # Tao ten file anh (dung ten file goc)
        image_name = inkml_file.stem + ".png"
        
        # Tao anh
        img = traces_to_image(traces)
        if img is None:
            error_count += 1
            continue
        
        # Kiem tra co ground truth khong
        truth = get_ground_truth(inkml_file)
        
        if truth:
            # Co ground truth
            img_path = gt_images_dir / image_name
            img.save(img_path)
            gt_data.append({
                'formula': truth,
                'image': image_name
            })
            gt_count += 1
        else:
            # Khong co ground truth
            img_path = ngt_images_dir / image_name
            img.save(img_path)
            ngt_data.append({
                'image': image_name
            })
            ngt_count += 1
    
    # Luu CSV cho ground truth
    if gt_data:
        gt_df = pd.DataFrame(gt_data)
        gt_csv_path = ground_truth_dir / "dataset.csv"
        gt_df.to_csv(gt_csv_path, index=False, encoding='utf-8')
    
    # Luu CSV cho non ground truth
    if ngt_data:
        ngt_df = pd.DataFrame(ngt_data)
        ngt_csv_path = non_ground_truth_dir / "dataset.csv"
        ngt_df.to_csv(ngt_csv_path, index=False, encoding='utf-8')
    
    # Bao cao tong hop
    print(f"\n[OK] Xu ly xong: {gt_count} co ground truth | {ngt_count} khong co | {error_count} loi")
    
    return {
        'ground_truth': gt_count,
        'non_ground_truth': ngt_count,
        'errors': error_count,
        'total': len(inkml_files),
        'output_dir': str(output_path)
    }


def main():
    """
    Ham chinh de chay toan bo pipeline: tai dataset va xu ly
    """
    print("[START] Tai va xu ly datasets\n")
    
    # Buoc 1: Tai datasets
    print("[STEP 1] Tai datasets (CROHME va IM2LATEX)")
    download_result = download_datasets(target_folder="data")
    
    if download_result is None:
        print("[ERROR] Khong the tai datasets")
        return
    
    # Buoc 2: Xu ly CROHME dataset (InkML -> Images)
    print("\n[STEP 2] Xu ly CROHME (InkML -> Images)")
    
    crohme_path = "data/CROHME"
    if os.path.exists(crohme_path):
        process_result = process_crohme_dataset(
            crohme_path=crohme_path,
            output_dir="data/preprocessed/crohme"
        )
        print(f"[OK] CROHME: {process_result['ground_truth']} samples -> data/preprocessed/crohme/")
    else:
        print(f"[SKIP] CROHME khong ton tai tai {crohme_path}")
    
    # Buoc 3: Kiem tra IM2LATEX (da co san images, chi can verify)
    print("\n[STEP 3] Kiem tra IM2LATEX")
    
    im2latex_path = "data/IM2LATEX"
    if os.path.exists(im2latex_path):
        # IM2LATEX da co san images va CSV, chi can kiem tra
        csv_files = list(Path(im2latex_path).glob("*.csv"))
        image_dirs = [d for d in Path(im2latex_path).iterdir() if d.is_dir()]
        
        print(f"[OK] IM2LATEX: {len(csv_files)} CSV files, {len(image_dirs)} image folders")
        for csv in csv_files:
            df = pd.read_csv(csv)
            print(f"     - {csv.name}: {len(df)} samples")
    else:
        print(f"[SKIP] IM2LATEX khong ton tai tai {im2latex_path}")
    
    print(f"\n[DONE] Hoan tat!")
    print(f"  - CROHME processed: data/preprocessed/crohme/")
    print(f"  - IM2LATEX raw: data/IM2LATEX/")


if __name__ == "__main__":
    main()
