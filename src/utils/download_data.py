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
    """Setup Kaggle credentials"""
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
    
    if os.path.exists(kaggle_json):
        return True
    
    username = username or os.environ.get("KAGGLE_USERNAME")
    key = key or os.environ.get("KAGGLE_KEY")
    
    if not username or not key:
        if required:
            print("[ERROR] Kaggle credentials not found!")
            raise SystemExit(1)
        return False
    
    os.makedirs(kaggle_dir, exist_ok=True)
    
    import json
    with open(kaggle_json, "w") as f:
        json.dump({"username": username, "key": key}, f)
    
    os.chmod(kaggle_json, 0o600)
    return True

def download_with_kaggle_cli(dataset_name, target_path):
    """Download dataset bang Kaggle CLI"""
    try:
        os.makedirs(target_path, exist_ok=True)
        cmd = ["kaggle", "datasets", "download", "-d", dataset_name, "-p", target_path, "--unzip"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False


def download_with_kagglehub(dataset_name, target_path):
    """Download dataset bang kagglehub library"""
    try:
        import kagglehub
        cache_path = kagglehub.dataset_download(dataset_name)
        
        if not os.path.exists(cache_path) or not os.listdir(cache_path):
            return False
        
        os.makedirs(target_path, exist_ok=True)
        
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
    except:
        return False


def download_single_dataset(dataset_name, target_path):
    """Download mot dataset, thu nhieu phuong thuc"""
    if setup_kaggle_credentials():
        if download_with_kaggle_cli(dataset_name, target_path):
            return True
    
    if download_with_kagglehub(dataset_name, target_path):
        return True
    
    return False


def download_datasets(target_folder="data"):
    """Tai datasets tu Kaggle (CROHME va IM2LATEX)"""
    setup_kaggle_credentials(required=True)
    
    datasets = [
        {
            "name": "rtatman/handwritten-mathematical-expressions",
            "target": os.path.join(target_folder, "CROHME"),
            "check": lambda p: any(f.endswith('.inkml') for r, d, files in os.walk(p) for f in files)
        },
        {
            "name": "shahrukhkhan/im2latex100k", 
            "target": os.path.join(target_folder, "IM2LATEX"),
            "check": lambda p: any(f.endswith('.csv') for f in os.listdir(p)) if os.path.exists(p) else False
        }
    ]
    
    for dataset in datasets:
        dataset_name = dataset["name"]
        target_path = dataset["target"]
        check_func = dataset["check"]
        
        if os.path.exists(target_path) and check_func(target_path):
            print(f"[OK] {os.path.basename(target_path)} da ton tai")
            continue
        
        if os.path.exists(target_path) and not os.listdir(target_path):
            os.rmdir(target_path)
        
        print(f"[DOWNLOAD] {dataset_name}...")
        if not download_single_dataset(dataset_name, target_path):
            print(f"[ERROR] Khong the tai {dataset_name}")
            raise SystemExit(1)
        print(f"[OK] Da tai {os.path.basename(target_path)}")
    
    return True


def parse_inkml_traces(inkml_file):
    """Parse InkML file va trích xuat traces"""
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
    """Lay ground truth (công thức LaTeX) tu file InkML"""
    try:
        tree = ET.parse(inkml_file)
        root = tree.getroot()
        ns = {'ink': 'http://www.w3.org/2003/InkML'}
        
        for annotation in root.findall('.//ink:annotation', ns):
            if annotation.get('type') == 'truth':
                truth = annotation.text.strip() if annotation.text else ""
                truth = truth.strip('$').strip()
                return truth
        
        return None
    except Exception:
        return None


def traces_to_image(traces, img_size=(400, 400), padding=20, line_width=2):
    """Chuyen doi traces thanh anh"""
    if not traces:
        return None
    
    try:
        all_points = [p for trace in traces for p in trace]
        if not all_points:
            return None
        
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        width = max_x - min_x
        height = max_y - min_y
        
        if width == 0 or height == 0:
            return None
        
        scale = min((img_size[0] - 2*padding) / width, 
                    (img_size[1] - 2*padding) / height)
        
        img = Image.new('RGB', img_size, 'white')
        draw = ImageDraw.Draw(img)
        
        for trace in traces:
            if len(trace) < 2:
                continue
            
            scaled_trace = [
                (
                    (x - min_x) * scale + padding,
                    (y - min_y) * scale + padding
                )
                for x, y in trace
            ]
            
            for i in range(len(scaled_trace) - 1):
                draw.line([scaled_trace[i], scaled_trace[i+1]], 
                         fill='black', width=line_width)
        
        return img
    except Exception:
        return None


def process_crohme_dataset(crohme_path, output_dir="data/preprocessed/crohme"):
    """Xu ly dataset CROHME va tao cau truc dataset hoan chinh"""
    crohme_path = Path(crohme_path)
    output_path = Path(output_dir)
    
    ground_truth_dir = output_path / "ground_truth"
    non_ground_truth_dir = output_path / "non_ground_truth"
    
    gt_images_dir = ground_truth_dir / "images"
    ngt_images_dir = non_ground_truth_dir / "images"
    
    gt_images_dir.mkdir(parents=True, exist_ok=True)
    ngt_images_dir.mkdir(parents=True, exist_ok=True)
    
    gt_data = []
    ngt_data = []
    
    inkml_files = list(crohme_path.rglob("*.inkml"))
    inkml_files = [f for f in inkml_files if output_dir not in str(f)]
    
    if not inkml_files:
        print(f"[WARNING] Khong tim thay file InkML nao")
        return {'ground_truth': 0, 'non_ground_truth': 0, 'errors': 0, 'total': 0}
    
    print(f"[INFO] Xu ly {len(inkml_files)} file InkML...")
    
    gt_count = 0
    ngt_count = 0
    error_count = 0
    
    for inkml_file in tqdm(inkml_files, desc="Processing CROHME"):
        traces = parse_inkml_traces(inkml_file)
        if not traces:
            error_count += 1
            continue
        
        image_name = inkml_file.stem + ".png"
        img = traces_to_image(traces)
        
        if img is None:
            error_count += 1
            continue
        
        truth = get_ground_truth(inkml_file)
        
        if truth:
            img_path = gt_images_dir / image_name
            img.save(img_path)
            gt_data.append({'formula': truth, 'image': image_name})
            gt_count += 1
        else:
            img_path = ngt_images_dir / image_name
            img.save(img_path)
            ngt_data.append({'image': image_name})
            ngt_count += 1
    
    if gt_data:
        pd.DataFrame(gt_data).to_csv(ground_truth_dir / "dataset.csv", index=False, encoding='utf-8')
    
    if ngt_data:
        pd.DataFrame(ngt_data).to_csv(non_ground_truth_dir / "dataset.csv", index=False, encoding='utf-8')
    
    print(f"[OK] CROHME: {gt_count} GT | {ngt_count} NGT | {error_count} errors")
    
    return {
        'ground_truth': gt_count,
        'non_ground_truth': ngt_count,
        'errors': error_count,
        'total': len(inkml_files)
    }


def merge_datasets(crohme_processed_path, im2latex_path, output_path="data/final_dataset"):
    """Hop nhat CROHME va IM2LATEX vao final_dataset"""
    crohme_path = Path(crohme_processed_path)
    im2latex_path = Path(im2latex_path)
    final_path = Path(output_path)
    
    gt_dir = final_path / "ground_truth" / "images"
    ngt_dir = final_path / "non_ground_truth" / "images"
    gt_dir.mkdir(parents=True, exist_ok=True)
    ngt_dir.mkdir(parents=True, exist_ok=True)
    
    gt_data = []
    ngt_data = []
    
    print("[INFO] Merge datasets...")
    
    # Xu ly CROHME
    crohme_gt_csv = crohme_path / "ground_truth" / "dataset.csv"
    crohme_ngt_csv = crohme_path / "non_ground_truth" / "dataset.csv"
    crohme_gt_img_dir = crohme_path / "ground_truth" / "images"
    crohme_ngt_img_dir = crohme_path / "non_ground_truth" / "images"
    
    crohme_gt_count = 0
    crohme_ngt_count = 0
    
    if crohme_gt_csv.exists():
        df = pd.read_csv(crohme_gt_csv)
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Merging CROHME GT"):
            src = crohme_gt_img_dir / row['image']
            dst = gt_dir / f"CROHME_{row['image']}"
            if src.exists():
                Image.open(src).save(dst)
                gt_data.append({'formula': row['formula'], 'image': dst.name})
                crohme_gt_count += 1
    
    if crohme_ngt_csv.exists():
        df = pd.read_csv(crohme_ngt_csv)
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Merging CROHME NGT"):
            src = crohme_ngt_img_dir / row['image']
            dst = ngt_dir / f"CROHME_{row['image']}"
            if src.exists():
                Image.open(src).save(dst)
                ngt_data.append({'image': dst.name})
                crohme_ngt_count += 1
    
    # Xu ly IM2LATEX
    im2latex_csv = im2latex_path / "dataset.csv"
    im2latex_img_dir = im2latex_path / "dataset_images"
    
    im2latex_gt_count = 0
    im2latex_ngt_count = 0
    
    if im2latex_csv.exists() and im2latex_img_dir.exists():
        df = pd.read_csv(im2latex_csv)
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Merging IM2LATEX"):
            src = im2latex_img_dir / row['image']
            if not src.exists():
                continue
                
            if 'formula' in row and pd.notna(row['formula']) and str(row['formula']).strip():
                dst = gt_dir / f"IM2LATEX_{row['image']}"
                Image.open(src).save(dst)
                gt_data.append({'formula': row['formula'], 'image': dst.name})
                im2latex_gt_count += 1
            else:
                dst = ngt_dir / f"IM2LATEX_{row['image']}"
                Image.open(src).save(dst)
                ngt_data.append({'image': dst.name})
                im2latex_ngt_count += 1
    
    # Luu CSV tong hop
    if gt_data:
        pd.DataFrame(gt_data).to_csv(final_path / "ground_truth" / "dataset.csv", 
                                      index=False, encoding='utf-8')
    
    if ngt_data:
        pd.DataFrame(ngt_data).to_csv(final_path / "non_ground_truth" / "dataset.csv", 
                                       index=False, encoding='utf-8')
    
    print(f"[OK] Final dataset: {len(gt_data)} GT | {len(ngt_data)} NGT")
    print(f"     CROHME: {crohme_gt_count} GT, {crohme_ngt_count} NGT")
    print(f"     IM2LATEX: {im2latex_gt_count} GT, {im2latex_ngt_count} NGT")
    
    return {
        'ground_truth': len(gt_data),
        'non_ground_truth': len(ngt_data),
        'output_dir': str(final_path)
    }


def main():
    """Ham chinh de chay toan bo pipeline"""
    print("="*80)
    print("DATASET PROCESSING PIPELINE - CROHME & IM2LATEX")
    print("="*80)
    
    # Buoc 1: Tai datasets
    print("\n[STEP 1/3] Download datasets")
    print("-"*80)
    download_datasets(target_folder="data")
    
    # Buoc 2: Xu ly CROHME (InkML -> Images)
    print("\n[STEP 2/3] Process CROHME (InkML -> Images)")
    print("-"*80)
    crohme_path = "data/CROHME"
    if os.path.exists(crohme_path):
        process_crohme_dataset(
            crohme_path=crohme_path,
            output_dir="data/preprocessed/crohme"
        )
    else:
        print(f"[SKIP] CROHME khong ton tai tai {crohme_path}")
    
    # Buoc 3: Merge CROHME + IM2LATEX -> final_dataset
    print("\n[STEP 3/3] Merge datasets -> final_dataset")
    print("-"*80)
    im2latex_path = "data/IM2LATEX"
    if os.path.exists(im2latex_path):
        merge_result = merge_datasets(
            crohme_processed_path="data/preprocessed/crohme",
            im2latex_path=im2latex_path,
            output_path="data/final_dataset"
        )
        
        print("\n" + "="*80)
        print("HOAN TAT!")
        print("="*80)
        print(f"Location: {merge_result['output_dir']}")
        print(f"  Ground truth: {merge_result['ground_truth']} samples")
        print(f"  Non ground truth: {merge_result['non_ground_truth']} samples")
        print("="*80)
    else:
        print(f"[SKIP] IM2LATEX khong ton tai tai {im2latex_path}")


if __name__ == "__main__":
    main()
