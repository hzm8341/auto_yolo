"""
Dual Engine Inspector: YOLO + SubspaceAD 融合推理
Phase 3: 双引擎融合

融合逻辑: YOLO 检测到缺陷 OR SubspaceAD 异常分数 > 阈值 → 报缺陷
"""
import os, pickle, numpy as np
from pathlib import Path
from PIL import Image
import torch
import cv2
from sklearn.metrics import roc_curve, roc_auc_score
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModel


class SubspaceADInferencer:
    """SubspaceAD 推理器：基于 DINOv2 + PCA 重建残差"""

    def __init__(self, pca_path, model_ckpt="facebook/dinov2-base"):
        self.processor = AutoImageProcessor.from_pretrained(model_ckpt)
        self.model = AutoModel.from_pretrained(model_ckpt).eval().cuda()
        with open(pca_path, "rb") as f:
            self.pca = pickle.load(f)
        print(f"SubspaceAD loaded: {model_ckpt}, PCA components: {self.pca.n_components_}")

    @torch.no_grad()
    def get_anomaly_map(self, image_path):
        """返回像素级异常热图 (H, W)"""
        img = Image.open(image_path).convert("RGB")
        w, h = img.size

        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs = self.model(**inputs)

        patch_features = outputs.last_hidden_state[:, 1:, :].squeeze(0).detach().cpu().numpy()

        # 投影到 PCA 子空间，计算重建残差
        projected = self.pca.transform(patch_features)
        reconstructed = self.pca.inverse_transform(projected)
        residuals = np.sum((patch_features - reconstructed) ** 2, axis=1)

        h_p = w_p = int(np.sqrt(len(residuals)))
        patch_grid = residuals.reshape(h_p, w_p)

        anomaly_map = cv2.resize(patch_grid, (w, h), interpolation=cv2.INTER_CUBIC)
        return anomaly_map

    def get_anomaly_score(self, image_path):
        """返回图像级异常分数（取热图最大值）"""
        anomaly_map = self.get_anomaly_map(image_path)
        return float(anomaly_map.max())


class SubspaceADFitter:
    """SubspaceAD PCA 拟合器"""

    def __init__(self, model_ckpt="facebook/dinov2-base", pca_ev=0.99):
        self.processor = AutoImageProcessor.from_pretrained(model_ckpt)
        self.model = AutoModel.from_pretrained(model_ckpt).eval().cuda()
        self.pca_ev = pca_ev

    @torch.no_grad()
    def extract_features(self, image_path):
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs = self.model(**inputs)
        patch_features = outputs.last_hidden_state[:, 1:, :].squeeze(0).detach().cpu().numpy()
        return patch_features

    def fit(self, good_image_paths, aug_count=30):
        from torchvision.transforms.functional import rotate
        all_features = []

        for path in good_image_paths:
            all_features.append(self.extract_features(path))
            img = Image.open(path).convert("RGB")
            angles = np.linspace(0, 360, aug_count, endpoint=False)
            for angle in angles[1:]:
                rotated = rotate(img, float(angle))
                inputs = self.processor(images=rotated, return_tensors="pt")
                inputs = {k: v.cuda() for k, v in inputs.items()}
                outputs = self.model(**inputs)
                patch_features = outputs.last_hidden_state[:, 1:, :].squeeze(0).detach().cpu().numpy()
                all_features.append(patch_features)

        all_features = np.concatenate(all_features, axis=0)
        print(f"Fitting PCA: {all_features.shape}, ev={self.pca_ev}")

        from sklearn.decomposition import PCA
        self.pca = PCA(n_components=self.pca_ev)
        self.pca.fit(all_features)
        print(f"PCA fitted: {self.pca.n_components_} components")
        return self

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.pca, f)
        print(f"PCA saved to {path}")


class DualEngineInspector:
    """双引擎检测器: YOLO + SubspaceAD OR 融合"""

    def __init__(self, yolo_path, pca_path,
                 yolo_conf=0.25, anomaly_threshold=None,
                 model_ckpt="facebook/dinov2-base"):
        self.yolo = YOLO(yolo_path)
        self.subspace = SubspaceADInferencer(pca_path, model_ckpt)
        self.yolo_conf = yolo_conf
        self.anomaly_threshold = anomaly_threshold

    def inspect(self, image_path):
        results = {
            "image_path": image_path,
            "yolo_detections": [],
            "anomaly_score": 0.0,
            "is_defect": False,
            "trigger": None,
        }

        # 引擎1: YOLO
        yolo_results = self.yolo(image_path, conf=self.yolo_conf)
        if len(yolo_results[0].boxes) > 0:
            results["yolo_detections"] = yolo_results[0].boxes
            results["is_defect"] = True
            results["trigger"] = "yolo"

        # 引擎2: SubspaceAD
        try:
            anomaly_score = self.subspace.get_anomaly_score(image_path)
            results["anomaly_score"] = anomaly_score

            if self.anomaly_threshold is not None and anomaly_score > self.anomaly_threshold:
                results["is_defect"] = True
                if results["trigger"] == "yolo":
                    results["trigger"] = "both"
                else:
                    results["trigger"] = "subspace"
        except Exception as e:
            print(f"SubspaceAD error: {e}")

        return results


def find_pseudo_good_images(yolo_path, val_images, top_n=30):
    """
    用 YOLO 找出最不可能有缺陷的图片作为伪良品。
    策略：YOLO 置信度最低的图片 → 最可能是良品
    """
    yolo = YOLO(yolo_path)
    scores = []
    for img_path in val_images:
        results = yolo(img_path, conf=0.01, verbose=False)
        if len(results[0].boxes) == 0:
            scores.append((img_path, 0.0))  # 无检测 = 最高良品分
        else:
            conf = float(results[0].boxes.conf.max())
            scores.append((img_path, conf))

    # 按 YOLO 置信度升序：越低越可能是良品
    scores.sort(key=lambda x: x[1])
    pseudo_good = [s[0] for s in scores[:top_n]]
    return pseudo_good


def calibrate_threshold(inspector, pseudo_good_paths, defect_paths, target_recall=0.95):
    """
    改进的阈值标定：
    - pseudo_good: YOLO 判定为最可能是良品的图
    - defect_paths: 已知有缺陷的图
    """
    print(f"Calibrating with {len(pseudo_good_paths)} pseudo-good, {len(defect_paths)} defect")

    good_scores = []
    for p in pseudo_good_paths:
        try:
            s = inspector.subspace.get_anomaly_score(p)
            good_scores.append(s)
        except:
            pass

    defect_scores = []
    for p in defect_paths:
        try:
            s = inspector.subspace.get_anomaly_score(p)
            defect_scores.append(s)
        except:
            pass

    all_scores = good_scores + defect_scores
    all_labels = [0] * len(good_scores) + [1] * len(defect_scores)

    auroc = roc_auc_score(all_labels, all_scores)
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)

    print(f"SubspaceAD AUROC: {auroc:.4f}")

    # 找满足目标召回率的最小阈值
    valid_idx = np.where(tpr >= target_recall)[0]
    if len(valid_idx) > 0:
        best_threshold = thresholds[valid_idx[0]]
        actual_fpr = fpr[valid_idx[0]]
        print(f"Threshold (recall>={target_recall}): {best_threshold:.4f}, FPR: {actual_fpr:.4f}")
        return best_threshold
    else:
        print("Could not find threshold meeting recall target")
        return None


if __name__ == "__main__":
    import glob

    YOLO_PATH = "runs/detect/runs/baseline/yolov8n_baseline/weights/best.pt"
    PCA_PATH = "models/subspace_pca.pkl"
    GOOD_IMAGES = "data/good_images"
    VAL_IMAGES = "data/val/images"
    os.makedirs("models", exist_ok=True)

    defect_types = ["crazing", "inclusion", "patches", "pittedsurface", "rolled-inscale", "scratches"]
    val_all = sorted(glob.glob(f"{VAL_IMAGES}/*.jpg"))
    defect_paths = [p for p in val_all if any(t in os.path.basename(p) for t in defect_types)]

    # === Step 1: 找伪良品 + 拟合 PCA ===
    print("=" * 60)
    print("Step 1: Finding pseudo-good images + fitting SubspaceAD PCA")
    print("=" * 60)

    # 用 YOLO 置信度最低的图片作为伪良品
    pseudo_good = find_pseudo_good_images(YOLO_PATH, val_all, top_n=30)
    print(f"Pseudo-good images: {len(pseudo_good)}")
    for p in pseudo_good[:3]:
        print(f"  {os.path.basename(p)}")

    fitter = SubspaceADFitter(model_ckpt="facebook/dinov2-base", pca_ev=0.99)
    fitter.fit(pseudo_good, aug_count=30)
    fitter.save(PCA_PATH)

    # === Step 2: 阈值标定 ===
    print("\n" + "=" * 60)
    print("Step 2: Calibrating anomaly threshold")
    print("=" * 60)

    inspector = DualEngineInspector(YOLO_PATH, PCA_PATH, yolo_conf=0.25, anomaly_threshold=None)
    threshold = calibrate_threshold(inspector, pseudo_good, defect_paths[:100])

    # === Step 3: 融合评估 ===
    print("\n" + "=" * 60)
    print("Step 3: Dual engine evaluation")
    print("=" * 60)

    inspector_eval = DualEngineInspector(YOLO_PATH, PCA_PATH, yolo_conf=0.25, anomaly_threshold=threshold)

    yolo_hit = subspace_hit = 0
    total_defect = len(defect_paths[:100])
    total_good = len(pseudo_good)
    fp = 0

    for path in defect_paths[:100]:
        r = inspector_eval.inspect(path)
        if r["trigger"] in ["yolo", "both"]:
            yolo_hit += 1
        if r["trigger"] in ["subspace", "both"]:
            subspace_hit += 1

    for path in pseudo_good:
        r = inspector_eval.inspect(path)
        if r["is_defect"]:
            fp += 1

    yolo_recall = yolo_hit / total_defect
    subspace_recall = subspace_hit / total_defect
    fpr_val = fp / total_good

    print(f"\n--- Dual Engine Results ({total_defect} defect, {total_good} pseudo-good) ---")
    print(f"YOLO recall:      {yolo_recall:.3f} ({yolo_hit}/{total_defect})")
    print(f"SubspaceAD recall: {subspace_recall:.3f} ({subspace_hit}/{total_defect})")
    print(f"OR fusion recall:  {(yolo_hit + subspace_hit)/total_defect:.3f} (union)")
    print(f"FPR (pseudo-good): {fpr_val:.3f} ({fp}/{total_good})")

    print("\nDone!")
