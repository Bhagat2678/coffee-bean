"""
Module: arcface.py
Description: ArcFace-inspired front-back bean mapping infrastructure.

Maps individual beans from a front-facing image to their counterparts in a
back-facing image using learned feature embeddings.  The initial version uses
a ResNet-18 backbone as a generic feature extractor.  Once a dedicated ArcFace
model is trained on paired coffee-bean crops, swap the backbone via
``ArcFaceMapper.load_model(path)``.

Workflow
--------
1.  Detect beans in both front and back images  (done by ObjectDetector).
2.  Crop each detected bean region.
3.  Extract an embedding vector for every crop.
4.  Match front→back using cosine similarity + Hungarian assignment.
5.  Optionally save matched pairs to disk for training data collection.
"""

import os
import cv2
import numpy as np
from pathlib import Path

# Optional deep-learning imports — gracefully degrade if missing
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    import torchvision.models as models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class ArcFaceMapper:
    """
    Extract visual embeddings from bean crops and match front ↔ back.

    Parameters
    ----------
    model_path : str or None
        Path to a trained ArcFace/embedding model (.pt).  If *None*, a
        pre-trained ResNet-18 (ImageNet) is used as a generic feature
        extractor — good enough for coarse matching and dataset building.
    embedding_dim : int
        Dimensionality of the output embedding vector (default 512).
    device : str
        ``'cuda'`` or ``'cpu'``.
    """

    CROP_SIZE = 112  # Standard ArcFace input size

    def __init__(self, model_path=None, embedding_dim=512, device=None):
        self.embedding_dim = embedding_dim
        self.model_path = model_path
        self.model = None

        if not TORCH_AVAILABLE:
            print("[WARN] PyTorch not available — ArcFaceMapper will use histogram fallback.")
            self.device = "cpu"
            return

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if model_path and os.path.exists(model_path):
            self._load_custom_model(model_path)
        else:
            self._load_resnet_backbone()

        # Standard ImageNet normalisation
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((self.CROP_SIZE, self.CROP_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    # ── Model loading ──────────────────────────────────────────────────

    def _load_resnet_backbone(self):
        """Use ResNet-18 (minus FC layer) as a generic feature extractor."""
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove final FC, keeping avgpool output → 512-d vector
        self.model = nn.Sequential(*list(base.children())[:-1])
        self.model.eval()
        self.model.to(self.device)
        print("[INFO] ArcFaceMapper: using ResNet-18 backbone (generic features)")

    def _load_custom_model(self, path):
        """Load a trained ArcFace-style embedding model."""
        try:
            self.model = torch.load(path, map_location=self.device)
            self.model.eval()
            print(f"[INFO] ArcFaceMapper: loaded custom model from {path}")
        except Exception as e:
            print(f"[WARN] Failed to load custom model ({e}), falling back to ResNet-18")
            self._load_resnet_backbone()

    def load_model(self, path):
        """Hot-swap the embedding model at runtime."""
        self._load_custom_model(path)

    # ── Embedding extraction ───────────────────────────────────────────

    def _crop_bean(self, image, box):
        """Crop and return the bean region from the image (BGR)."""
        x1, y1, x2, y2 = map(int, box)
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((self.CROP_SIZE, self.CROP_SIZE, 3), dtype=np.uint8)
        return crop

    def _embed_with_model(self, crop_bgr):
        """Run the deep model on a single BGR crop → 1-D numpy embedding."""
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model(tensor)
        return emb.squeeze().cpu().numpy().flatten()

    def _embed_with_histogram(self, crop_bgr):
        """Fallback: colour + shape histogram when PyTorch is missing."""
        if crop_bgr.size == 0:
            return np.zeros(256 * 3 + 64, dtype=np.float32)

        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        hists = []
        for ch in range(3):
            h = cv2.calcHist([hsv], [ch], None, [256], [0, 256])
            h = cv2.normalize(h, h).flatten()
            hists.append(h)

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (8, 8))
        hists.append(gray.flatten().astype(np.float32) / 255.0)

        return np.concatenate(hists)

    def extract_embeddings(self, image, detections):
        """
        Extract an embedding for every detected bean.

        Parameters
        ----------
        image : np.ndarray
            Full image (BGR, from cv2.imread).
        detections : list[dict]
            Each dict must have ``'box': [x1, y1, x2, y2]``.

        Returns
        -------
        list[dict]
            Each entry: ``{'index': int, 'box': list, 'embedding': np.ndarray}``.
        """
        results = []
        for i, det in enumerate(detections):
            box = det.get("box")
            if not box:
                continue
            crop = self._crop_bean(image, box)
            if self.model is not None:
                emb = self._embed_with_model(crop)
            else:
                emb = self._embed_with_histogram(crop)
            results.append({
                "index": i,
                "box": box,
                "embedding": emb,
            })
        return results

    # ── Front ↔ Back matching ──────────────────────────────────────────

    @staticmethod
    def _cosine_similarity(a, b):
        """Cosine similarity between two 1-D vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def match_front_back(self, front_embeddings, back_embeddings, threshold=0.5):
        """
        Match front beans to back beans using cosine similarity.

        Uses the Hungarian algorithm for optimal 1-to-1 assignment when
        scipy is available; otherwise a greedy approach.

        Parameters
        ----------
        front_embeddings : list[dict]
            Output of ``extract_embeddings`` on the front image.
        back_embeddings : list[dict]
            Output of ``extract_embeddings`` on the back image.
        threshold : float
            Minimum similarity to accept a match.

        Returns
        -------
        list[dict]
            Each: ``{'front_index', 'back_index', 'similarity',
                      'front_box', 'back_box'}``.
        """
        n_front = len(front_embeddings)
        n_back = len(back_embeddings)
        if n_front == 0 or n_back == 0:
            return []

        # Build similarity matrix
        sim_matrix = np.zeros((n_front, n_back), dtype=np.float32)
        for i, fe in enumerate(front_embeddings):
            for j, be in enumerate(back_embeddings):
                sim_matrix[i, j] = self._cosine_similarity(fe["embedding"], be["embedding"])

        # Optimal assignment
        if SCIPY_AVAILABLE:
            cost = 1.0 - sim_matrix  # minimise cost = maximise similarity
            row_idx, col_idx = linear_sum_assignment(cost)
        else:
            # Greedy fallback
            row_idx, col_idx = self._greedy_match(sim_matrix)

        pairs = []
        for ri, ci in zip(row_idx, col_idx):
            sim = float(sim_matrix[ri, ci])
            if sim >= threshold:
                pairs.append({
                    "front_index": int(front_embeddings[ri]["index"]),
                    "back_index": int(back_embeddings[ci]["index"]),
                    "similarity": round(sim, 4),
                    "front_box": front_embeddings[ri]["box"],
                    "back_box": back_embeddings[ci]["box"],
                })

        pairs.sort(key=lambda p: p["similarity"], reverse=True)
        return pairs

    @staticmethod
    def _greedy_match(sim_matrix):
        """Simple greedy 1-to-1 matching by descending similarity."""
        n_rows, n_cols = sim_matrix.shape
        flat = []
        for i in range(n_rows):
            for j in range(n_cols):
                flat.append((sim_matrix[i, j], i, j))
        flat.sort(reverse=True)

        used_rows = set()
        used_cols = set()
        rows, cols = [], []
        for sim, i, j in flat:
            if i not in used_rows and j not in used_cols:
                rows.append(i)
                cols.append(j)
                used_rows.add(i)
                used_cols.add(j)
        return rows, cols

    # ── Dataset saving ─────────────────────────────────────────────────

    def save_paired_dataset(self, front_image, back_image, pairs,
                            front_detections, back_detections, output_dir):
        """
        Save matched front/back bean crops as paired training data.

        Creates a directory structure:
            output_dir/
                pair_001/
                    front.jpg
                    back.jpg
                    meta.txt
                pair_002/
                    ...
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        for idx, pair in enumerate(pairs):
            pair_dir = out / f"pair_{idx + 1:04d}"
            pair_dir.mkdir(exist_ok=True)

            # Front crop
            fi = pair["front_index"]
            fbox = front_detections[fi]["box"]
            fcrop = self._crop_bean(front_image, fbox)
            fcrop = cv2.resize(fcrop, (self.CROP_SIZE, self.CROP_SIZE))
            cv2.imwrite(str(pair_dir / "front.jpg"), fcrop)

            # Back crop
            bi = pair["back_index"]
            bbox = back_detections[bi]["box"]
            bcrop = self._crop_bean(back_image, bbox)
            bcrop = cv2.resize(bcrop, (self.CROP_SIZE, self.CROP_SIZE))
            cv2.imwrite(str(pair_dir / "back.jpg"), bcrop)

            # Metadata
            meta = (
                f"front_index={fi}\n"
                f"back_index={bi}\n"
                f"similarity={pair['similarity']}\n"
                f"front_box={fbox}\n"
                f"back_box={bbox}\n"
            )
            (pair_dir / "meta.txt").write_text(meta)

        print(f"[INFO] Saved {len(pairs)} bean pairs to {out}")
        return str(out)
