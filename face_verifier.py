"""Face verification engine for Chibi Ram using OpenCV YuNet + SFace."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
ENROLLED_EMBEDDINGS_FILE = DATA_DIR / "haru_face_embeddings.npy"

YUNET_MODEL_FILE = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL_FILE = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

YUNET_DOWNLOAD_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
SFACE_DOWNLOAD_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
)

# SFace cosine similarity threshold: >= 0.363 is considered same person.
# We set default threshold to 0.38 for tight sibling/stranger rejection.
DEFAULT_MATCH_THRESHOLD = 0.38


def ensure_models() -> None:
    """Ensure ONNX models are present, downloading if necessary."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not YUNET_MODEL_FILE.exists():
        print(f"[FaceAuth] Downloading YuNet model to {YUNET_MODEL_FILE}...")
        try:
            urllib.request.urlretrieve(YUNET_DOWNLOAD_URL, YUNET_MODEL_FILE)
            print("[FaceAuth] YuNet download complete.")
        except Exception as exc:
            print(f"[FaceAuth ERROR] Failed to download YuNet: {exc}")

    if not SFACE_MODEL_FILE.exists():
        print(f"[FaceAuth] Downloading SFace model to {SFACE_MODEL_FILE}...")
        try:
            urllib.request.urlretrieve(SFACE_DOWNLOAD_URL, SFACE_MODEL_FILE)
            print("[FaceAuth] SFace download complete.")
        except Exception as exc:
            print(f"[FaceAuth ERROR] Failed to download SFace: {exc}")


class FaceVerifier:
    def __init__(self, match_threshold: float = DEFAULT_MATCH_THRESHOLD) -> None:
        self.match_threshold = match_threshold
        self.detector: Any = None
        self.recognizer: Any = None
        self.enrolled_embeddings: list[np.ndarray] = []
        self._init_models()
        self._load_enrolled()

    def _init_models(self) -> None:
        try:
            ensure_models()
            if YUNET_MODEL_FILE.exists() and SFACE_MODEL_FILE.exists():
                self.detector = cv2.FaceDetectorYN.create(
                    str(YUNET_MODEL_FILE),
                    "",
                    (320, 320),
                    score_threshold=0.6,
                    nms_threshold=0.3,
                    top_k=5000,
                )
                self.recognizer = cv2.FaceRecognizerSF.create(str(SFACE_MODEL_FILE), "")
                print("[FaceAuth] YuNet & SFace models initialized successfully.")
            else:
                print("[FaceAuth WARN] ONNX model files missing; face verification will run in bypass mode.")
        except Exception as exc:
            print(f"[FaceAuth WARN] Face models failed to initialize ({exc}); running in bypass mode.")
            self.detector = None
            self.recognizer = None

    def _load_enrolled(self) -> None:
        self.enrolled_embeddings = []
        if ENROLLED_EMBEDDINGS_FILE.exists():
            try:
                arr = np.load(ENROLLED_EMBEDDINGS_FILE, allow_pickle=True)
                if isinstance(arr, np.ndarray) and arr.size > 0:
                    if arr.ndim == 1 and arr.shape[0] == 128:
                        self.enrolled_embeddings.append(arr.reshape(1, 128).astype(np.float32))
                    elif arr.ndim == 2:
                        for row in arr:
                            self.enrolled_embeddings.append(row.reshape(1, 128).astype(np.float32))
                print(f"[FaceAuth] Loaded {len(self.enrolled_embeddings)} enrolled embedding(s) for Haru-sama.")
            except Exception as exc:
                print(f"[FaceAuth WARN] Failed to load {ENROLLED_EMBEDDINGS_FILE}: {exc}")

    def save_enrolled(self) -> None:
        if not self.enrolled_embeddings:
            if ENROLLED_EMBEDDINGS_FILE.exists():
                ENROLLED_EMBEDDINGS_FILE.unlink(missing_ok=True)
            return

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        stacked = np.vstack(self.enrolled_embeddings)
        np.save(ENROLLED_EMBEDDINGS_FILE, stacked)
        print(f"[FaceAuth] Saved {len(self.enrolled_embeddings)} enrolled embedding(s) to {ENROLLED_EMBEDDINGS_FILE}.")

    def clear_enrolled(self) -> int:
        count = len(self.enrolled_embeddings)
        self.enrolled_embeddings.clear()
        if ENROLLED_EMBEDDINGS_FILE.exists():
            try:
                ENROLLED_EMBEDDINGS_FILE.unlink(missing_ok=True)
            except Exception:
                pass
        print(f"[FaceAuth] Cleared {count} enrolled embeddings.")
        return count

    @property
    def num_enrolled(self) -> int:
        return len(self.enrolled_embeddings)

    def extract_face_embedding(self, image_bytes: bytes) -> tuple[np.ndarray | None, str]:
        """Detects the most prominent face and extracts its 128-d feature vector."""
        if not self.detector or not self.recognizer:
            return None, "Face models not loaded"

        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return None, "Failed to decode image bytes"

            h, w, _ = img.shape
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(img)

            if faces is None or len(faces) == 0:
                return None, "No face detected"

            # Select face with largest area (width * height = face[2] * face[3])
            best_face = max(faces, key=lambda f: float(f[2] * f[3]))
            aligned = self.recognizer.alignCrop(img, best_face)
            feature = self.recognizer.feature(aligned)
            return feature.astype(np.float32), "OK"
        except Exception as exc:
            return None, f"Extraction failed: {exc}"

    def enroll_face(self, image_bytes: bytes, replace_all: bool = False) -> tuple[bool, str, int]:
        """Enrolls a reference photo of Haru-sama."""
        feature, status = self.extract_face_embedding(image_bytes)
        if feature is None:
            return False, f"Enrollment failed: {status}", len(self.enrolled_embeddings)

        if replace_all:
            self.enrolled_embeddings.clear()

        # Check if this sample is already close to an existing enrolled sample to avoid duplicates
        duplicate = False
        for existing in self.enrolled_embeddings:
            score = float(self.recognizer.match(existing, feature, cv2.FaceRecognizerSF_FR_COSINE))
            if score > 0.85:
                duplicate = True
                break

        if not duplicate:
            self.enrolled_embeddings.append(feature)
            # Cap at 8 reference samples for fast matching
            if len(self.enrolled_embeddings) > 8:
                self.enrolled_embeddings.pop(0)
            self.save_enrolled()

        return True, "Haru-sama's face enrolled successfully!", len(self.enrolled_embeddings)

    def verify_face(self, image_bytes: bytes) -> tuple[str, float]:
        """
        Verifies identity against enrolled references of Haru-sama.

        Returns:
            (identity, similarity_score)
            identity: "HARU", "STRANGER", "NO_FACE", or "NOT_ENROLLED"
        """
        if not self.enrolled_embeddings:
            # Not enrolled yet: bypass so Ram functions until Haru enrolls
            return "NOT_ENROLLED", 0.0

        feature, status = self.extract_face_embedding(image_bytes)
        if feature is None:
            return "NO_FACE", 0.0

        scores = [
            float(self.recognizer.match(ref, feature, cv2.FaceRecognizerSF_FR_COSINE))
            for ref in self.enrolled_embeddings
        ]
        max_score = max(scores) if scores else 0.0

        if max_score >= self.match_threshold:
            print(f"[FaceAuth] Identity: HARU (Score: {max_score:.3f} >= {self.match_threshold:.2f})")
            return "HARU", max_score

        print(f"[FaceAuth] Identity: STRANGER (Score: {max_score:.3f} < {self.match_threshold:.2f})")
        return "STRANGER", max_score


# Global Singleton
face_verifier = FaceVerifier()
