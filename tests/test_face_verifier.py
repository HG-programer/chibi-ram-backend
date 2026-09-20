"""Unit tests for FaceVerifier module."""

import unittest
import numpy as np
from pathlib import Path
from face_verifier import FaceVerifier


class TestFaceVerifier(unittest.TestCase):
    def setUp(self):
        self.verifier = FaceVerifier(match_threshold=0.38)
        self.verifier.clear_enrolled()

    def tearDown(self):
        self.verifier.clear_enrolled()

    def test_empty_verifier_returns_not_enrolled(self):
        self.assertEqual(self.verifier.num_enrolled, 0)
        identity, score = self.verifier.verify_face(b"FAKEDATA")
        self.assertEqual(identity, "NOT_ENROLLED")
        self.assertEqual(score, 0.0)

    def test_direct_embedding_matching(self):
        # Create a synthetic 128-d reference feature
        ref_feat = np.random.randn(1, 128).astype(np.float32)
        ref_feat = ref_feat / np.linalg.norm(ref_feat)
        self.verifier.enrolled_embeddings.append(ref_feat)
        self.assertEqual(self.verifier.num_enrolled, 1)

        # Test self match using SFace recognizer if available
        if self.verifier.recognizer:
            score = self.verifier.recognizer.match(ref_feat, ref_feat, 0)
            self.assertAlmostEqual(score, 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
