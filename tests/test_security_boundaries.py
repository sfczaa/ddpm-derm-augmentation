"""Regression checks for upload bounds and restricted checkpoint loading."""

import asyncio
import io
import pickle
import random
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from fastapi import HTTPException
from starlette.datastructures import Headers

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from app.uploads import MAX_REQUEST_BYTES, image_upload
from ddpm_derm.checkpoint import load_checkpoint


class RequestStub:
    def __init__(self, chunks, headers=None):
        self.headers = Headers(headers or {})
        self.chunks = chunks

    async def stream(self):
        for chunk in self.chunks:
            yield chunk


class SecurityBoundaryTests(unittest.TestCase):
    def test_oversized_stream_is_rejected_before_multipart_parsing(self):
        for headers in ({}, {"content-length": "1"}):
            request = RequestStub([b"x" * MAX_REQUEST_BYTES, b"x"], headers)
            with patch("app.uploads.MemoryMultipartParser") as parser:
                with self.assertRaises(HTTPException) as exc:
                    asyncio.run(anext(image_upload(request)))
            self.assertEqual(exc.exception.status_code, 413)
            parser.assert_not_called()

    def test_upload_above_default_spool_threshold_stays_in_memory(self):
        payload = b"x" * (2 * 1024 * 1024)
        body = (b'--boundary\r\nContent-Disposition: form-data; name="image"; '
                b'filename="test.png"\r\nContent-Type: image/png\r\n\r\n'
                + payload + b'\r\n--boundary--\r\n')
        request = RequestStub([body], {"content-type": "multipart/form-data; boundary=boundary"})

        async def check():
            dependency = image_upload(request)
            upload = await anext(dependency)
            try:
                self.assertEqual(await upload.read(), payload)
                self.assertFalse(upload.file._rolled)
            finally:
                await dependency.aclose()
            self.assertTrue(upload.file.closed)

        asyncio.run(check())

    def test_legacy_rng_checkpoint_round_trip(self):
        rng = np.random.get_state()
        stream = io.BytesIO()
        torch.save({"rng_state": {"numpy": rng, "python": random.getstate()},
                    "weight": torch.tensor([1.0]), "score": np.float64(0.5)}, stream)
        stream.seek(0)
        restored = load_checkpoint(stream)
        np.testing.assert_array_equal(restored["rng_state"]["numpy"][1], rng[1])
        self.assertEqual(restored["rng_state"]["python"], random.getstate())
        self.assertEqual(restored["weight"].item(), 1.0)
        self.assertEqual(restored["score"], 0.5)

    def test_arbitrary_pickle_callable_is_rejected(self):
        class Unsupported:
            def __reduce__(self):
                return eval, ("1 + 1",)

        stream = io.BytesIO()
        torch.save(Unsupported(), stream)
        stream.seek(0)
        with self.assertRaises(pickle.UnpicklingError):
            load_checkpoint(stream)


if __name__ == "__main__":
    unittest.main()
