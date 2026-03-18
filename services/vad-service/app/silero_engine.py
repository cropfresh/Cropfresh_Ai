"""Silero ONNX scoring for the Sprint 08 VAD service."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import VadServiceSettings


class SileroOnnxScorer:
    """Load and score Silero ONNX frames using CPU inference."""

    def __init__(self, settings: VadServiceSettings) -> None:
        self.settings = settings
        self._session = None
        self._use_v5_format = False
        self._state: np.ndarray | None = None
        self._h: np.ndarray | None = None
        self._c: np.ndarray | None = None

    async def load(self) -> None:
        """Load the model artifact if one is mounted into the repo or container."""
        import onnxruntime as ort

        model_path = self._resolve_model_path()
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = 1

        self._session = ort.InferenceSession(
            str(model_path),
            session_options=session_options,
            providers=["CPUExecutionProvider"],
        )

        input_names = {input_value.name for input_value in self._session.get_inputs()}
        self._use_v5_format = "state" in input_names
        self._reset_state()

    def score_pcm16(self, pcm16: bytes, sample_rate: int) -> float:
        """Score one PCM16 frame and return a probability in [0.0, 1.0]."""
        if self._session is None:
            raise RuntimeError("SileroOnnxScorer is not loaded")

        audio = np.frombuffer(pcm16[: len(pcm16) - (len(pcm16) % 2)], dtype=np.int16)
        audio = audio.astype(np.float32) / 32768.0

        if audio.size < self.settings.frame_samples:
            audio = np.pad(audio, (0, self.settings.frame_samples - audio.size))
        elif audio.size > self.settings.frame_samples:
            audio = audio[: self.settings.frame_samples]

        sr = np.array([sample_rate], dtype=np.int64)
        if self._use_v5_format:
            assert self._state is not None
            outputs = self._session.run(
                None,
                {"input": audio.reshape(1, -1), "state": self._state, "sr": sr},
            )
            probability, self._state = outputs
        else:
            assert self._h is not None
            assert self._c is not None
            outputs = self._session.run(
                None,
                {"input": audio.reshape(1, -1), "h": self._h, "c": self._c, "sr": sr},
            )
            probability, self._h, self._c = outputs

        return float(probability[0][0])

    def _resolve_model_path(self) -> Path:
        candidates = [
            Path(self.settings.model_path) if self.settings.model_path else None,
            Path.cwd() / "models" / "silero_vad.onnx",
            Path.home() / ".cache" / "silero_vad" / "silero_vad.onnx",
            Path.cwd() / "src" / "voice" / "vad" / "models" / "silero_vad.onnx",
        ]

        for candidate in candidates:
            if candidate is not None and candidate.exists():
                return candidate

        # ! Service readiness stays false until an explicit Silero artifact is mounted.
        raise FileNotFoundError("No Silero ONNX model was found for the VAD service")

    def _reset_state(self) -> None:
        if self._use_v5_format:
            state_dim = 128 if self.settings.sample_rate == 16000 else 64
            self._state = np.zeros((2, 1, state_dim), dtype=np.float32)
            return

        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)
