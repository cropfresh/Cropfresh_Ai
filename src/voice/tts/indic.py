"""
Indic TTS Provider
==================
AI4Bharat IndicTTS/IndicF5 implementation for Indian languages.
"""

import io
from loguru import logger

from .models import SynthesisResult


class IndicTTS:
    """
    Text-to-Speech using AI4Bharat IndicTTS.
    
    IndicTTS provides high-quality speech synthesis for 20 Indian languages
    with multiple voice options and emotion support.
    
    Usage:
        tts = IndicTTS()
        result = await tts.synthesize("नमस्ते, कैसे हैं आप?", language="hi")
    """
    
    # Model options
    MODEL_PARLER = "ai4bharat/indic-parler-tts"
    MODEL_F5 = "ai4bharat/IndicF5"
    
    # Language to voice mapping
    LANGUAGE_VOICES = {
        "hi": ["hindi_male_1", "hindi_female_1"],
        "kn": ["kannada_male_1", "kannada_female_1"],
        "te": ["telugu_male_1", "telugu_female_1"],
        "ta": ["tamil_male_1", "tamil_female_1"],
        "ml": ["malayalam_male_1", "malayalam_female_1"],
        "mr": ["marathi_male_1", "marathi_female_1"],
        "gu": ["gujarati_male_1", "gujarati_female_1"],
        "bn": ["bengali_male_1", "bengali_female_1"],
        "pa": ["punjabi_male_1", "punjabi_female_1"],
        "or": ["odia_male_1", "odia_female_1"],
        "en": ["english_male_1", "english_female_1"],
    }
    
    # Sample rate for output
    OUTPUT_SAMPLE_RATE = 22050
    
    def __init__(
        self,
        model_name: str = MODEL_PARLER,
        device: str = "auto",
        use_edge_fallback: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.use_edge_fallback = use_edge_fallback
        
        self._vocoder = None
        self._ema_model = None
        self._tokenizer = None
        self._initialized = False
        
        logger.info(f"IndicTTS initialized with model: {model_name}")
    
    async def _ensure_initialized(self):
        if self._initialized:
            return
            
        try:
            await self._load_model()
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to load local TTS model: {e}")
            raise RuntimeError(f"Failed to load local TTS model: {e}")
    
    async def _load_model(self):
        logger.info(f"Loading IndicTTS model: {self.model_name}")
        
        try:
            import torch
            import torchaudio
            import soundfile as sf
            
            # Monkeypatch torchaudio.load to use soundfile
            def _safe_audio_load(filepath, *args, **kwargs):
                data, sr = sf.read(filepath)
                tensor = torch.from_numpy(data).to(torch.float32)
                if tensor.ndim == 1:
                    tensor = tensor.unsqueeze(0)
                else:
                    tensor = tensor.T
                return tensor, sr
            torchaudio.load = _safe_audio_load
            
            from huggingface_hub import hf_hub_download
            
            if self.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device
            
            from f5_tts.model import DiT
            from f5_tts.infer.utils_infer import load_model, load_vocoder
            
            self._vocoder = load_vocoder("vocos", is_local=False, device=device)
            
            model_cls = DiT
            model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
            
            vocab_path = hf_hub_download(repo_id=self.model_name, filename="checkpoints/vocab.txt")
            
            self._ema_model = load_model(
                model_cls, 
                model_cfg, 
                mel_spec_type="vocos",
                vocab_file=vocab_path, 
                device=device
            )
            
            ckpt_path = hf_hub_download(repo_id=self.model_name, filename="model.safetensors")
            
            from safetensors.torch import load_file
            raw_state_dict = load_file(ckpt_path, device=device)
            
            cleaned_state_dict = {}
            for k, v in raw_state_dict.items():
                new_k = k.replace("ema_model.", "").replace("_orig_mod.", "").replace("vocoder.", "")
                cleaned_state_dict[new_k] = v
                
            self._ema_model.load_state_dict(cleaned_state_dict, strict=False)
            self._device = device
            logger.info(f"IndicTTS loaded on {device}")
            
        except ImportError as e:
            logger.warning(f"IndicTTS unavailable (missing dependency): {e}")
            raise RuntimeError(f"IndicTTS unavailable: {e}")
        except Exception as e:
            logger.warning(f"IndicTTS unavailable (load failed): {type(e).__name__} - {e}")
            raise RuntimeError(f"IndicTTS unavailable: {e}")
    
    async def synthesize(
        self,
        text: str,
        language: str,
        voice: str = "default",
        emotion: str = "neutral",
        speed: float = 1.0,
    ) -> SynthesisResult:
        await self._ensure_initialized()
        
        if not text or not text.strip():
            return SynthesisResult(
                audio=b"",
                format="wav",
                sample_rate=self.OUTPUT_SAMPLE_RATE,
                duration_seconds=0.0,
                language=language,
                voice=voice,
                provider="error"
            )
        
        text = self._normalize_text(text, language)
        
        if self._ema_model is not None and self._vocoder is not None:
            return await self._synthesize_local(text, language, voice, emotion, speed)
        
        raise RuntimeError("Local TTS model is not loaded and no fallbacks are allowed")
    
    async def _synthesize_local(
        self,
        text: str,
        language: str,
        voice: str,
        emotion: str,
        speed: float,
    ) -> SynthesisResult:
        import numpy as np
        from importlib.resources import files
        from f5_tts.infer.utils_infer import infer_process
        
        actual_voice = self._get_voice(language, voice)
        ref_file = str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav"))
        ref_text = "some call me nature, others call me mother nature."
        
        wav, sr, spect = infer_process(
            ref_audio=ref_file,
            ref_text=ref_text,
            gen_text=text,
            model_obj=self._ema_model,
            vocoder=self._vocoder,
            mel_spec_type="vocos",
            speed=speed,
            device=self._device
        )
        
        audio_array = wav.flatten()
        max_val = np.abs(audio_array).max()
        if max_val > 0:
            audio_array = audio_array / max_val
            
        audio_bytes = self._array_to_wav(audio_array, sr)
        duration = len(audio_array) / sr
        
        return SynthesisResult(
            audio=audio_bytes,
            format="wav",
            sample_rate=sr,
            duration_seconds=duration,
            language=language,
            voice=actual_voice,
            provider="indicf5"
        )
    
    def _normalize_text(self, text: str, language: str) -> str:
        text = " ".join(text.split())
        if text and text[-1] not in ".!?।":
            text += "।" if language in ["hi", "mr", "kn", "te", "ta"] else "."
        return text
    
    def _get_voice(self, language: str, voice: str) -> str:
        voices = self.LANGUAGE_VOICES.get(language, self.LANGUAGE_VOICES["en"])
        if voice in ["male", "male_default"]:
            return voices[0]
        elif voice in ["female", "female_default"]:
            return voices[1] if len(voices) > 1 else voices[0]
        elif voice == "default":
            return voices[0]
        return voice
    
    def _array_to_wav(self, audio_array, sample_rate: int) -> bytes:
        import struct
        import numpy as np
        
        audio_int16 = (audio_array * 32767).astype(np.int16)
        
        num_channels = 1
        sample_width = 2
        byte_rate = sample_rate * num_channels * sample_width
        block_align = num_channels * sample_width
        data_size = len(audio_int16) * sample_width
        
        header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF',
            36 + data_size,
            b'WAVE',
            b'fmt ',
            16,
            1,
            num_channels,
            sample_rate,
            byte_rate,
            block_align,
            16,
            b'data',
            data_size,
        )
        return header + audio_int16.tobytes()
    
    def get_supported_languages(self) -> list[str]:
        return list(self.LANGUAGE_VOICES.keys())
    
    def get_available_voices(self, language: str) -> list[str]:
        return self.LANGUAGE_VOICES.get(language, ["default"])
