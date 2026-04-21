#!/usr/bin/env python3


import io
import json
import numpy as np
import wave

SAMPLE_RATE    = 16000
N_MELS         = 80
N_FFT          = 400       # 25ms window at 16kHz
HOP_LENGTH     = 160       # 10ms hop at 16kHz
CHUNK_SEC      = 30.0      # max audio length per chunk
MAX_FRAMES     = int(CHUNK_SEC * SAMPLE_RATE / HOP_LENGTH)  

def load_audio_from_bytes(audio_bytes: bytes) -> np.ndarray:
    
    buf = io.BytesIO(audio_bytes)
    with wave.open(buf, 'rb') as wf:
        n_channels  = wf.getnchannels()
        sampwidth   = wf.getsampwidth()
        framerate   = wf.getframerate()
        raw_frames  = wf.readframes(wf.getnframes())

    
    if sampwidth == 2:
        audio = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw_frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    
    if n_channels == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)

    if framerate != SAMPLE_RATE:
        new_len = int(len(audio) * SAMPLE_RATE / framerate)
        audio = np.interp(
            np.linspace(0, len(audio), new_len),
            np.arange(len(audio)),
            audio
        ).astype(np.float32)

    return audio




def _stft_magnitude(audio: np.ndarray) -> np.ndarray:
    
    window = np.hanning(N_FFT).astype(np.float32)
    n_frames = 1 + (len(audio) - N_FFT) // HOP_LENGTH

    frames = np.stack([
        audio[i * HOP_LENGTH : i * HOP_LENGTH + N_FFT] * window
        for i in range(n_frames)
    ])  # (n_frames, N_FFT)

    spectrum = np.fft.rfft(frames, n=N_FFT)
    return np.abs(spectrum).astype(np.float32)  # (n_frames, N_FFT//2 + 1)


def _mel_filterbank(n_mels: int, n_fft: int, sr: int) -> np.ndarray:
    
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)
    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    low_mel  = hz_to_mel(0.0)
    high_mel = hz_to_mel(sr / 2.0)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points  = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filterbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_m_minus = bin_points[m - 1]
        f_m       = bin_points[m]
        f_m_plus  = bin_points[m + 1]
        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus:
                filterbank[m-1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m:
                filterbank[m-1, k] = (f_m_plus - k) / (f_m_plus - f_m)
    return filterbank



_MEL_FILTERBANK = None

def get_mel_filterbank() -> np.ndarray:
    global _MEL_FILTERBANK
    if _MEL_FILTERBANK is None:
        _MEL_FILTERBANK = _mel_filterbank(N_MELS, N_FFT, SAMPLE_RATE)
    return _MEL_FILTERBANK


def compute_log_mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    
    max_samples = int(CHUNK_SEC * SAMPLE_RATE)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    else:
        audio = np.pad(audio, (0, max_samples - len(audio)))

    magnitude = _stft_magnitude(audio)             # (T, F)
    filterbank = get_mel_filterbank()              # (N_MELS, F)
    mel = filterbank @ magnitude.T                 # (N_MELS, T)

    
    mel = np.log10(np.maximum(mel, 1e-10))
    mel = np.maximum(mel, mel.max() - 8.0)
    mel = (mel + 4.0) / 4.0

    
    if mel.shape[1] < MAX_FRAMES:
        mel = np.pad(mel, ((0, 0), (0, MAX_FRAMES - mel.shape[1])))
    else:
        mel = mel[:, :MAX_FRAMES]

    return mel.astype(np.float32)  # (N_MELS, MAX_FRAMES)


def normalize_features(features: np.ndarray) -> np.ndarray:
    
    mean = features.mean(axis=1, keepdims=True)
    std  = features.std(axis=1, keepdims=True) + 1e-8
    return ((features - mean) / std).astype(np.float32)



def extract_features(audio_bytes: bytes, normalize: bool = True) -> dict:
    
    audio    = load_audio_from_bytes(audio_bytes)
    duration = len(audio) / SAMPLE_RATE
    features = compute_log_mel_spectrogram(audio)

    if normalize:
        features = normalize_features(features)

    return {
        "features"    : features,
        "duration_sec": round(duration, 3),
        "sample_rate" : SAMPLE_RATE,
        "shape"       : features.shape,
        "n_mels"      : N_MELS,
        "max_frames"  : MAX_FRAMES,
    }


def features_to_dict(result: dict) -> dict:
    return {
        "features_b64" : __import__("base64").b64encode(
                            result["features"].tobytes()).decode(),
        "dtype"        : str(result["features"].dtype),
        "shape"        : list(result["shape"]),
        "duration_sec" : result["duration_sec"],
        "sample_rate"  : result["sample_rate"],
        "n_mels"       : result["n_mels"],
        "max_frames"   : result["max_frames"],
    }
