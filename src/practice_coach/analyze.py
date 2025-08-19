"""
Core analysis utilities for Practice Coach.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


@dataclass
class AnalysisResult:
    tempo_bpm: float
    stability_pct: float
    onsets_sec: np.ndarray


def load_audio(path: str | pathlib.Path, sr: int = 22050) -> Tuple[np.ndarray, int]:
    y, srr = librosa.load(path, sr=sr, mono=True)
    return y, srr


def detect_onsets(y: np.ndarray, sr: int) -> np.ndarray:
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onsets_sec = librosa.frames_to_time(onsets_frames, sr=sr)
    return onsets_sec


def estimate_tempo(onsets_sec: np.ndarray) -> tuple[float, float]:
    if len(onsets_sec) < 3:
        return 0.0, 0.0
    ioi = np.diff(onsets_sec)  # inter-onset intervals (sec)
    tempo_inst = 60.0 / np.maximum(ioi, 1e-6)  # instantaneous tempo (BPM)
    tempo_bpm = float(np.median(tempo_inst))
    stability = float(np.std(ioi) / np.maximum(np.mean(ioi), 1e-6) * 100.0)  # %
    return tempo_bpm, stability


def plot_waveform_onsets(y: np.ndarray, sr: int, onsets_sec: np.ndarray, out_path: str | pathlib.Path) -> None:
    plt.figure(figsize=(12, 3.2))
    librosa.display.waveshow(y, sr=sr)
    for t in onsets_sec:
        plt.axvline(t, alpha=0.3, linestyle="--")
    plt.xlabel("Time (s)")
    plt.title("Waveform with Detected Onsets")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_tempo_curve(onsets_sec: np.ndarray, out_path: str | pathlib.Path) -> None:
    plt.figure(figsize=(12, 3.2))
    if len(onsets_sec) >= 3:
        ioi = np.diff(onsets_sec)
        tempo_inst = 60.0 / np.maximum(ioi, 1e-6)
        t_mid = (onsets_sec[:-1] + onsets_sec[1:]) / 2.0
        plt.plot(t_mid, tempo_inst, marker="o", linewidth=1.5)
        plt.ylabel("Tempo (BPM)")
        plt.xlabel("Time (s)")
        plt.title("Tempo Over Time")
    else:
        plt.text(0.5, 0.5, "Not enough onsets to estimate tempo curve", ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_onsets_csv(onsets_sec: np.ndarray, out_path: str | pathlib.Path) -> None:
    arr = np.asarray(onsets_sec).reshape(-1, 1)
    header = "onset_seconds"
    np.savetxt(out_path, arr, delimiter=",", header=header, comments="")


def analyze_file(input_path: str | pathlib.Path, out_dir: str | pathlib.Path) -> AnalysisResult:
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y, sr = load_audio(input_path)
    onsets_sec = detect_onsets(y, sr)
    tempo_bpm, stability_pct = estimate_tempo(onsets_sec)

    # plots & csv
    plot_waveform_onsets(y, sr, onsets_sec, out_dir / "waveform_onsets.png")
    plot_tempo_curve(onsets_sec, out_dir / "tempo_curve.png")
    save_onsets_csv(onsets_sec, out_dir / "onsets.csv")

    # metrics
    metrics = {
        "tempo_bpm": tempo_bpm,
        "stability_pct": stability_pct,
        "n_onsets": int(len(onsets_sec)),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return AnalysisResult(tempo_bpm=tempo_bpm, stability_pct=stability_pct, onsets_sec=onsets_sec)
