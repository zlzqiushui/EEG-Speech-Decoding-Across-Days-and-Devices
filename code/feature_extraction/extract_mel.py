#!/usr/bin/env python3
"""
提取 Mel 频谱特征 (Mel Spectrogram Extraction)

从音频文件中提取 mel 频谱特征，输出与现有 mel embedding 格式兼容的 .npy 文件。

现有 mel embedding:
    - 形状: (time_frames, 80)        # 80 个 mel 频带，时间帧数取决于音频长度
    - 帧率: 50 Hz                    # sr / hop_length = 48000 / 960 = 50
    - dtype: float64
    - 值范围: dB 尺度 (power_to_db)

用法:
    python extract_mel.py \
        --audio_dir /path/to/audio \
        --output_dir /path/to/output \
        --sr 48000 \
        --n_fft 2048 \
        --hop_length 960 \
        --n_mels 80

依赖:
    pip install librosa soundfile numpy tqdm
"""

import argparse
import os
import sys
import numpy as np
import librosa


def extract_mel(
    audio_path: str,
    sr: int = 48000,
    n_fft: int = 2048,
    hop_length: int = 960,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: float = None,
    htk: bool = False,
    norm: str = None,
    power: float = 2.0,
    top_db: float = None,
    output_format: str = "db",
) -> np.ndarray:
    """
    提取单个音频文件的 mel 频谱。

    Parameters
    ----------
    audio_path : str
        音频文件路径 (支持 mp3, wav, flac 等 librosa 可读取的格式)。
    sr : int
        目标采样率 (Hz)。默认 48000。
    n_fft : int
        FFT 窗口大小。默认 2048。
    hop_length : int
        帧移 (samples)。帧率 = sr / hop_length。
        默认 960 → 48000/960 = 50 Hz 帧率。
    n_mels : int
        Mel 频带数量。默认 80。
    fmin : float
        最低频率 (Hz)。默认 0.0。
    fmax : float
        最高频率 (Hz)。None 表示 sr/2。
    power : float
        功率谱指数。2.0 = 功率谱, 1.0 = 幅度谱。
    top_db : float or None
        dB 动态范围上限。None 表示不截断。
    output_format : str
        输出格式:
        - "power"  : 原始功率谱 (power mel spectrogram)
        - "db"     : dB 尺度 (librosa.power_to_db)
        - "log"    : 自然对数 (log(1 + mel))

    Returns
    -------
    mel_features : np.ndarray, shape=(time_frames, n_mels), dtype=float64
        转置后的 mel 频谱特征 (time_frames 作为第一维)。
    """
    # 加载音频 (单声道)
    y, sr_actual = librosa.load(audio_path, sr=sr, mono=True)
    if sr_actual != sr:
        print(f"  Warning: requested sr={sr}, got sr={sr_actual}")

    # 计算 mel 频谱
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        htk=htk,
        norm=norm,
        power=power,
        window="hann",
    )

    # 转换成目标格式
    if output_format == "power":
        mel_features = mel_spec
    elif output_format == "db":
        mel_features = librosa.power_to_db(mel_spec, top_db=top_db)
    elif output_format == "log":
        mel_features = np.log1p(mel_spec)  # log(1 + x)
    else:
        raise ValueError(f"Unknown output_format: {output_format}")

    # 转置: (n_mels, time) → (time, n_mels)
    mel_features = mel_features.T.astype(np.float64)

    return mel_features


def main():
    parser = argparse.ArgumentParser(
        description="从音频文件中提取 Mel 频谱特征",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- 路径参数 ---
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help="输入音频目录路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出 .npy 文件目录",
    )
    parser.add_argument(
        "--audio_ext",
        type=str,
        default=".mp3",
        help="音频文件扩展名 (默认: .mp3)",
    )

    # --- Mel 频谱参数 ---
    parser.add_argument(
        "--sr",
        type=int,
        default=48000,
        help="目标采样率 Hz (默认: 48000)",
    )
    parser.add_argument(
        "--n_fft",
        type=int,
        default=2048,
        help="FFT 窗口大小 (默认: 2048)",
    )
    parser.add_argument(
        "--hop_length",
        type=int,
        default=960,
        help="帧移 samples, 帧率 = sr/hop_length (默认: 960 → 50 Hz)",
    )
    parser.add_argument(
        "--n_mels",
        type=int,
        default=80,
        help="Mel 频带数量 (默认: 80)",
    )
    parser.add_argument(
        "--fmin",
        type=float,
        default=0.0,
        help="最低频率 Hz (默认: 0.0)",
    )
    parser.add_argument(
        "--fmax",
        type=str,
        default="8000",
        help="最高频率 Hz (默认: 8000, 使用 'none' 表示 sr/2)",
    )
    parser.add_argument(
        "--htk",
        action="store_true",
        default=False,
        help="使用 HTK 风格的 mel 尺度 (默认: False)",
    )
    parser.add_argument(
        "--norm",
        type=str,
        default=None,
        choices=["slaney", None],
        help="Mel 滤波器组归一化方式: slaney 或 None (默认: None)",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=2.0,
        help="功率谱指数: 2.0=功率谱, 1.0=幅度谱 (默认: 2.0)",
    )
    parser.add_argument(
        "--top_db",
        type=float,
        default=None,
        help="dB 动态范围截断。None 表示不截断，80 表示截断低于峰值-80dB 的值 (默认: None)",
    )

    # --- 输出参数 ---
    parser.add_argument(
        "--output_format",
        type=str,
        default="db",
        choices=["power", "db", "log"],
        help="输出格式: power (功率谱), db (dB尺度), log (自然对数) (默认: db)",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_mel",
        help="输出文件名后缀 (默认: _mel)",
    )

    # --- 其他 ---
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="文件名匹配模式 (glob), 只处理匹配的文件。None 表示处理所有 audio_ext 文件",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=False,
        help="跳过已存在的输出文件",
    )

    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 收集音频文件
    audio_files = []
    for fname in sorted(os.listdir(args.audio_dir)):
        if fname.endswith(args.audio_ext):
            audio_files.append(fname)

    if not audio_files:
        print(f"错误: 在 {args.audio_dir} 中未找到 {args.audio_ext} 文件")
        sys.exit(1)

    # 解析 fmax
    fmax = None if args.fmax.lower() == "none" else float(args.fmax)

    print(f"找到 {len(audio_files)} 个音频文件")
    print(f"Mel 参数: sr={args.sr}, n_fft={args.n_fft}, hop_length={args.hop_length}, "
          f"n_mels={args.n_mels}, fmin={args.fmin}, fmax={fmax}, htk={args.htk}, norm={args.norm}")
    print(f"帧率: {args.sr / args.hop_length:.1f} Hz")
    print(f"输出格式: {args.output_format}")
    print(f"输出目录: {args.output_dir}")
    print("-" * 60)

    # 逐文件处理
    for fname in audio_files:
        # 输入路径
        audio_path = os.path.join(args.audio_dir, fname)

        # 输出文件名: 去掉音频扩展名 + suffix + .npy
        base_name = os.path.splitext(fname)[0]
        output_path = os.path.join(args.output_dir, f"{base_name}{args.suffix}.npy")

        if args.skip_existing and os.path.exists(output_path):
            print(f"跳过 (已存在): {fname} → {os.path.basename(output_path)}")
            continue

        try:
            mel = extract_mel(
                audio_path=audio_path,
                sr=args.sr,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                n_mels=args.n_mels,
                fmin=args.fmin,
                fmax=fmax,
                htk=args.htk,
                norm=args.norm,
                power=args.power,
                top_db=args.top_db,
                output_format=args.output_format,
            )
            np.save(output_path, mel)
            duration = len(mel) / (args.sr / args.hop_length)
            print(f"完成: {fname} → {os.path.basename(output_path)}  "
                  f"shape={mel.shape}, duration={duration:.1f}s")
        except Exception as e:
            print(f"失败: {fname} — {e}")

    print("-" * 60)
    print(f"处理完成。输出文件数: {len(os.listdir(args.output_dir))}")


if __name__ == "__main__":
    main()
