import numpy as np
import soundfile as sf
import subprocess
import pyloudnorm as pyln
import librosa
from scipy.optimize import differential_evolution
import tkinter as tk
from tkinter import filedialog

TARGET_SR = 44100
FRAGMENT_SECONDS = 30
TARGET_LUFS = -23.0
meter = pyln.Meter(TARGET_SR)


def select_file_dialog(title="Select file"):
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title=title)
    root.destroy()
    return path


def run_ffmpeg(input_file, fl, fr, fc, sl, sr, output_file):
    pan_filter = (
        f"pan=stereo|"
        f"c0={fl:.3f}*FL+{fc:.3f}*FC+{sl:.3f}*SL|"
        f"c1={fr:.3f}*FR+{fc:.3f}*FC+{sr:.3f}*SR"
    )
    cmd = [
        'ffmpeg', '-y', '-i', input_file,
        '-filter_complex', pan_filter,
        '-c:a', 'pcm_s16le',
        output_file
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("FFmpeg error:", result.stderr.decode())
        return False
    return True


def loudness_normalize(audio, target_lufs=TARGET_LUFS):
    loudness = meter.integrated_loudness(audio)
    return pyln.normalize.loudness(audio, loudness, target_lufs)


def stereo_mfcc_distance(a, b, sr=TARGET_SR):
    # Match channels and duration
    min_len = min(len(a), len(b))
    a, b = a[:min_len], b[:min_len]

    if a.ndim == 1: a = np.stack([a, a], axis=-1)
    if b.ndim == 1: b = np.stack([b, b], axis=-1)

    # Normalize loudness
    a = loudness_normalize(a)
    b = loudness_normalize(b)

    dist_l = np.linalg.norm(librosa.feature.mfcc(y=a[:, 0], sr=sr) - librosa.feature.mfcc(y=b[:, 0], sr=sr))
    dist_r = np.linalg.norm(librosa.feature.mfcc(y=a[:, 1], sr=sr) - librosa.feature.mfcc(y=b[:, 1], sr=sr))

    stereo_diff_penalty = np.abs(dist_l - dist_r) * 0.5  # encourage stereo balance
    return (dist_l + dist_r) / 2 + stereo_diff_penalty


def objective(params, input_5_1, ref_path, temp_out):
    fl, fr, fc, sl, sr = np.clip(params, 0, 1)

    if not run_ffmpeg(input_5_1, fl, fr, fc, sl, sr, temp_out):
        return 1e6

    try:
        out_audio, out_sr = sf.read(temp_out)
        ref_audio, ref_sr = sf.read(ref_path)
    except Exception as e:
        print("Read error:", e)
        return 1e6

    # Resample
    if out_sr != TARGET_SR:
        out_audio = librosa.resample(out_audio.T, orig_sr=out_sr, target_sr=TARGET_SR).T
    if ref_sr != TARGET_SR:
        ref_audio = librosa.resample(ref_audio.T, orig_sr=ref_sr, target_sr=TARGET_SR).T

    # Crop fragment
    max_len = FRAGMENT_SECONDS * TARGET_SR
    out_audio = out_audio[:max_len]
    ref_audio = ref_audio[:max_len]

    # Distance
    dist = stereo_mfcc_distance(out_audio, ref_audio)
    print(f"fl={fl:.3f}, fr={fr:.3f}, fc={fc:.3f}, sl={sl:.3f}, sr={sr:.3f} -> MFCC dist={dist:.4f}")
    return dist


def main():
    print("Select your 5.1 audio file:")
    input_5_1 = select_file_dialog("Select 5.1 input file")
    if not input_5_1:
        print("No file selected.")
        return

    print("Select your stereo reference file:")
    reference = select_file_dialog("Select stereo reference file")
    if not reference:
        print("No reference selected.")
        return

    temp_out = "temp_output.wav"
    bounds = [(0, 1)] * 5

    result = differential_evolution(
        lambda p: objective(p, input_5_1, reference, temp_out),
        bounds, maxiter=100, polish=True, tol=1e-6
    )

    if result.success:
        fl, fr, fc, sl, sr = result.x
        print("\n🎉 Optimization Complete!")
        print(f"Best Weights:\n  FL: {fl:.3f}\n  FR: {fr:.3f}\n  FC: {fc:.3f}\n  SL: {sl:.3f}\n  SR: {sr:.3f}")
        print("\n🔧 Recommended FFmpeg command:")
        print(
            f'ffmpeg -i "{input_5_1}" -filter_complex "pan=stereo|c0={fl:.3f}*FL+{fc:.3f}*FC+{sl:.3f}*SL|c1={fr:.3f}*FR+{fc:.3f}*FC+{sr:.3f}*SR" output_best_stereo.wav')
    else:
        print("❌ Optimization failed.")


if __name__ == "__main__":
    main()
