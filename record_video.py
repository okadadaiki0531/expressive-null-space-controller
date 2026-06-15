"""
record_video.py — 全PADパターンを録画して1本の動画に結合
=========================================================
5パターンを順番に実行・録画し、最終的に1本のMP4にまとめる。
各パターンの前にタイトルカードを表示する。

実行方法:
    conda run -n robotics_env python record_video.py

出力:
    results/expressive_null_space_demo.mp4
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from main import run_simulation, pad_to_label


def make_title_card(text, width=640, height=480, n_frames=45):
    """
    タイトルカード画像（黒背景に白テキスト）を生成。
    OpenCVを使用。
    """
    try:
        import cv2
        frames = []
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # テキストを中央に配置
        font = cv2.FONT_HERSHEY_SIMPLEX
        lines = text.split('\n')
        y_start = height // 2 - len(lines) * 25
        for i, line in enumerate(lines):
            size = cv2.getTextSize(line, font, 0.8, 2)[0]
            x = (width - size[0]) // 2
            y = y_start + i * 50
            cv2.putText(img, line, (x, y), font, 0.8, (255, 255, 255), 2)

        for _ in range(n_frames):
            frames.append(img.copy())
        return frames
    except ImportError:
        # OpenCVなければ黒フレームのみ
        return [np.zeros((height, width, 3), dtype=np.uint8)] * n_frames


def save_video(frames, output_path, fps=15, width=640, height=480):
    """
    フレームリストをMP4動画として保存。
    """
    try:
        import cv2
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for frame in frames:
            # RGB → BGR（OpenCVはBGR形式）
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\n✅ 動画を保存しました: {output_path}")
        print(f"   サイズ: {size_mb:.1f} MB | フレーム数: {len(frames)}")
    except ImportError:
        print("⚠️  OpenCVがありません。pip install opencv-python を実行してください。")
        print("   代替: numpy配列でフレームを保存します...")
        np.save(output_path.replace('.mp4', '_frames.npy'),
                np.array(frames))


if __name__ == "__main__":
    # 全PADパターン
    pad_configs = [
        {"pad": [0,  0,  0],  "duration": 12.0, "label": "Baseline"},
        {"pad": [1,  1,  1],  "duration": 12.0, "label": "Joyful"},
        {"pad": [-1, -1, -1], "duration": 12.0, "label": "Sad"},
        {"pad": [-1, 1, -1],  "duration": 12.0, "label": "Anxious"},
        {"pad": [-1, 1,  1],  "duration": 12.0, "label": "Angry"},
    ]

    all_frames = []

    for config in pad_configs:
        pad = np.array(config["pad"])
        label = pad_to_label(pad)

        print(f"\n🎬 録画中: {label}")

        # タイトルカード（3秒）
        title_text = f"{config['label']}\nPAD = {config['pad']}"
        title_frames = make_title_card(title_text, n_frames=45)
        all_frames.extend(title_frames)

        # シミュレーション録画
        _, frames = run_simulation(
            pad=pad,
            gui=False,          # ヘッドレスモード（高速）
            duration=config["duration"],
            record=True
        )
        all_frames.extend(frames)

    # 動画保存
    output_path = os.path.join(
        os.path.dirname(__file__), "results",
        "expressive_null_space_demo.mp4"
    )
    save_video(all_frames, output_path, fps=15)
