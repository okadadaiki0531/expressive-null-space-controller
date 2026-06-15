# Expressive Null-Space Controller

Franka Panda（7DoF）を用いた **Expressive Null-Space Controller** の実装。

手先の3D位置追従タスク（3DoF）を維持しながら、余剰な4DoFのヌル空間を使い、
PAD（Pleasure-Arousal-Dominance）感情モデルに基づく表現的な動きを生成する。

## 冗長性

| 項目 | 値 |
|------|-----|
| ロボット | Franka Panda |
| 関節数 | **7 DoF** |
| タスク | 手先3D位置追従 |
| タスク自由度 | 3 DoF |
| **冗長DoF** | **4 DoF（条件：≥2 ✅）** |

## 制御則

```
q̇ = J⁺ ẋ_task + (I - J⁺J) q̇_null(PAD)
```

- `J⁺ ẋ_task`  : タスク（位置追従）を達成する最小ノルム解
- `(I - J⁺J)` : ヌル空間投影行列（タスクに影響しない）
- `q̇_null(PAD)`: PADから生成した感情的な関節速度

## PADマッピング

| 次元 | 意味 | +1 の動き | -1 の動き |
|------|------|-----------|-----------|
| P (Pleasure)  | 快/不快 | ホームへ開放的に伸びる | 縮こまる |
| A (Arousal)   | 覚醒度  | 素早い振動・興奮 | ゆっくり・抑制 |
| D (Dominance) | 支配感  | 手首・末端で表現 | 肩・根元で表現 |

## 環境構築

```bash
conda create -n robotics_env python=3.10 -y
conda activate robotics_env
conda install -c conda-forge pybullet -y
pip install numpy scipy matplotlib opencv-python
```

## 実行方法

```bash
# GUIで確認（PADを指定）
conda run -n robotics_env python main.py --pad 0 0 0
conda run -n robotics_env python main.py --pad 1 1 1
conda run -n robotics_env python main.py --pad -1 -1 -1

# 全5パターンをGUIで連続実行
conda run -n robotics_env python main.py --all

# 動画録画（ヘッドレス）
conda run -n robotics_env python record_video.py
```

## ファイル構成

```
.
├── main.py                          # メイン実行スクリプト
├── record_video.py                  # 全パターン録画・MP4出力
├── controller/
│   └── null_space_controller.py    # ヌル空間制御器（PADマッピング含む）
├── simulation/
│   └── sim.py                      # PyBulletシミュレーション環境
└── results/
    └── expressive_null_space_demo.mp4
```

## 課題要件チェック

- [x] ロボット：Franka Panda（7DoF）
- [x] タスク：手先3D位置追従（円形軌道）
- [x] 冗長DoF：4DoF（≥2）
- [x] PAD=[0,0,0] ベースライン
- [x] PAD=[1,1,1]
- [x] PAD=[-1,-1,-1]
- [x] PAD=[-1,1,-1]
- [x] PAD=[-1,1,1]
