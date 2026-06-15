"""
main.py — Expressive Null-Space Controller デモ
=================================================
Franka Panda (7DoF) が円形軌道（3DoFタスク）を描きながら、
異なるPAD値に応じた感情的な動きをヌル空間で表現するデモ。

実行方法:
    conda run -n robotics_env python main.py --pad 0 0 0
    conda run -n robotics_env python main.py --pad 1 1 1
    conda run -n robotics_env python main.py --gui   # GUIで確認

PADパターン:
    [0, 0, 0]   : ベースライン（感情なし）
    [1, 1, 1]   : 喜び・興奮・自信
    [-1,-1,-1]  : 悲しみ・沈静・従属
    [-1, 1,-1]  : 不安・興奮・従属
    [-1, 1, 1]  : 怒り・興奮・自信
"""

import numpy as np
import argparse
import sys
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(__file__))

from simulation.sim import PandaSimulation
from controller.null_space_controller import NullSpaceController


# ------------------------------------------------------------------
# タスク定義：円形軌道（手先3DoF）
# ------------------------------------------------------------------
def circle_trajectory(t, center, radius=0.15, freq=0.2):
    """
    手先の目標位置を円形軌道として生成。

    Parameters
    ----------
    t : float
        時間 [s]
    center : array (3,)
        円の中心位置
    radius : float
        円の半径 [m]
    freq : float
        周波数 [Hz]（1周にかかる時間 = 1/freq 秒）

    Returns
    -------
    target : array (3,)
        目標手先位置
    """
    angle = 2 * np.pi * freq * t
    x = center[0] + radius * np.cos(angle)
    y = center[1] + radius * np.sin(angle)
    z = center[2] + 0.05 * np.sin(2 * angle)  # 少し上下にも動く
    return np.array([x, y, z])


# ------------------------------------------------------------------
# PAD名の変換（表示用）
# ------------------------------------------------------------------
def pad_to_label(pad):
    labels = {
        (0, 0, 0):    "Baseline [P=0, A=0, D=0]",
        (1, 1, 1):    "Joyful   [P=+1, A=+1, D=+1]",
        (-1, -1, -1): "Sad      [P=-1, A=-1, D=-1]",
        (-1, 1, -1):  "Anxious  [P=-1, A=+1, D=-1]",
        (-1, 1, 1):   "Angry    [P=-1, A=+1, D=+1]",
    }
    key = tuple(int(v) for v in pad)
    return labels.get(key, f"PAD={pad}")


# ------------------------------------------------------------------
# メインシミュレーション実行
# ------------------------------------------------------------------
def run_simulation(pad, gui=True, duration=10.0, record=False,
                   output_path=None):
    """
    指定したPADでシミュレーションを実行。

    Parameters
    ----------
    pad : array (3,)
        [P, A, D]
    gui : bool
        GUIウィンドウを表示するか
    duration : float
        シミュレーション時間 [s]
    record : bool
        フレームを記録するか（動画用）
    output_path : str
        動画保存パス（recordがTrueの時）

    Returns
    -------
    results : dict
        シミュレーション結果（誤差履歴など）
    frames : list
        録画フレーム（recordがTrueの時）
    """
    label = pad_to_label(pad)
    print(f"\n{'='*60}")
    print(f"  PAD: {label}")
    print(f"  Duration: {duration}s")
    print(f"{'='*60}")

    # シミュレーション初期化
    sim = PandaSimulation(gui=gui, dt=1/240.0)

    # コントローラー初期化
    ctrl = NullSpaceController(
        n_joints=7,
        task_dim=3,
        k_task=5.0,     # タスクゲイン
        k_null=2.0,     # ヌル空間ゲイン
        damping=0.05    # 特異点回避ダンピング
    )

    # 手先のホーム位置を取得（軌道の中心に使用）
    q0 = sim.get_joint_positions()
    ee_home = sim.get_ee_position()
    traj_center = ee_home.copy()

    print(f"  EE home position: {ee_home.round(3)}")

    # -------------------------------------------------------
    # シミュレーションループ
    # -------------------------------------------------------
    dt = sim.dt
    steps = int(duration / dt)
    record_every = 4  # 4ステップに1フレーム録画（60fps → 15fps相当）

    results = {"time": [], "x_error": [], "q": []}
    frames = []

    # PADラベルの表示テキストID
    text_id = None

    for step in range(steps):
        t = step * dt

        # 現在の関節角度・手先位置を取得
        q = sim.get_joint_positions()
        x_ee = sim.get_ee_position()

        # 目標位置（円軌道）
        x_target = circle_trajectory(t, traj_center)

        # ヤコビアン計算
        J = sim.get_jacobian(q)

        # 制御量計算
        q_dot, info = ctrl.compute(
            J=J,
            x_current=x_ee,
            x_target=x_target,
            q=q,
            q_rest=PandaSimulation.HOME_JOINTS,
            pad=pad
        )

        # 関節速度を適用
        sim.apply_joint_velocities(q_dot, max_vel=1.5)

        # シミュレーション1ステップ進める
        sim.step(real_time=gui)

        # マーカー更新（GUI時）
        if gui:
            sim.update_markers(x_target, x_ee)
            if step % 20 == 0:
                sim.add_text(
                    f"{label} | Err={info['x_error_norm']:.4f}m",
                    position=[0, 0, 1.2]
                )

        # 結果を記録
        if step % 10 == 0:
            results["time"].append(t)
            results["x_error"].append(info["x_error_norm"])
            results["q"].append(q.copy())

            if step % 240 == 0:
                print(f"  t={t:.1f}s | EE_err={info['x_error_norm']:.4f}m"
                      f" | null_vel={info['q_dot_null_norm']:.4f}")

        # フレーム録画
        if record and step % record_every == 0:
            frame = sim.capture_frame(width=640, height=480)
            frames.append(frame)

    sim.close()
    results["time"] = np.array(results["time"])
    results["x_error"] = np.array(results["x_error"])

    print(f"\n  完了 | 平均誤差: {results['x_error'].mean():.4f}m"
          f" | 最大誤差: {results['x_error'].max():.4f}m")

    return results, frames


# ------------------------------------------------------------------
# エントリーポイント
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Expressive Null-Space Controller Demo"
    )
    parser.add_argument(
        "--pad", nargs=3, type=float, default=[0, 0, 0],
        metavar=("P", "A", "D"),
        help="PAD値 例: --pad 1 1 1"
    )
    parser.add_argument(
        "--gui", action="store_true", default=True,
        help="GUIウィンドウを表示（デフォルト: True）"
    )
    parser.add_argument(
        "--no-gui", action="store_false", dest="gui",
        help="ヘッドレスモード（録画時）"
    )
    parser.add_argument(
        "--duration", type=float, default=15.0,
        help="シミュレーション時間 [s]"
    )
    parser.add_argument(
        "--record", action="store_true",
        help="動画を録画する"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="全PADパターンを順番に実行"
    )
    args = parser.parse_args()

    if args.all:
        # 全5パターンを順番に（動画録画モード）
        all_pads = [
            [0,  0,  0],
            [1,  1,  1],
            [-1, -1, -1],
            [-1, 1, -1],
            [-1, 1,  1],
        ]
        for pad in all_pads:
            run_simulation(
                pad=np.array(pad),
                gui=args.gui,
                duration=args.duration,
                record=args.record
            )
    else:
        pad = np.array(args.pad)
        run_simulation(
            pad=pad,
            gui=args.gui,
            duration=args.duration,
            record=args.record
        )
