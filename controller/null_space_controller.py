"""
Expressive Null-Space Controller
=================================
Franka Panda (7DoF) の手先位置追従タスク (3DoF) に対して、
残り4DoFのヌル空間にPAD（Pleasure-Arousal-Dominance）感情を
マッピングして表現的な動きを生成するコントローラ。

制御則：
  q̇ = J⁺ ẋ_task + (I - J⁺J) q̇_null(PAD)

  - J⁺ ẋ_task  : タスク達成（位置追従）のための最小ノルム関節速度
  - (I - J⁺J)  : ヌル空間投影行列（タスクに影響しない方向）
  - q̇_null(PAD): PADから生成するヌル空間内の関節速度
"""

import numpy as np


class NullSpaceController:
    """
    Expressive Null-Space Controller for Franka Panda (7DoF).

    Parameters
    ----------
    n_joints : int
        ロボットの関節数（Pandaは7）
    task_dim : int
        タスクの次元数（位置追従なら3）
    k_task : float
        タスク空間のPゲイン（位置追従の速さ）
    k_null : float
        ヌル空間のゲイン（感情表現の強さ）
    damping : float
        ダンピング最小二乗法のλ（特異点回避用）
    """

    def __init__(self, n_joints=7, task_dim=3,
                 k_task=5.0, k_null=2.0, damping=0.01):
        self.n_joints = n_joints
        self.task_dim = task_dim
        self.k_task = k_task
        self.k_null = k_null
        self.damping = damping

        # 冗長DoF確認
        redundancy = n_joints - task_dim
        print(f"[Controller] DoF={n_joints}, Task={task_dim}DoF, "
              f"Redundancy={redundancy}DoF")
        assert redundancy >= 2, "冗長DoFが2未満です！"

    # ------------------------------------------------------------------
    # PAD → ヌル空間速度のマッピング
    # ------------------------------------------------------------------
    def pad_to_null_velocity(self, pad, q, q_rest):
        """
        PAD値をヌル空間内の関節速度に変換する。

        マッピング設計：
          P (Pleasure)  : ホームポーズへの引力の強さ（+: 開放的, -: 縮こまる）
          A (Arousal)   : 動きの振動周波数・振幅（+: 速く大きく, -: 遅く小さく）
          D (Dominance) : 各関節の重み付け（+: 末端関節優位, -: 根元関節優位）

        Parameters
        ----------
        pad : array-like, shape (3,)
            [P, A, D] それぞれ -1 〜 +1
        q : array (7,)
            現在の関節角度
        q_rest : array (7,)
            安静姿勢（ヌル空間引力の目標）

        Returns
        -------
        q_dot_null : array (7,)
            ヌル空間内の関節速度ベクトル
        """
        P, A, D = np.clip(pad, -1.0, 1.0)

        # --- P (Pleasure): ホームポーズへの引力 ---
        # P>0: ホームへ積極的に戻る（開放的・伸びやか）
        # P<0: ホームから離れたがる（縮こまる・防御的）
        q_error_to_home = q_rest - q
        pleasure_vel = P * q_error_to_home  # (7,)

        # --- A (Arousal): 振動的な動き ---
        # A>0: 関節が素早くブルブルする（興奮・活発）
        # A<0: 動きがとても遅くなる（眠い・抑制）
        # ここでは各関節に位相をずらした正弦波を乗せる
        phase_offsets = np.linspace(0, np.pi, self.n_joints)
        import time
        t = time.time()
        arousal_freq = 1.0 + abs(A) * 3.0   # 周波数: 1〜4 Hz
        arousal_amp = abs(A) * 0.3           # 振幅
        arousal_vel = A * arousal_amp * np.sin(
            2 * np.pi * arousal_freq * t + phase_offsets
        )  # (7,)

        # --- D (Dominance): 関節ウェイトの分布 ---
        # D>0: 末端関節（手首）を大きく動かす（自信・積極的なアクション）
        # D<0: 根元関節（肩）を大きく動かす（全身での表現・支配的）
        weights = np.zeros(self.n_joints)
        if D >= 0:
            # 末端寄り重み (joint 4〜6 が大きい)
            weights = np.array([0.05, 0.1, 0.15, 0.2, D*0.8+0.1,
                                 D*0.9+0.1, D*1.0+0.1])
        else:
            # 根元寄り重み (joint 0〜2 が大きい)
            w = abs(D)
            weights = np.array([w*1.0+0.1, w*0.9+0.1, w*0.8+0.1,
                                 0.2, 0.15, 0.1, 0.05])
        weights = weights / (np.sum(weights) + 1e-8)

        # PADの合成
        q_dot_null = (pleasure_vel + arousal_vel) * weights * self.k_null

        return q_dot_null

    # ------------------------------------------------------------------
    # ヤコビアン疑似逆行列（ダンピング最小二乗法）
    # ------------------------------------------------------------------
    def damped_pseudo_inverse(self, J):
        """
        ダンピング最小二乗法 (DLS) による疑似逆行列。
        特異点付近でも安定した計算が可能。

        J⁺ = Jᵀ(JJᵀ + λ²I)⁻¹

        Parameters
        ----------
        J : array (m, n)
            ヤコビアン行列

        Returns
        -------
        J_pinv : array (n, m)
            疑似逆行列
        """
        m, n = J.shape
        lam2 = self.damping ** 2
        J_pinv = J.T @ np.linalg.inv(J @ J.T + lam2 * np.eye(m))
        return J_pinv

    # ------------------------------------------------------------------
    # メイン制御計算
    # ------------------------------------------------------------------
    def compute(self, J, x_current, x_target, q, q_rest, pad):
        """
        ヌル空間制御の関節速度を計算する。

        制御則：
          q̇ = J⁺(ẋ_task) + (I - J⁺J) q̇_null

        Parameters
        ----------
        J : array (3, 7)
            現在姿勢でのヤコビアン（位置成分のみ）
        x_current : array (3,)
            現在の手先位置
        x_target : array (3,)
            目標手先位置
        q : array (7,)
            現在の関節角度
        q_rest : array (7,)
            安静姿勢（ヌル空間の基準）
        pad : array (3,)
            [P, A, D] 感情パラメータ

        Returns
        -------
        q_dot : array (7,)
            制御入力（関節速度）
        info : dict
            デバッグ用情報
        """
        # --- Step 1: タスク誤差 ---
        x_error = x_target - x_current  # (3,)

        # --- Step 2: 疑似逆行列 ---
        J_pinv = self.damped_pseudo_inverse(J)  # (7, 3)

        # --- Step 3: ヌル空間投影行列 ---
        I = np.eye(self.n_joints)
        N = I - J_pinv @ J  # (7, 7)  ← ヌル空間プロジェクタ

        # --- Step 4: タスク空間の速度（位置追従PD制御）---
        x_dot_task = self.k_task * x_error  # (3,)

        # --- Step 5: PAD由来のヌル空間速度 ---
        q_dot_null = self.pad_to_null_velocity(pad, q, q_rest)  # (7,)

        # --- Step 6: 合成 ---
        q_dot_task = J_pinv @ x_dot_task        # タスク成分
        q_dot_null_proj = N @ q_dot_null         # ヌル空間成分（タスクに影響しない）
        q_dot = q_dot_task + q_dot_null_proj     # 最終関節速度

        info = {
            "x_error": x_error,
            "x_error_norm": np.linalg.norm(x_error),
            "q_dot_task_norm": np.linalg.norm(q_dot_task),
            "q_dot_null_norm": np.linalg.norm(q_dot_null_proj),
        }

        return q_dot, info
