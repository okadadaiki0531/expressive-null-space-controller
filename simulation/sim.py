"""
PyBullet Simulation for Franka Panda
=====================================
Franka PandaロボットをPyBulletで動かすシミュレーター。
ヤコビアンの取得・関節速度の適用・可視化を担当。
"""

import pybullet as p
import pybullet_data
import numpy as np
import time


class PandaSimulation:
    """
    Franka Pandaのシミュレーション環境。

    Parameters
    ----------
    gui : bool
        True → GUIウィンドウ表示, False → ヘッドレス（動画録画用）
    dt : float
        シミュレーションのタイムステップ [s]
    """

    # Pandaの関節インデックス（PyBulletのURDF構造に基づく）
    # joint 0-6: アーム7関節
    # joint 7-8: フィンガー（今回は固定）
    JOINT_INDICES = [0, 1, 2, 3, 4, 5, 6]
    FINGER_INDICES = [9, 10]
    END_EFFECTOR_LINK = 11  # panda_hand リンク

    # 安静姿勢（ホームポジション）
    HOME_JOINTS = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4,
                             0.0, np.pi/2, np.pi/4])

    # 関節の可動範囲 [rad]
    JOINT_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718,
                              -2.8973, -0.0175, -2.8973])
    JOINT_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698,
                              2.8973, 3.7525, 2.8973])

    def __init__(self, gui=True, dt=1/240.0):
        self.dt = dt
        self.gui = gui

        # PyBullet接続
        mode = p.GUI if gui else p.DIRECT
        self.physics_client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(dt)

        # 地面を読み込む
        p.loadURDF("plane.urdf")

        # Panda URDFを読み込む
        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[0, 0, 0],
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION
        )

        # ホームポジションに初期化
        self.reset_to_home()

        if gui:
            self._setup_camera()
            self._add_coordinate_axes()
            # 目標位置マーカー（赤い球）
            self.target_marker = self._create_sphere(
                radius=0.03, color=[1, 0, 0, 0.8]
            )
            # 手先位置マーカー（青い球）
            self.ee_marker = self._create_sphere(
                radius=0.02, color=[0, 0, 1, 0.8]
            )

        print(f"[Sim] Panda loaded. Links: {p.getNumJoints(self.robot_id)}")

    # ------------------------------------------------------------------
    # 初期化・リセット
    # ------------------------------------------------------------------
    def reset_to_home(self):
        """全関節をホームポジションにリセット"""
        for i, ji in enumerate(self.JOINT_INDICES):
            p.resetJointState(self.robot_id, ji, self.HOME_JOINTS[i])
        # フィンガーを開く
        for fi in self.FINGER_INDICES:
            p.resetJointState(self.robot_id, fi, 0.04)

    # ------------------------------------------------------------------
    # 関節状態の取得
    # ------------------------------------------------------------------
    def get_joint_positions(self):
        """現在の関節角度を取得 shape: (7,)"""
        states = p.getJointStates(self.robot_id, self.JOINT_INDICES)
        return np.array([s[0] for s in states])

    def get_joint_velocities(self):
        """現在の関節速度を取得 shape: (7,)"""
        states = p.getJointStates(self.robot_id, self.JOINT_INDICES)
        return np.array([s[1] for s in states])

    # ------------------------------------------------------------------
    # 手先（エンドエフェクタ）の情報
    # ------------------------------------------------------------------
    def get_ee_position(self):
        """手先位置を取得 shape: (3,)"""
        state = p.getLinkState(self.robot_id, self.END_EFFECTOR_LINK)
        return np.array(state[0])

    def get_ee_orientation(self):
        """手先姿勢（クォータニオン）を取得 shape: (4,)"""
        state = p.getLinkState(self.robot_id, self.END_EFFECTOR_LINK)
        return np.array(state[1])

    # ------------------------------------------------------------------
    # ヤコビアン計算（PyBullet組み込み関数を使用）
    # ------------------------------------------------------------------
    def get_jacobian(self, q):
        """
        位置ヤコビアン J_pos (3×7) を計算する。

        PyBulletのcalculateJacobianは全ヤコビアン(6×7)を返すが、
        今回は位置タスク(3DoF)のみ使用するので上3行を取り出す。

        Parameters
        ----------
        q : array (7,)
            現在の関節角度

        Returns
        -------
        J_pos : array (3, 7)
            位置ヤコビアン
        """
        q_list = list(q) + [0.04, 0.04]  # フィンガー含む9関節
        zero_vel = [0.0] * 9
        zero_acc = [0.0] * 9

        # EEリンクのローカル位置（重心）
        link_state = p.getLinkState(self.robot_id, self.END_EFFECTOR_LINK,
                                     computeForwardKinematics=True)
        local_pos = [0.0, 0.0, 0.0]

        jac_t, jac_r = p.calculateJacobian(
            self.robot_id,
            self.END_EFFECTOR_LINK,
            local_pos,
            q_list,
            zero_vel,
            zero_acc
        )

        # アーム7関節分だけ取り出す（フィンガー列を除く）
        J_full = np.array(jac_t)   # (3, 9)
        J_pos = J_full[:, :7]       # (3, 7)

        return J_pos

    # ------------------------------------------------------------------
    # 関節速度の適用
    # ------------------------------------------------------------------
    def apply_joint_velocities(self, q_dot, max_vel=1.0):
        """
        計算した関節速度をロボットに適用する。
        速度飽和処理つき。

        Parameters
        ----------
        q_dot : array (7,)
            目標関節速度 [rad/s]
        max_vel : float
            最大関節速度 [rad/s]
        """
        # 速度飽和（安全のため）
        norm = np.linalg.norm(q_dot)
        if norm > max_vel:
            q_dot = q_dot * max_vel / norm

        # 速度制御モードで適用
        p.setJointMotorControlArray(
            self.robot_id,
            self.JOINT_INDICES,
            p.VELOCITY_CONTROL,
            targetVelocities=q_dot.tolist(),
            forces=[87, 87, 87, 87, 12, 12, 12]  # Pandaの最大トルク
        )

    # ------------------------------------------------------------------
    # シミュレーションのステップ実行
    # ------------------------------------------------------------------
    def step(self, real_time=True):
        """1ステップ進める"""
        p.stepSimulation()
        if real_time and self.gui:
            time.sleep(self.dt)

    # ------------------------------------------------------------------
    # 可視化補助
    # ------------------------------------------------------------------
    def _setup_camera(self):
        """カメラ視点の設定"""
        p.resetDebugVisualizerCamera(
            cameraDistance=1.2,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0.4, 0, 0.4]
        )

    def _add_coordinate_axes(self):
        """原点に座標軸を表示"""
        length = 0.1
        p.addUserDebugLine([0,0,0], [length,0,0], [1,0,0], 2)
        p.addUserDebugLine([0,0,0], [0,length,0], [0,1,0], 2)
        p.addUserDebugLine([0,0,0], [0,0,length], [0,0,1], 2)

    def _create_sphere(self, radius=0.03, color=[1,0,0,0.8]):
        """マーカー球を作成"""
        vis_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius,
                                      rgbaColor=color)
        body_id = p.createMultiBody(0, -1, vis_id, [0, 0, 0])
        return body_id

    def update_markers(self, target_pos, ee_pos):
        """目標位置と手先位置のマーカーを更新"""
        p.resetBasePositionAndOrientation(
            self.target_marker, target_pos, [0,0,0,1])
        p.resetBasePositionAndOrientation(
            self.ee_marker, ee_pos, [0,0,0,1])

    def add_text(self, text, position, color=[1,1,1]):
        """デバッグテキストを画面に表示"""
        return p.addUserDebugText(text, position, color,
                                   textSize=1.2, lifeTime=0.1)

    # ------------------------------------------------------------------
    # 後処理
    # ------------------------------------------------------------------
    def close(self):
        """シミュレーションを終了"""
        p.disconnect(self.physics_client)

    def capture_frame(self, width=640, height=480):
        """
        現在のフレームを画像として取得（動画録画用）

        Returns
        -------
        img : array (H, W, 3) uint8
        """
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.4, 0, 0.4],
            distance=1.2,
            yaw=45,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=width/height,
            nearVal=0.1, farVal=100.0
        )
        _, _, rgba, _, _ = p.getCameraImage(
            width, height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER
        )
        img = np.array(rgba, dtype=np.uint8).reshape(height, width, 4)
        return img[:, :, :3]  # RGBのみ
