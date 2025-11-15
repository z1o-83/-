import numpy as np
from scipy.interpolate import splprep, BSpline, PPoly, splev
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def set_aspect_equal_3d(ax):
    """Matplotlib 3D 플롯의 축 스케일을 동일하게 설정합니다."""
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

def generate_track_from_images():
    """
    image_312b7d.png (원형+노이즈)와 image_312b3f.png (C2 주기성) 이론 기반
    3D 경로 및 가변 폭 노이즈 스플라인 생성, 구간별 함수 출력
    """
    
    # 1. 제어점 생성
    
    # --- 파라미터 설정 ---
    N_POINTS = 20      # 제어점 개수 (n)
    RADIUS = 800       # 기준 원의 반지름 (R)
    SIGMA_XY = 250     # x, y축 복잡도 (sigma)
    SIGMA_Z = 0.5       # z축 경사 복잡도 (sigma_z)
    # ----------------------
    
    control_points = [] # 3D 경로 (x,y,z)
    noise_control = []  # 가변폭 Noise(t) (0~1)

    print(f"--- [제어점 생성] (N={N_POINTS}, R={RADIUS}, sigma_xy={SIGMA_XY}, sigma_z={SIGMA_Z}) ---")

    for i in range(N_POINTS):
        angle = (2 * np.pi * i) / N_POINTS
        
        # 1-1. 기준 원 위치 P_ref,i
        # 3D로 확장 (z=0)
        p_ref_x = RADIUS * np.cos(angle)
        p_ref_y = RADIUS * np.sin(angle)
        p_ref_z = 0 # 기준 높이
        
        # 1-2. 무작위 편차 (rand(-sigma, sigma))
        rand_x = random.uniform(-SIGMA_XY, SIGMA_XY)
        rand_y = random.uniform(-SIGMA_XY, SIGMA_XY)
        rand_z = random.uniform(-SIGMA_Z, SIGMA_Z)
        
        # 1-3. 최종 제어점 P_i 
        # P_i = P_ref,i + rand
        p_final_x = p_ref_x + rand_x
        p_final_y = p_ref_y + rand_y
        p_final_z = p_ref_z + rand_z
        
        control_points.append(np.array([p_final_x, p_final_y, p_final_z]))
        
        # 가변 폭 스플라인을 위한 노이즈 제어점도 함께 생성
        noise_control.append(random.uniform(0.1, 0.9))

    control_points_arr = np.array(control_points).T  # (3, N)
    noise_control_arr = np.array(noise_control)      # (N,)

    # 2. 스플라인 피팅
    
    # 2.1. 3D 경로 스플라인 (tck_path)
    # per=True (주기적 매듭벡터) 사용
    tck_path, u_path = splprep(control_points_arr, s=0.0, per=True, k=3)
    
    # 2.2. 가변폭 노이즈 스플라인 
    # u_path를 x축으로, noise_control_arr를 y축으로 하는 1D 스플라인 피팅
    tck_noise, _ = splprep([u_path, noise_control_arr], s=0.0, per=True, k=3)

    # 3. 구간별 매개변수 함수 출력 
    print("--- [생성된 3D 트랙의 구간별 매개변수 함수] ---\n")
    
    t, c, k = tck_path
    bspline_x = BSpline(t, c[0], k)
    bspline_y = BSpline(t, c[1], k)
    bspline_z = BSpline(t, c[2], k) 

    ppoly_x = PPoly.from_spline(bspline_x)
    ppoly_y = PPoly.from_spline(bspline_y)
    ppoly_z = PPoly.from_spline(bspline_z) 

    for i in range(len(ppoly_x.x) - 1):
        u_start = ppoly_x.x[i]
        u_end = ppoly_x.x[i+1]

        cx = ppoly_x.c[:, i]  # x(u) 계수
        cy = ppoly_y.c[:, i]  # y(u) 계수
        cz = ppoly_z.c[:, i]  # z(u) 계수 

        x_expr = f"{cx[0]: .4f}*(u-{u_start:.4f})^3 + {cx[1]: .4f}*(u-{u_start:.4f})^2 + {cx[2]: .4f}*(u-{u_start:.4f}) + {cx[3]: .4f}"
        y_expr = f"{cy[0]: .4f}*(u-{u_start:.4f})^3 + {cy[1]: .4f}*(u-{u_start:.4f})^2 + {cy[2]: .4f}*(u-{u_start:.4f}) + {cy[3]: .4f}"
        z_expr = f"{cz[0]: .4f}*(u-{u_start:.4f})^3 + {cz[1]: .4f}*(u-{u_start:.4f})^2 + {cz[2]: .4f}*(u-{u_start:.4f}) + {cz[3]: .4f}"

        print(f"--- 구간 {i+1} (u: {u_start:.4f} ~ {u_end:.4f}) ---")
        print(f"x(u) = {x_expr}")
        print(f"y(u) = {y_expr}")
        print(f"z(u) = {z_expr}\n")
        
    return tck_path, tck_noise

def visualize_3d_track(tck_path, tck_noise):
    """
    생성된 스플라인을 4개의 다른 뷰(Default, Z, Y, X)로 시각화합니다.
    """
    
    # 1. 트랙 설정 
    W_min = 10.0       # 최소 트랙 너비 (m)
    W_rand_max = 50.0  # 최대 랜덤 너비 (m) (사용자 설정 값)
    num_samples = 1000 # 샘플링 개수

    u_fine = np.linspace(0, 1, num_samples)

    # 2. 스플라인 샘플링
    P_center = np.array(splev(u_fine, tck_path))
    D_center = np.array(splev(u_fine, tck_path, der=1))
    noise = splev(u_fine, tck_noise)[1]

    # 3. 가변 폭 및 법선 벡터 계산
    W_offset = (W_min / 2) + (noise * W_rand_max)
    D_2d_norm = np.linalg.norm(D_center[:2], axis=0)
    D_2d_norm[D_2d_norm == 0] = 1 
    Nx = -D_center[1] / D_2d_norm
    Ny = D_center[0] / D_2d_norm
    N_3d = np.array([Nx, Ny, np.zeros_like(Nx)])
    
    # 4. 트랙 경계선 계산
    P_inner = P_center - N_3d * W_offset
    P_outer = P_center + N_3d * W_offset

    # 5. 3D 시각화 (4개의 뷰)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14), 
                                                 subplot_kw={'projection': '3d'})
    fig.suptitle("3D Track Views", fontsize=16)

    # 공통 플로팅 데이터
    X = np.array([P_inner[0], P_outer[0]])
    Y = np.array([P_inner[1], P_outer[1]])
    Z = np.array([P_inner[2], P_outer[2]])

    def plot_track_on_ax(ax):
        """지정된 축에 트랙을 그리는 헬퍼 함수"""
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, rstride=20, cstride=20)
        ax.plot(P_center[0], P_center[1], P_center[2], 'r--', linewidth=1, label='Centerline C(u)')
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend()

    # --- 뷰 1: Default 3D View  ---
    plot_track_on_ax(ax1)
    ax1.view_init(elev=30, azim=45)
    ax1.set_title('Default 3D View (Equal Aspect)')
    set_aspect_equal_3d(ax1) # 실제 비율

    # --- 뷰 2: Top-down (Z-axis) ---
    plot_track_on_ax(ax2)
    ax2.view_init(elev=90, azim=0)
    ax2.set_title('Top-down (Z-axis) View\n(Shows Variable Width)')
    set_aspect_equal_3d(ax2) 

    # --- 뷰 3: Front-on (Y-axis) View ---
    plot_track_on_ax(ax3)
    ax3.view_init(elev=0, azim=0)
    ax3.set_title('Front-on (Y-axis) View\n(Elevation STRETCHED, X-Zoom)')
   
    ax3.set_zlim([-5, 5])

    # --- 뷰 4: Side-on (X-axis) View ---
    plot_track_on_ax(ax4)
    ax4.view_init(elev=0, azim=-90)
    ax4.set_title('Side-on (X-axis) View\n(Elevation STRETCHED, Y-Zoom)')
    
    ax4.set_zlim([-5, 5])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- 메인 실행 ---
if __name__ == "__main__":
    tck_path, tck_noise = generate_track_from_images()
    visualize_3d_track(tck_path, tck_noise)