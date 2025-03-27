import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import os

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'  # 맥OS의 경우
# 윈도우의 경우: 'Malgun Gothic' 또는 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False


def fast_radon_transform(image, angles):
    """
    FFT와 Fourier Slice Theorem을 이용한 고속 Radon Transform
    
    Parameters:
    -----------
    image : 2D array
        입력 이미지 (정사각형을 가정)
    angles : array
        투영 각도(degree) 배열
    
    Returns:
    --------
    sinogram : 2D array
        라돈 변환 결과 (sinogram)
    """
    # 이미지 크기 확인 및 정사각형 만들기
    if image.shape[0] != image.shape[1]:
        max_dim = max(image.shape)
        padded = np.zeros((max_dim, max_dim))
        x_offset = (max_dim - image.shape[1]) // 2
        y_offset = (max_dim - image.shape[0]) // 2
        padded[y_offset:y_offset+image.shape[0], x_offset:x_offset+image.shape[1]] = image
        image = padded
    
    n = image.shape[0]
    
    # 패딩 및 원형 마스크 적용
    padded_size = int(np.ceil(n * np.sqrt(2)))
    if padded_size % 2 == 1:
        padded_size += 1  # 짝수로 만들기
    
    pad = (padded_size - n) // 2
    padded_img = np.zeros((padded_size, padded_size))
    padded_img[pad:pad+n, pad:pad+n] = image
    
    # 원형 마스크 생성 (선택사항)
    x, y = np.ogrid[:padded_size, :padded_size]
    center = padded_size // 2
    mask = (x - center)**2 + (y - center)**2 <= (n//2)**2
    padded_img = padded_img * mask
    
    # 2D FFT 적용
    fft_img = fftshift(fft2(ifftshift(padded_img)))
    
    # 결과 sinogram 초기화
    diag_len = int(np.ceil(np.sqrt(2) * n))
    sinogram = np.zeros((diag_len, len(angles)))
    
    # 각 각도에 대해 Fourier Slice Theorem 적용
    y_mid = x_mid = fft_img.shape[0] // 2
    
    for i, angle in enumerate(angles):
        # 각도를 라디안으로 변환
        theta = np.deg2rad(angle)
        
        # Fourier 공간에서 직선 추출 (중심을 지나는)
        samples = np.zeros(diag_len, dtype=complex)
        
        # 보간을 위한 좌표 계산
        steps = np.arange(-diag_len//2, diag_len//2)
        x_coords = x_mid + steps * np.cos(theta)
        y_coords = y_mid + steps * np.sin(theta)
        
        # 정수 좌표와 가중치
        x0, y0 = np.floor(x_coords).astype(int), np.floor(y_coords).astype(int)
        x1, y1 = x0 + 1, y0 + 1
        
        # 유효 범위 내 좌표만 사용
        valid = (x0 >= 0) & (x0 < padded_size-1) & (y0 >= 0) & (y0 < padded_size-1)
        
        # 이중 선형 보간법
        dx, dy = x_coords - x0, y_coords - y0
        
        # 유효 범위 내 값만 샘플링
        for j in np.where(valid)[0]:
            w00 = (1 - dx[j]) * (1 - dy[j])
            w01 = (1 - dx[j]) * dy[j]
            w10 = dx[j] * (1 - dy[j])
            w11 = dx[j] * dy[j]
            
            samples[j] = (w00 * fft_img[y0[j], x0[j]] + 
                         w01 * fft_img[y1[j], x0[j]] + 
                         w10 * fft_img[y0[j], x1[j]] + 
                         w11 * fft_img[y1[j], x1[j]])
        
        # 1D 역 FFT
        projection = np.real(np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(samples))))
        sinogram[:, i] = projection
    
    return sinogram


class RadonTransformApp:
    def __init__(self, root):
        self.root = root
        self.root.title("고속 Radon Transform 앱")
        self.root.geometry("1200x800")
        
        # 기본 이미지 및 변수 초기화
        self.current_image = None
        self.current_sinogram = None
        self.angles = np.linspace(0, 180, 180, endpoint=False)
        
        # 메인 프레임 구성
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 제어 프레임 (상단)
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 파일 로드 버튼
        self.load_btn = ttk.Button(self.control_frame, text="이미지 로드", command=self.load_image)
        self.load_btn.pack(side=tk.LEFT, padx=5)
        
        # 테스트 이미지 생성 버튼
        self.test_btn = ttk.Button(self.control_frame, text="테스트 이미지 생성", command=self.create_test_image)
        self.test_btn.pack(side=tk.LEFT, padx=5)
        
        # Radon Transform 실행 버튼
        self.transform_btn = ttk.Button(self.control_frame, text="Radon Transform 실행", command=self.run_transform)
        self.transform_btn.pack(side=tk.LEFT, padx=5)
        
        # 각도 수 설정
        ttk.Label(self.control_frame, text="각도 수:").pack(side=tk.LEFT, padx=(20, 5))
        self.angle_var = tk.StringVar(value="180")
        self.angle_entry = ttk.Entry(self.control_frame, textvariable=self.angle_var, width=5)
        self.angle_entry.pack(side=tk.LEFT, padx=5)
        
        # 그래프 프레임 (하단)
        self.graph_frame = ttk.Frame(self.main_frame)
        self.graph_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Matplotlib 그림 설정
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Matplotlib 툴바 추가
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.graph_frame)
        self.toolbar.update()
        
        # 초기 테스트 이미지 생성
        self.create_test_image()
    
    def load_image(self):
        """사용자가 이미지 파일을 선택하면 로드합니다."""
        file_path = filedialog.askopenfilename(
            title="이미지 파일 선택",
            # macOS에서 문제가 있는 filetypes 형식 수정
            filetypes=[
                ("PNG 파일", "*.png"),
                ("JPEG 파일", "*.jpg *.jpeg"),
                ("TIFF 파일", "*.tif *.tiff"),
                ("BMP 파일", "*.bmp"),
                ("모든 파일", "*")
            ]
        )
        
        if not file_path:  # 파일 선택 취소한 경우
            return
        
        try:
            # PIL 이미지로 로드
            img = Image.open(file_path)
            # Grayscale로 변환
            if img.mode != 'L':
                img = img.convert('L')
            
            # Numpy 배열로 변환
            self.current_image = np.array(img)
            
            # 이미지가 너무 크면 리사이즈
            max_size = 512
            if max(self.current_image.shape) > max_size:
                scale = max_size / max(self.current_image.shape)
                new_size = (
                    int(self.current_image.shape[1] * scale),
                    int(self.current_image.shape[0] * scale)
                )
                img = img.resize(new_size, Image.LANCZOS)
                self.current_image = np.array(img)
            
            # 화면 업데이트
            self.update_display()
            self.root.title(f"고속 Radon Transform 앱 - {os.path.basename(file_path)}")
            
        except Exception as e:
            tk.messagebox.showerror("오류", f"이미지 로드 중 오류 발생: {str(e)}")
        
    def create_test_image(self):
        """원형 테스트 이미지를 생성합니다."""
        size = 256
        x, y = np.ogrid[:size, :size]
        center = size // 2
        r = 70
        test_img = np.zeros((size, size))
        test_img[(x - center)**2 + (y - center)**2 <= r**2] = 1.0
        
        self.current_image = test_img
        self.update_display()
        self.root.title("고속 Radon Transform 앱 - 테스트 이미지")
    
    def run_transform(self):
        """현재 이미지에 대해 Radon Transform을 실행합니다."""
        if self.current_image is None:
            tk.messagebox.showwarning("경고", "먼저 이미지를 로드하거나 생성하세요.")
            return
        
        try:
            # 각도 수 설정 가져오기
            num_angles = int(self.angle_var.get())
            if num_angles < 1:
                raise ValueError("각도 수는 1 이상이어야 합니다.")
            
            self.angles = np.linspace(0, 180, num_angles, endpoint=False)
            
            # 변환 실행 (시간이 오래 걸릴 수 있음)
            self.root.config(cursor="wait")
            self.transform_btn.config(state="disabled")
            self.root.update()
            
            try:
                self.current_sinogram = fast_radon_transform(self.current_image, self.angles)
                self.update_display()
            finally:
                self.root.config(cursor="")
                self.transform_btn.config(state="normal")
            
        except Exception as e:
            tk.messagebox.showerror("오류", f"Radon Transform 실행 중 오류 발생: {str(e)}")
    
    def update_display(self):
        """디스플레이 업데이트"""
        # 기존 그래프 초기화
        self.ax1.clear()
        self.ax2.clear()
        
        # 원본 이미지 표시
        if self.current_image is not None:
            self.ax1.imshow(self.current_image, cmap='gray')
            self.ax1.set_title("입력 이미지")
            self.ax1.axis('off')
        
        # Sinogram 표시
        if self.current_sinogram is not None:
            self.ax2.imshow(self.current_sinogram, cmap='gray', aspect='auto',
                           extent=(0, 180, 0, self.current_sinogram.shape[0]))
            self.ax2.set_title("Sinogram (Radon Transform)")
            self.ax2.set_xlabel("각도 (degree)")
            self.ax2.set_ylabel("투영 위치")
        
        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = RadonTransformApp(root)
    root.mainloop()