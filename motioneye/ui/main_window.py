"""GUI Aplikacija za Detekciju Pokreta koristeći Frame Differencing metodu."""

import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QPoint, QRect, QTimer
from PyQt6.QtGui import QAction, QCloseEvent, QImage, QMouseEvent, QPainter, QPen, QPixmap, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)
from numpy.typing import NDArray

from motioneye.core import get_detections, get_mask


class VideoLabel(QLabel):
    """Prilagođen QLabel za prikaz videa sa interakcijom miša."""

    def __init__(self, parent: Optional["MotionDetectorGUI"] = None) -> None:
        """Inicijalizuje video label."""
        super().__init__()
        self.parent_widget = parent

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Obrađuje klik miša za selekciju ROI (Region of Interest)."""
        if self.parent_widget and self.parent_widget.selecting_roi:
            self.parent_widget.roi_start = event.pos()
            self.parent_widget.roi_end = None

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Obrađuje pomeranje miša za crtanje ROI pravougaonika."""
        if (
            self.parent_widget
            and self.parent_widget.selecting_roi
            and self.parent_widget.roi_start
        ):
            self.parent_widget.roi_end = event.pos()
            self.parent_widget._draw_current_frame_with_roi()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Obrađuje puštanje klika miša i finalizuje ROI selekciju."""
        if self.parent_widget and self.parent_widget.selecting_roi:
            self.parent_widget.roi_end = event.pos()
            self.parent_widget._finalize_roi()




class MotionDetectorGUI(QMainWindow):
    """Glavni prozor aplikacije za detekciju pokreta."""

    def __init__(self) -> None:
        """Inicijalizuje glavni prozor aplikacije."""
        super().__init__()
        self.setWindowTitle("MotionEye - Frame Differencing App")
        self.setGeometry(100, 100, 1600, 800)
        
        # Postavljanje ikone prozora
        logo_path = Path(__file__).parent.parent / "assets" / "logo.png"
        if logo_path.exists():
            self.setWindowIcon(QIcon(str(logo_path)))
        
        self.video_path: Optional[str] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.total_frames: int = 0
        self.frame_width: int = 0
        self.frame_height: int = 0
        self.fps: float = 0.0
        self.current_frame_idx: int = 0
        self.is_playing: bool = False
        self.start_frame: int = 0
        self.end_frame: int = 0
        self.roi: Optional[Tuple[int, int, int, int]] = None
        self.roi_start: Optional[QPoint] = None
        self.roi_end: Optional[QPoint] = None
        self.selecting_roi: bool = False
        self.detections: dict[int, NDArray] = {}
        self.masks: dict[int, NDArray] = {}  # Čuva maske pokreta
        self.current_frame_cache: Optional[NDArray] = None
        self.is_camera: bool = False  # Prati da li je izvor kamera
        self.camera_frame_count: int = 0  # Brojač frejmova za kamera mod
        self.prev_camera_frame: Optional[NDArray] = None  # Čuva prethodni frejm za detekciju
        
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self._next_frame)
        
        self._setup_ui()
        self._setup_menubar()
        self._setup_statusbar()
        
        self.statusBar().showMessage("Spremno")

    def closeEvent(self, event: QCloseEvent) -> None:
        """Obrađuje zatvaranje prozora."""
        if self.cap is not None:
            self.cap.release()
        event.accept()

    def _setup_ui(self) -> None:
        """Kreira i konfiguriše komponente korisničkog interfejsa."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Dodavanje logoa na vrhu
        header_layout = QHBoxLayout()
        
        # Logo
        logo_path = Path(__file__).parent.parent / "assets" / "logo.png"
        if logo_path.exists():
            logo_label = QLabel()
            logo_pixmap = QPixmap(str(logo_path))
            # Skaliranje logoa na razumnu veličinu (čuva aspect ratio)
            scaled_logo = logo_pixmap.scaled(
                60, 60,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            logo_label.setPixmap(scaled_logo)
            logo_label.setStyleSheet("QLabel { padding: 5px; }")
            header_layout.addWidget(logo_label)
        
        header_layout.addStretch()
        
        layout.addLayout(header_layout)
        
        # Kreiranje podeljenog prikaza za video i masku
        video_layout = QHBoxLayout()
        
        # Leva strana - Video sa detekcijama
        left_container = QVBoxLayout()
        video_title = QLabel("Video sa Detekcijama")
        video_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        video_title.setStyleSheet("QLabel { font-weight: bold; font-size: 14px; }")
        left_container.addWidget(video_title)
        
        self.video_label = VideoLabel(self)
        self.video_label.setText("Video nije učitan")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(600, 450)
        self.video_label.setStyleSheet("QLabel { background-color: black; color: white; }")
        left_container.addWidget(self.video_label)
        
        video_layout.addLayout(left_container)
        
        # Desna strana - Maska pokreta
        right_container = QVBoxLayout()
        mask_title = QLabel("Maska Pokreta")
        mask_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mask_title.setStyleSheet("QLabel { font-weight: bold; font-size: 14px; }")
        right_container.addWidget(mask_title)
        
        self.mask_label = QLabel()
        self.mask_label.setText("Maska nije izračunata")
        self.mask_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mask_label.setMinimumSize(600, 450)
        self.mask_label.setStyleSheet("QLabel { background-color: black; color: white; }")
        right_container.addWidget(self.mask_label)
        
        video_layout.addLayout(right_container)
        
        layout.addLayout(video_layout)
        
        controls_layout = QHBoxLayout()
        
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self._toggle_playback)
        self.play_button.setEnabled(False)
        controls_layout.addWidget(self.play_button)
        
        self.frame_label = QLabel("Frame: 0/0")
        controls_layout.addWidget(self.frame_label)
        
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.valueChanged.connect(self._on_slider_change)
        self.frame_slider.setEnabled(False)
        layout.addWidget(self.frame_slider)
        
        range_layout = QHBoxLayout()
        
        range_layout.addWidget(QLabel("Start Frame:"))
        self.start_spin = QSpinBox()
        self.start_spin.setMinimum(0)
        self.start_spin.setMaximum(0)
        self.start_spin.valueChanged.connect(self._update_frame_range)
        self.start_spin.setEnabled(False)
        range_layout.addWidget(self.start_spin)
        
        range_layout.addWidget(QLabel("End Frame:"))
        self.end_spin = QSpinBox()
        self.end_spin.setMinimum(0)
        self.end_spin.setMaximum(0)
        self.end_spin.valueChanged.connect(self._update_frame_range)
        self.end_spin.setEnabled(False)
        range_layout.addWidget(self.end_spin)
        
        range_layout.addStretch()
        
        layout.addLayout(range_layout)
        
        detection_layout = QHBoxLayout()
        
        self.roi_button = QPushButton("Select ROI")
        self.roi_button.clicked.connect(self._start_roi_selection)
        self.roi_button.setEnabled(False)
        detection_layout.addWidget(self.roi_button)
        
        self.reset_roi_button = QPushButton("Reset ROI")
        self.reset_roi_button.clicked.connect(self._reset_roi)
        self.reset_roi_button.setEnabled(False)
        detection_layout.addWidget(self.reset_roi_button)
        
        detection_layout.addWidget(QLabel("Bbox Thresh:"))
        self.bbox_thresh_spin = QSpinBox()
        self.bbox_thresh_spin.setRange(100, 10000)
        self.bbox_thresh_spin.setValue(400)
        self.bbox_thresh_spin.setSingleStep(100)
        self.bbox_thresh_spin.valueChanged.connect(self._on_parameter_change)
        detection_layout.addWidget(self.bbox_thresh_spin)
        
        detection_layout.addWidget(QLabel("NMS Thresh (IoU):"))
        self.nms_thresh_spin = QSpinBox()
        self.nms_thresh_spin.setRange(1, 90)
        self.nms_thresh_spin.setValue(30)
        self.nms_thresh_spin.setSingleStep(5)
        self.nms_thresh_spin.valueChanged.connect(self._on_parameter_change)
        detection_layout.addWidget(self.nms_thresh_spin)
        
        # Display actual NMS threshold value (0.01 to 0.90)
        self.nms_value_label = QLabel("(0.30)")
        self.nms_value_label.setStyleSheet("QLabel { color: gray; }")
        self.nms_thresh_spin.valueChanged.connect(lambda v: self.nms_value_label.setText(f"({v/100:.2f})"))
        detection_layout.addWidget(self.nms_value_label)
        
        detection_layout.addWidget(QLabel("Motion Thresh:"))
        self.motion_thresh_spin = QSpinBox()
        self.motion_thresh_spin.setRange(1, 255)
        self.motion_thresh_spin.setValue(30)
        self.motion_thresh_spin.setSingleStep(1)
        self.motion_thresh_spin.valueChanged.connect(self._on_parameter_change)
        detection_layout.addWidget(self.motion_thresh_spin)
        
        detection_layout.addWidget(QLabel("Gaussian Blur:"))
        self.gaussian_blur_spin = QSpinBox()
        self.gaussian_blur_spin.setRange(1, 31)
        self.gaussian_blur_spin.setValue(7)
        self.gaussian_blur_spin.setSingleStep(2)
        self.gaussian_blur_spin.valueChanged.connect(self._ensure_odd_gaussian)
        self.gaussian_blur_spin.valueChanged.connect(self._on_parameter_change)
        detection_layout.addWidget(self.gaussian_blur_spin)
        
        self.save_video_button = QPushButton("Sačuvaj Video sa Detekcijama")
        self.save_video_button.clicked.connect(self._save_video_with_detections)
        self.save_video_button.setEnabled(False)
        detection_layout.addWidget(self.save_video_button)
        
        detection_layout.addStretch()
        
        layout.addLayout(detection_layout)

    def _setup_menubar(self) -> None:
        """Kreira i konfiguriše meni bar."""
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu("&Fajl")
        
        open_action = QAction("&Otvori Video", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_video)
        file_menu.addAction(open_action)
        
        camera_action = QAction("Otvori &Kameru", self)
        camera_action.setShortcut("Ctrl+C")
        camera_action.triggered.connect(self._open_camera)
        file_menu.addAction(camera_action)
        
        close_source_action = QAction("&Zatvori Izvor", self)
        close_source_action.setShortcut("Ctrl+W")
        close_source_action.triggered.connect(self._close_source)
        file_menu.addAction(close_source_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("I&zlaz", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def _setup_statusbar(self) -> None:
        """Kreira i konfiguriše status bar."""
        self.setStatusBar(QStatusBar(self))

    def _open_video(self) -> None:
        """Obrađuje otvaranje video fajla."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Otvorite Video Fajl",
            "",
            "Video Fajlovi (*.mp4 *.avi *.mov *.mkv);;Svi Fajlovi (*.*)",
        )
        
        if not file_path:
            return
        
        try:
            self._load_video(file_path)
        except Exception as e:
            QMessageBox.critical(self, "Greška", f"Neuspešno učitavanje videa: {str(e)}")
            self.statusBar().showMessage("Greška pri učitavanju videa")

    def _open_camera(self) -> None:
        """Obrađuje otvaranje kamere."""
        # Pokušaj otvaranja podrazumevane kamere (index 0)
        try:
            self._load_camera(0)
        except Exception as e:
            QMessageBox.critical(self, "Greška", f"Neuspešno otvaranje kamere: {str(e)}")
            self.statusBar().showMessage("Greška pri otvaranju kamere")

    def _close_source(self) -> None:
        """Zatvara trenutni video ili kamera izvor."""
        if self.cap is not None:
            if self.is_playing:
                self._toggle_playback()
            
            self.cap.release()
            self.cap = None
            self.is_camera = False
            self.camera_frame_count = 0
            self.prev_camera_frame = None
            self.detections.clear()
            self.masks.clear()
            
            self.video_label.setText("Video nije učitan")
            self.mask_label.setText("Maska nije izračunata")
            self.frame_slider.setEnabled(False)
            self.play_button.setEnabled(False)
            self.start_spin.setEnabled(False)
            self.end_spin.setEnabled(False)
            self.roi_button.setEnabled(False)
            self.reset_roi_button.setEnabled(False)
            self.save_video_button.setEnabled(False)
            
            self.statusBar().showMessage("Izvor zatvoren")

    def _load_video(self, video_path: str) -> None:
        """Učitava video fajl i prikazuje prvi frejm."""
        if self.cap is not None:
            self.cap.release()
        
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Nije moguće otvoriti video fajl: {video_path}")
        
        self.is_camera = False
        self.camera_frame_count = 0
        self.prev_camera_frame = None
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.video_path = video_path
        self.current_frame_idx = 0
        self.start_frame = 0
        self.end_frame = self.total_frames - 1
        self.detections.clear()
        self.masks.clear()
        
        self.frame_slider.setMaximum(self.total_frames - 1)
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(True)
        self.play_button.setEnabled(True)
        
        self.start_spin.setMaximum(self.total_frames - 1)
        self.start_spin.setValue(0)
        self.start_spin.setEnabled(True)
        
        self.end_spin.setMaximum(self.total_frames - 1)
        self.end_spin.setValue(self.total_frames - 1)
        self.end_spin.setEnabled(True)
        
        self.roi_button.setEnabled(True)
        self.save_video_button.setEnabled(True)
        self.play_button.setText("Play")
        
        self.statusBar().showMessage(
            f"Loaded: {Path(video_path).name} | "
            f"{self.total_frames} frames | {self.frame_width}x{self.frame_height} | {self.fps:.2f} fps"
        )
        
        self._display_frame(0)

    def _load_camera(self, camera_index: int = 0) -> None:
        """Učitava kamera feed i započinje live preview.
        
        Args:
            camera_index: Indeks kamera uređaja (podrazumevano 0 za primarnu kameru)
        """
        if self.cap is not None:
            self.cap.release()
        
        self.cap = cv2.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            raise ValueError(f"Nije moguće otvoriti kameru {camera_index}")
        
        # Čitanje prvog frejma da bi se dobile dimenzije
        ret, first_frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise ValueError("Nije moguće čitati iz kamere")
        
        self.is_camera = True
        self.camera_frame_count = 0
        self.prev_camera_frame = None
        
        self.fps = 30.0  # Podrazumevani FPS za kameru
        self.total_frames = 0  # Beskonačno za kameru
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.video_path = None
        self.current_frame_idx = 0
        self.detections.clear()
        self.masks.clear()
        
        # Onemogućavanje slajdera i kontrola opsega za kameru
        self.frame_slider.setEnabled(False)
        self.frame_slider.setValue(0)
        
        self.start_spin.setEnabled(False)
        self.end_spin.setEnabled(False)
        
        self.play_button.setEnabled(True)
        self.play_button.setText("Pokreni Kameru")
        self.roi_button.setEnabled(True)
        
        self.statusBar().showMessage(
            f"Kamera {camera_index} otvorena | "
            f"{self.frame_width}x{self.frame_height} | Uživo"
        )
        
        # Prikaz prvog frejma
        self._display_camera_frame(first_frame)

    def _read_frame(self, frame_idx: int) -> Optional[NDArray]:
        """Čita specifičan frejm iz videa."""
        if self.cap is None or frame_idx < 0 or frame_idx >= self.total_frames:
            return None
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        
        return frame if ret else None

    def _display_frame(self, frame_idx: int, compute_realtime: bool = False) -> None:
        """Prikazuje specifičan frejm.
        
        Args:
            frame_idx: Indeks frejma za prikaz
            compute_realtime: Ako je True, izračunava detekcije u realnom vremenu tokom reprodukcije
        """
        if self.cap is None:
            return
        
        frame = self._read_frame(frame_idx)
        if frame is None:
            return
        
        frame = frame.copy()
        mask: Optional[NDArray] = None
        det_count = 0
        
        # Izračunavanje detekcija u realnom vremenu ako je traženo i ako imamo prethodni frejm
        if compute_realtime and frame_idx > 0:
            try:
                prev_frame = self._read_frame(frame_idx - 1)
                if prev_frame is not None:
                    bbox_thresh = self.bbox_thresh_spin.value()
                    nms_thresh = self.nms_thresh_spin.value() / 100.0
                    motion_thresh = self.motion_thresh_spin.value()
                    
                    # Primena ROI ako je selektovan
                    detection_frame = frame.copy()
                    prev_detection_frame = prev_frame.copy()
                    roi_offset = (0, 0)
                    
                    if self.roi:
                        x1, y1, x2, y2 = self.roi
                        x1 = max(0, min(x1, frame.shape[1]))
                        y1 = max(0, min(y1, frame.shape[0]))
                        x2 = max(x1 + 1, min(x2, frame.shape[1]))
                        y2 = max(y1 + 1, min(y2, frame.shape[0]))
                        
                        detection_frame = detection_frame[y1:y2, x1:x2]
                        prev_detection_frame = prev_detection_frame[y1:y2, x1:x2]
                        roi_offset = (x1, y1)
                    
                    # Konverzija u grayscale
                    gray_curr = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2GRAY)
                    gray_prev = cv2.cvtColor(prev_detection_frame, cv2.COLOR_BGR2GRAY)
                    
                    # Dobijanje maske
                    gaussian_blur = self.gaussian_blur_spin.value()
                    mask = get_mask(gray_prev, gray_curr, threshold_value=motion_thresh, gaussian_blur_size=gaussian_blur)
                    
                    # Dobijanje detekcija
                    dets = get_detections(
                        gray_prev,
                        gray_curr,
                        bbox_thresh,
                        nms_thresh,
                        threshold_value=motion_thresh,
                        gaussian_blur_size=gaussian_blur,
                    )
                    
                    # Crtanje detekcija sa ROI offsetom
                    for bbox in dets:
                        x1, y1, x2, y2 = map(int, bbox)
                        x1 += roi_offset[0]
                        y1 += roi_offset[1]
                        x2 += roi_offset[0]
                        y2 += roi_offset[1]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                    
                    det_count = len(dets)
                    
                    # Kreiranje pune maske ako se koristi ROI
                    if self.roi and mask is not None:
                        x1, y1, x2, y2 = self.roi
                        full_mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
                        full_mask[y1:y2, x1:x2] = mask
                        mask = full_mask
                        
            except Exception as e:
                print(f"Greška pri real-time detekciji: {e}")
        else:
            # Korišćenje prethodno izračunatih detekcija ako su dostupne
            if frame_idx in self.detections:
                for bbox in self.detections[frame_idx]:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                
                det_count = len(self.detections[frame_idx])
                mask = self.masks.get(frame_idx)
        
        # Crtanje ROI ako je selektovan
        if self.roi:
            x1, y1, x2, y2 = self.roi
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        
        self.video_label.setPixmap(scaled_pixmap)
        self.current_frame_idx = frame_idx
        self.current_frame_cache = frame.copy()
        
        self.frame_label.setText(
            f"Frejm: {frame_idx + 1}/{self.total_frames} | Detekcija: {det_count}"
        )
        
        # Ažuriranje prikaza maske
        if mask is not None:
            self._display_mask_direct(mask)
        else:
            self._display_mask(frame_idx)

    def _display_camera_frame(self, frame: NDArray) -> None:
        """Prikazuje frejm sa kamere sa detekcijom u realnom vremenu.
        
        Args:
            frame: Frejm sa kamere za prikaz
        """
        if frame is None:
            return
        
        frame_display = frame.copy()
        mask: Optional[NDArray] = None
        
        # Izvršavanje detekcije u realnom vremenu ako imamo prethodni frejm
        if self.prev_camera_frame is not None:
            try:
                bbox_thresh = self.bbox_thresh_spin.value()
                nms_thresh = self.nms_thresh_spin.value() / 100.0
                motion_thresh = self.motion_thresh_spin.value()
                
                # Primena ROI ako je selektovan
                detection_frame = frame.copy()
                prev_detection_frame = self.prev_camera_frame.copy()
                roi_offset = (0, 0)
                
                if self.roi:
                    x1, y1, x2, y2 = self.roi
                    x1 = max(0, min(x1, frame.shape[1]))
                    y1 = max(0, min(y1, frame.shape[0]))
                    x2 = max(x1 + 1, min(x2, frame.shape[1]))
                    y2 = max(y1 + 1, min(y2, frame.shape[0]))
                    
                    detection_frame = detection_frame[y1:y2, x1:x2]
                    prev_detection_frame = prev_detection_frame[y1:y2, x1:x2]
                    roi_offset = (x1, y1)
                
                # Konverzija u grayscale
                gray_curr = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2GRAY)
                gray_prev = cv2.cvtColor(prev_detection_frame, cv2.COLOR_BGR2GRAY)
                
                # Dobijanje maske
                gaussian_blur = self.gaussian_blur_spin.value()
                mask = get_mask(gray_prev, gray_curr, threshold_value=motion_thresh, gaussian_blur_size=gaussian_blur)
                
                # Dobijanje detekcija
                dets = get_detections(
                    gray_prev,
                    gray_curr,
                    bbox_thresh,
                    nms_thresh,
                    threshold_value=motion_thresh,
                    gaussian_blur_size=gaussian_blur,
                )
                
                # Crtanje detekcija sa ROI offsetom
                for bbox in dets:
                    x1, y1, x2, y2 = map(int, bbox)
                    x1 += roi_offset[0]
                    y1 += roi_offset[1]
                    x2 += roi_offset[0]
                    y2 += roi_offset[1]
                    cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    cv2.circle(frame_display, (cx, cy), 5, (255, 0, 0), -1)
                
                det_count = len(dets)
                
            except Exception as e:
                det_count = 0
                print(f"Greška pri detekciji: {e}")
        else:
            det_count = 0
        
        # Crtanje ROI ako je selektovan
        if self.roi:
            x1, y1, x2, y2 = self.roi
            cv2.rectangle(frame_display, (x1, y1), (x2, y2), (255, 255, 0), 2)
        
        # Konverzija u RGB za prikaz
        frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
        
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        
        self.video_label.setPixmap(scaled_pixmap)
        self.current_frame_cache = frame_display.copy()
        
        # Ažuriranje labele frejma
        self.camera_frame_count += 1
        self.frame_label.setText(
            f"Kamera Frejm: {self.camera_frame_count} | Detekcija: {det_count}"
        )
        
        # Prikaz maske ako je dostupna
        if mask is not None:
            # Kreiranje pune maske ako se koristi ROI
            if self.roi:
                x1, y1, x2, y2 = self.roi
                full_mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
                full_mask[y1:y2, x1:x2] = mask
                mask = full_mask
            
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            
            h, w, ch = mask_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(mask_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(
                self.mask_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            
            self.mask_label.setPixmap(scaled_pixmap)
        
        # Čuvanje trenutnog frejma kao prethodnog za sledeću iteraciju
        self.prev_camera_frame = frame.copy()

    def _toggle_playback(self) -> None:
        """Uključuje/isključuje reprodukciju videa ili kamere."""
        if self.cap is None:
            return
        
        self.is_playing = not self.is_playing
        
        if self.is_playing:
            if self.is_camera:
                self.play_button.setText("Zaustavi Kameru")
            else:
                self.play_button.setText("Pauza")
            interval = int(1000 / self.fps) if self.fps > 0 else 33
            self.playback_timer.start(interval)
        else:
            if self.is_camera:
                self.play_button.setText("Pokreni Kameru")
            else:
                self.play_button.setText("Play")
            self.playback_timer.stop()

    def _next_frame(self) -> None:
        """Prelazi na sledeći frejm tokom reprodukcije ili čita frejm sa kamere."""
        if self.cap is None:
            return
        
        if self.is_camera:
            # Čitanje frejma sa kamere
            ret, frame = self.cap.read()
            if ret:
                self._display_camera_frame(frame)
            else:
                # Neuspelo čitanje sa kamere, zaustavi reprodukciju
                self._toggle_playback()
                QMessageBox.warning(self, "Greška Kamere", "Neuspešno čitanje sa kamere")
        else:
            # Reprodukcija videa sa real-time detekcijom
            next_idx = self.current_frame_idx + 1
            
            if next_idx >= self.total_frames:
                next_idx = 0
            
            # Prikaz frejma sa real-time izračunavanjem
            self._display_frame(next_idx, compute_realtime=True)
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(next_idx)
            self.frame_slider.blockSignals(False)

    def _on_slider_change(self, value: int) -> None:
        """Obrađuje promenu vrednosti slajdera."""
        if self.cap is None:
            return
        
        # Koristi real-time detekciju ako se pušta, inače prikazuje prethodno izračunate
        self._display_frame(value, compute_realtime=self.is_playing)

    def _update_frame_range(self) -> None:
        """Ažurira selektovani opseg frejmova."""
        start = self.start_spin.value()
        end = self.end_spin.value()
        
        if start > end:
            self.start_spin.setValue(end)
            start = end
        
        self.start_frame = start
        self.end_frame = end
        
        self.statusBar().showMessage(f"Opseg frejmova: {start} - {end}")

    def _start_roi_selection(self) -> None:
        """Pokreće režim selekcije ROI."""
        self.selecting_roi = True
        self.roi_start = None
        self.roi_end = None
        self.statusBar().showMessage("Nacrtajte ROI na video frejmu...")
        self.roi_button.setText("Selektovanje...")
        self.roi_button.setEnabled(False)

    def _draw_current_frame_with_roi(self) -> None:
        """Ponovo crta trenutni frejm sa ROI pravougaonikom."""
        if self.current_frame_cache is None or not self.roi_start or not self.roi_end:
            return
        
        frame = self.current_frame_cache.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        label_w = self.video_label.width()
        label_h = self.video_label.height()
        
        scale = min(label_w / w, label_h / h)
        scaled_w = int(w * scale)
        scaled_h = int(h * scale)
        
        offset_x = (label_w - scaled_w) // 2
        offset_y = (label_h - scaled_h) // 2
        
        canvas = QPixmap(label_w, label_h)
        canvas.fill(Qt.GlobalColor.black)
        
        painter = QPainter(canvas)
        
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            scaled_w, scaled_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        painter.drawPixmap(offset_x, offset_y, scaled_pixmap)
        
        pen = QPen(Qt.GlobalColor.green, 3, Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        
        x1 = min(self.roi_start.x(), self.roi_end.x())
        y1 = min(self.roi_start.y(), self.roi_end.y())
        x2 = max(self.roi_start.x(), self.roi_end.x())
        y2 = max(self.roi_start.y(), self.roi_end.y())
        
        painter.drawRect(x1, y1, x2 - x1, y2 - y1)
        painter.end()
        
        self.video_label.setPixmap(canvas)

    def _finalize_roi(self) -> None:
        """Finalizuje ROI selekciju."""
        if not self.roi_start or not self.roi_end or self.cap is None:
            self.selecting_roi = False
            self.roi_button.setText("Selektuj ROI")
            self.roi_button.setEnabled(True)
            return
        
        frame_h, frame_w = self.frame_height, self.frame_width
        label_w = self.video_label.width()
        label_h = self.video_label.height()
        
        scale_x = frame_w / label_w
        scale_y = frame_h / label_h
        scale = max(scale_x, scale_y)
        
        scaled_w = int(frame_w / scale)
        scaled_h = int(frame_h / scale)
        offset_x = (label_w - scaled_w) // 2
        offset_y = (label_h - scaled_h) // 2
        
        x1 = int((min(self.roi_start.x(), self.roi_end.x()) - offset_x) * scale)
        y1 = int((min(self.roi_start.y(), self.roi_end.y()) - offset_y) * scale)
        x2 = int((max(self.roi_start.x(), self.roi_end.x()) - offset_x) * scale)
        y2 = int((max(self.roi_start.y(), self.roi_end.y()) - offset_y) * scale)
        
        x1 = max(0, min(x1, frame_w))
        y1 = max(0, min(y1, frame_h))
        x2 = max(0, min(x2, frame_w))
        y2 = max(0, min(y2, frame_h))
        
        self.roi = (x1, y1, x2, y2)
        self.selecting_roi = False
        self.roi_button.setText("Selektuj ROI")
        self.roi_button.setEnabled(True)
        self.reset_roi_button.setEnabled(True)
        
        self.statusBar().showMessage(f"ROI selektovan: ({x1}, {y1}) do ({x2}, {y2})")
        self._display_frame(self.current_frame_idx)

    def _reset_roi(self) -> None:
        """Resetuje ROI selekciju."""
        self.roi = None
        self.roi_start = None
        self.roi_end = None
        self.reset_roi_button.setEnabled(False)
        self.statusBar().showMessage("ROI resetovan")
        self._display_frame(self.current_frame_idx)

    def _ensure_odd_gaussian(self, value: int) -> None:
        """Osigurava da veličina Gaussian blur kernela bude neparna."""
        if value % 2 == 0:
            self.gaussian_blur_spin.setValue(value + 1 if value < 31 else value - 1)
    
    def _on_parameter_change(self) -> None:
        """Obrađuje promenu parametara - osvežava trenutni frejm ako se pušta ili je pauziran."""
        if self.cap is None or self.is_camera:
            return
        
        # Osvežavanje trenutnog frejma sa novim parametrima (real-time ako se pušta)
        self._display_frame(self.current_frame_idx, compute_realtime=True)

    def _display_mask(self, frame_idx: int) -> None:
        """Prikazuje masku pokreta za specifičan frejm.
        
        Args:
            frame_idx: Indeks frejma
        """
        mask = self.masks.get(frame_idx)
        
        if mask is None:
            self.mask_label.setText("Maska nije izračunata za ovaj frejm")
            return
        
        self._display_mask_direct(mask)
    
    def _display_mask_direct(self, mask: NDArray) -> None:
        """Prikazuje masku pokreta direktno.
        
        Args:
            mask: Maska pokreta za prikaz
        """
        # Konverzija maske u RGB za prikaz
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        
        h, w, ch = mask_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(mask_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.mask_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        
        self.mask_label.setPixmap(scaled_pixmap)

    
    def _save_video_with_detections(self) -> None:
        """Čuva video sa nacrtanim detekcijama pokreta."""
        if self.cap is None or self.is_camera:
            QMessageBox.warning(
                self,
                "Nije moguće sačuvati",
                "Molimo učitajte video fajl pre čuvanja. Kamera feed-ovi ne mogu biti sačuvani na ovaj način.",
            )
            return
        
        # Pitanje korisnika za putanju izlaznog fajla
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Sačuvaj Video sa Detekcijama",
            "",
            "Video Fajlovi (*.mp4 *.avi);;Svi Fajlovi (*.*)",
        )
        
        if not output_path:
            return
        
        # Pokretanje procesa čuvanja u zasebnoj niti
        self._start_video_export(output_path)
    
    def _start_video_export(self, output_path: str) -> None:
        """Pokreće proces eksporta videa.
        
        Args:
            output_path: Putanja gde će video biti sačuvan
        """
        # Zaustavljanje reprodukcije ako je pokrenuta
        if self.is_playing:
            self._toggle_playback()
        
        # Onemogućavanje UI tokom eksporta
        self.save_video_button.setEnabled(False)
        self.play_button.setEnabled(False)
        self.frame_slider.setEnabled(False)
        self.start_spin.setEnabled(False)
        self.end_spin.setEnabled(False)
        self.roi_button.setEnabled(False)
        
        try:
            # Dobijanje svojstava videa
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Kodek
            out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.frame_width, self.frame_height))
            
            if not out.isOpened():
                raise ValueError("Neuspešno kreiranje izlaznog video fajla")
            
            # Dobijanje parametara detekcije
            bbox_thresh = self.bbox_thresh_spin.value()
            nms_thresh = self.nms_thresh_spin.value() / 100.0
            motion_thresh = self.motion_thresh_spin.value()
            
            total_frames = self.end_frame - self.start_frame
            
            # Obrada svakog frejma
            for frame_idx in range(self.start_frame, self.end_frame):
                frame = self._read_frame(frame_idx)
                if frame is None:
                    continue
                
                frame = frame.copy()
                
                # Izračunavanje detekcija za ovaj frejm
                if frame_idx > self.start_frame:
                    prev_frame = self._read_frame(frame_idx - 1)
                    if prev_frame is not None:
                        # Primena ROI ako je selektovan
                        detection_frame = frame.copy()
                        prev_detection_frame = prev_frame.copy()
                        roi_offset = (0, 0)
                        
                        if self.roi:
                            x1, y1, x2, y2 = self.roi
                            x1 = max(0, min(x1, frame.shape[1]))
                            y1 = max(0, min(y1, frame.shape[0]))
                            x2 = max(x1 + 1, min(x2, frame.shape[1]))
                            y2 = max(y1 + 1, min(y2, frame.shape[0]))
                            
                            detection_frame = detection_frame[y1:y2, x1:x2]
                            prev_detection_frame = prev_detection_frame[y1:y2, x1:x2]
                            roi_offset = (x1, y1)
                        
                        # Konverzija u grayscale
                        gray_curr = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2GRAY)
                        gray_prev = cv2.cvtColor(prev_detection_frame, cv2.COLOR_BGR2GRAY)
                        
                        # Dobijanje detekcija
                        gaussian_blur = self.gaussian_blur_spin.value()
                        dets = get_detections(
                            gray_prev,
                            gray_curr,
                            bbox_thresh,
                            nms_thresh,
                            threshold_value=motion_thresh,
                            gaussian_blur_size=gaussian_blur,
                        )
                        
                        # Crtanje detekcija sa ROI offsetom
                        for bbox in dets:
                            x1, y1, x2, y2 = map(int, bbox)
                            x1 += roi_offset[0]
                            y1 += roi_offset[1]
                            x2 += roi_offset[0]
                            y2 += roi_offset[1]
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                
                # Crtanje ROI ako je selektovan
                if self.roi:
                    x1, y1, x2, y2 = self.roi
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                
                # Upisivanje frejma u izlazni video
                out.write(frame)
                
                # Ažuriranje progresa
                progress = int(((frame_idx - self.start_frame + 1) * 100) / total_frames) if total_frames > 0 else 0
                self.statusBar().showMessage(
                    f"Čuvanje videa: {progress}% ({frame_idx - self.start_frame + 1}/{total_frames} frejmova)"
                )
                
                # Obrada događaja da bi UI ostao responsivan
                QApplication.processEvents()
            
            # Oslobađanje video writer-a
            out.release()
            
            self.statusBar().showMessage(f"Video uspešno sačuvan na {output_path}")
            QMessageBox.information(
                self,
                "Eksport Završen",
                f"Video sa detekcijama sačuvan na:\n{output_path}",
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Greška pri Eksportu",
                f"Neuspešno čuvanje videa: {str(e)}",
            )
            self.statusBar().showMessage(f"Eksport neuspešan: {str(e)}")
        
        finally:
            # Ponovno omogućavanje UI
            self.save_video_button.setEnabled(True)
            self.play_button.setEnabled(True)
            self.frame_slider.setEnabled(True)
            self.start_spin.setEnabled(True)
            self.end_spin.setEnabled(True)
            self.roi_button.setEnabled(True)


def main() -> None:
    """Pokreće aplikaciju."""
    app = QApplication(sys.argv)
    window = MotionDetectorGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
