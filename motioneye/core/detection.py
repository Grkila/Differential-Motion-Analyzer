
from typing import Literal, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray


def apply_kernel(
    image: NDArray[np.uint8],
    kernel: NDArray[np.float64],
    operation: Literal["convolution", "erosion", "dilation"] = "convolution",
    padding: Literal["zero", "replicate", "reflect"] = "zero",
) -> NDArray[np.uint8]:
    """Primenjuje kernel operaciju na sliku (konvolucija, erozija ili dilatacija).
    
    Args:
        image: Ulazna slika (grayscale ili RGB)
        kernel: Kernel matrica za operaciju
        operation: Tip operacije ("convolution", "erosion", "dilation")
        padding: Metod popunjavanja ivica ("zero", "replicate", "reflect")
        
    Returns:
        Slika nakon primene kernel operacije
    """
    # Dobijanje dimenzija slike
    if len(image.shape) == 2:
        h, w = image.shape
        channels = 1
        img = image[:, :, np.newaxis]  # Dodavanje dimenzije kanala
    else:
        h, w, channels = image.shape
        img = image

    # Dobijanje dimenzija kernela
    k_h, k_w = kernel.shape

    # Izračunavanje veličine padding-a (pretpostavlja kernel neparne veličine)
    pad_h = k_h // 2
    pad_w = k_w // 2

    # Kreiranje slike sa padding-om u zavisnosti od metoda
    if padding == "zero":
        padded = np.zeros((h + 2 * pad_h, w + 2 * pad_w, channels), dtype=np.float64)
        padded[pad_h : pad_h + h, pad_w : pad_w + w, :] = img.astype(np.float64)
    elif padding == "replicate":
        padded = np.pad(
            img.astype(np.float64),
            ((pad_h, pad_h), (pad_w, pad_w), (0, 0)),
            mode="edge",
        )
    elif padding == "reflect":
        padded = np.pad(
            img.astype(np.float64),
            ((pad_h, pad_h), (pad_w, pad_w), (0, 0)),
            mode="reflect",
        )
    else:
        raise ValueError(f"Nepoznat metod padding-a: {padding}")

    # Inicijalizacija izlazne slike
    output = np.zeros((h, w, channels), dtype=np.float64)

    # Prolazak kroz sve piksele slike
    for i in range(h):
        for j in range(w):
            # Izdvajanje regiona od interesa (ROI) oko trenutnog piksela
            roi = padded[i : i + k_h, j : j + k_w, :]

            if operation == "convolution":
                # Konvolucija: množenje kernela sa ROI i sumiranje
                kernel_expanded = kernel[:, :, np.newaxis]
                # Primena kernela na sve kanale
                output[i, j, :] = np.sum(roi * kernel_expanded, axis=(0, 1))

            elif operation == "erosion":
                # Erozija: pronalaženje minimalne vrednosti u ROI
                kernel_mask = kernel > 0
                # Primena operacije na svaki kanal posebno
                for c in range(channels):
                    # Izdvajanje vrednosti pokrivenih kernelom
                    masked_values = roi[:, :, c][kernel_mask]
                    output[i, j, c] = np.min(masked_values) if masked_values.size > 0 else 255.0

            elif operation == "dilation":
                # Dilatacija: pronalaženje maksimalne vrednosti u ROI
                kernel_mask = kernel > 0
                # Primena operacije na svaki kanal posebno
                for c in range(channels):
                    masked_values = roi[:, :, c][kernel_mask]
                    output[i, j, c] = np.max(masked_values) if masked_values.size > 0 else 0.0

    # Ograničavanje vrednosti na opseg [0, 255] i konverzija u uint8
    output = np.clip(output, 0, 255).astype(np.uint8)

    # Uklanjanje dimenzije kanala za grayscale slike
    if channels == 1:
        output = output[:, :, 0]

    return output


def get_mask(
    frame1: NDArray[np.uint8],
    frame2: NDArray[np.uint8],
    kernel: NDArray[np.uint8] = np.array((9, 9), dtype=np.uint8),
    threshold_value: int = 30,
    gaussian_blur_size: int = 7,
) -> NDArray[np.uint8]:
    """Generiše masku pokreta iz dva uzastopna frejma.
    
    Args:
        frame1: Prvi frejm (grayscale)
        frame2: Drugi frejm (grayscale)
        kernel: Morfološki kernel (trenutno se ne koristi u implementaciji)
        threshold_value: Prag za detekciju pokreta
        gaussian_blur_size: Veličina Gaussian blur kernela (mora biti neparan broj)
        
    Returns:
        Binarna maska pokreta
    """
    # Gaussian kernel za potencijalno blur-ovanje (trenutno neaktivan)
    kernel = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
    ]) / 16.0
    
    # Konverzija frejmova u int16 da bi se izbeglo overflow pri oduzimanju
    frame1_int = frame1.astype(np.int16)
    frame2_int = frame2.astype(np.int16)

    # Oduzimanje frejmova za dobijanje razlike (frame differencing)
    diff_int = frame2_int - frame1_int

    # Apsolutna vrednost razlike
    abs_diff_int = np.abs(diff_int)

    # Vraćanje u opseg [0, 255] i konverzija u uint8
    frame_diff = np.clip(abs_diff_int, 0, 255).astype(np.uint8)
 
    # Gaussian blur kernel za zaglađivanje (trenutno neaktivan)
    blur_kernel = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
    ]) / 16.0
    
    #frame_diff = apply_kernel(frame_diff, blur_kernel, operation="convolution") #implementrana funkcija od nule, radi ali je sporija pa koristim ugradjenu
    # Primena Gaussian blur-a za smanjenje šuma
    frame_diff = cv2.GaussianBlur(frame_diff, (gaussian_blur_size, gaussian_blur_size), 0)
    #erosion_kernel = np.ones((3, 3), dtype=np.float64)
    #dilation_kernel = np.ones((3, 3), dtype=np.float64)
    #frame_diff = apply_kernel(frame_diff, erosion_kernel, operation="erosion")
    #rame_diff = apply_kernel(frame_diff, dilation_kernel, operation="dilation")
    #frame_diff = cv2.morphologyEx(frame_diff, cv2.MORPH_CLOSE, kernel, iterations=5)
    
    # Inicijalizacija binarne maske
    mask = np.zeros_like(frame_diff)

    # Primena praga - pikseli sa razlikom većom od praga postaju beli (255)
    mask[frame_diff > threshold_value] = 255

    return mask


def get_contour_detections(
    mask: NDArray[np.uint8], thresh: int = 400
) -> NDArray[np.float64]:
    """Pronalazi konture i generiše detekcije iz binarne maske.
    
    Args:
        mask: Binarna maska pokreta
        thresh: Minimalna površina bounding box-a za validnu detekciju
        
    Returns:
        Niz detekcija [x1, y1, x2, y2, area]
    """
    # Pronalaženje kontura u binarnoj masci
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1
    )

    # Lista za čuvanje detekcija
    detections = []
    for cnt in contours:
        # Izračunavanje bounding box-a za svaku konturu
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        # Filtriranje malih detekcija na osnovu površine
        if area > thresh:
            detections.append([x, y, x + w, y + h, area])

    return np.array(detections) if detections else np.empty((0, 5))


def _remove_contained_bboxes(boxes: NDArray[np.float64]) -> list[int]:
    """Uklanja bounding box-ove koji su potpuno sadržani unutar drugih box-ova.
    
    Args:
        boxes: Niz bounding box-ova [x1, y1, x2, y2]
        
    Returns:
        Lista indeksa box-ova koje treba zadržati
    """
    # Niz za proveru da li je box j potpuno unutar box-a i
    check_array = np.array([True, True, False, False])
    keep = list(range(len(boxes)))

    for i in keep:
        for j in range(len(boxes)):
            # Provera da li je box j potpuno sadržan u box-u i
            if np.all((np.array(boxes[j]) >= np.array(boxes[i])) == check_array):
                try:
                    keep.remove(j)
                except ValueError:
                    continue

    return keep


def non_max_suppression(
    boxes: NDArray[np.float64],
    scores: NDArray[np.float64],
    threshold: float = 1e-1,
) -> NDArray[np.float64]:
    """Non-Maximum Suppression (NMS) za eliminaciju preklapajućih detekcija.
    
    Args:
        boxes: Niz bounding box-ova [x1, y1, x2, y2]
        scores: Skorovi (confidence) za svaki box
        threshold: IoU prag - box-ovi sa većim preklapanjem se uklanjaju
        
    Returns:
        Filtrirani niz bounding box-ova
    """
    # Ako nema box-ova, vrati prazan niz
    if len(boxes) == 0:
        return np.empty((0, 4))

    # Sortiranje box-ova po skorovima (najveći skorovi prvo)
    boxes = boxes[np.argsort(scores)[::-1]]
    # Uklanjanje potpuno sadržanih box-ova
    order = _remove_contained_bboxes(boxes)

    keep = []
    while order:
        # Uzmi box sa najvećim skorom
        i = order.pop(0)
        keep.append(i)

        # Proveri preklapanje sa preostalim box-ovima
        for j in list(order):
            # Izračunavanje preseka (intersection)
            intersection = max(
                0, min(boxes[i][2], boxes[j][2]) - max(boxes[i][0], boxes[j][0])
            ) * max(0, min(boxes[i][3], boxes[j][3]) - max(boxes[i][1], boxes[j][1]))

            # Izračunavanje unije
            union = (
                (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
                + (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1])
                - intersection
            )

            # Izračunavanje IoU (Intersection over Union)
            iou = intersection / union if union > 0 else 0

            # Ako je IoU veći od praga, ukloni box j
            if iou > threshold:
                order.remove(j)

    return boxes[keep]


def get_detections(
    frame1: NDArray[np.uint8],
    frame2: NDArray[np.uint8],
    bbox_thresh: int = 400,
    nms_thresh: float = 1e-3,
    mask_kernel: NDArray[np.uint8] = np.array((9, 9), dtype=np.uint8),
    threshold_value: int = 30,
    gaussian_blur_size: int = 7,
) -> NDArray[np.float64]:
    """Detektuje pokret između dva frejma koristeći frame differencing metodu.
    
    Args:
        frame1: Prvi frejm (grayscale)
        frame2: Drugi frejm (grayscale)
        bbox_thresh: Minimalna površina bounding box-a za validnu detekciju
        nms_thresh: IoU prag za Non-Maximum Suppression
        mask_kernel: Morfološki kernel (trenutno se ne koristi)
        threshold_value: Prag za detekciju pokreta
        gaussian_blur_size: Veličina Gaussian blur kernela (mora biti neparan broj)
        
    Returns:
        Niz bounding box-ova [x1, y1, x2, y2]
    """
    # Korak 1: Generisanje binarne maske pokreta
    mask = get_mask(frame1, frame2, mask_kernel, threshold_value, gaussian_blur_size)
    # Korak 2: Pronalaženje kontura i kreiranje inicijalnih detekcija
    detections = get_contour_detections(mask, bbox_thresh)

    # Ako nema detekcija, vrati prazan niz
    if len(detections) == 0:
        return np.empty((0, 4))

    # Razdvajanje bounding box-ova i skorova (površine)
    bboxes = detections[:, :4]
    scores = detections[:, -1]

    # Korak 3: Primena NMS za eliminaciju redundantnih detekcija
    return non_max_suppression(bboxes, scores, nms_thresh)


def compute_centroid(bbox: NDArray[np.float64]) -> Tuple[float, float]:
    """Izračunava centroid (centar mase) bounding box-a.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        
    Returns:
        Tuple (cx, cy) - koordinate centroida
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return cx, cy

