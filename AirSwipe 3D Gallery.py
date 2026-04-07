import cv2
import mediapipe as mp
import numpy as np
import os
import glob
import math
from collections import deque

GALLERY_PATH = "gallery"
THUMB_SIZE = (180, 120)

# 🔴 RED color instead of neon
RED = (0, 0, 255)

BG_COLOR = (10, 15, 20)
HUD_ALPHA = 0.35
MAX_HISTORY = 8

def load_images(path=GALLERY_PATH, size=THUMB_SIZE):
    images = []
    for f in sorted(glob.glob(os.path.join(path, "*.jpg")) + glob.glob(os.path.join(path, "*.png"))):
        img = cv2.imread(f)
        if img is not None:
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            images.append(img)
    return images

def draw_gallery(frame, images, angle, zoom, selected_idx, anim_t):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2 + 30
    n = len(images)
    radius = int(220 * zoom)
    depth = 180 * zoom
    img_w, img_h = THUMB_SIZE
    order = []

    for i in range(n):
        theta = (2 * np.pi * i / n) + angle
        x = int(cx + radius * np.cos(theta))
        y = int(cy + radius * np.sin(theta) * 0.5)
        z = int(depth * np.sin(theta))
        scale = 1 + 0.25 * (z / depth)

        img = cv2.resize(images[i], (int(img_w * scale), int(img_h * scale)))
        ix, iy = x - img.shape[1] // 2, y - img.shape[0] // 2
        order.append((z, img, ix, iy, i))

    order.sort(key=lambda tup: tup[0])

    for z, img, ix, iy, idx in order:
        ih, iw = img.shape[:2]
        x0, y0 = max(ix, 0), max(iy, 0)
        x1, y1 = min(ix+iw, w), min(iy+ih, h)

        img_x0, img_y0 = x0 - ix, y0 - iy
        img_x1, img_y1 = iw - (ix+iw-x1), ih - (iy+ih-y1)

        if x0 < x1 and y0 < y1:
            border = 4 if idx == selected_idx else 2
            color = RED
            cv2.rectangle(frame, (x0-6, y0-6), (x1+6, y1+6), color, border, cv2.LINE_AA)

            roi = frame[y0:y1, x0:x1]
            img_roi = img[img_y0:img_y1, img_x0:img_x1]

            if roi.shape == img_roi.shape:
                frame[y0:y1, x0:x1] = cv2.addWeighted(roi, 0.3, img_roi, 0.7, 0)

    return frame

def detect_gesture(hand_landmarks, history):
    if not hand_landmarks:
        return None, None, None, None, None

    lm = hand_landmarks[0].landmark
    def pt(idx): return np.array([lm[idx].x, lm[idx].y])

    thumb_tip = pt(4)
    index_tip = pt(8)

    center = (index_tip + thumb_tip) / 2
    pinch_dist = np.linalg.norm(thumb_tip - index_tip)

    history.append(center)

    move = history[-1] - history[-2] if len(history) > 1 else np.array([0, 0])

    return pinch_dist, move, center, thumb_tip, index_tip

# 🔥 CLEAN HUD (removed top bar + gesture text)
def draw_hud(frame, hand_landmarks, gesture, anim_t):
    h, w = frame.shape[:2]

    # Keep center crosshair
    cv2.drawMarker(frame, (w//2, h//2+30), RED, markerType=cv2.MARKER_CROSS, markerSize=32, thickness=2)

    if hand_landmarks and gesture:
        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands

        for hand in hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=RED, thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=RED, thickness=3, circle_radius=2)
            )

        thumb_tip, index_tip = gesture[3], gesture[4]

        if thumb_tip is not None and index_tip is not None:
            tx, ty = int(thumb_tip[0] * w), int(thumb_tip[1] * h)
            ix, iy = int(index_tip[0] * w), int(index_tip[1] * h)

            cx, cy = int((thumb_tip[0]+index_tip[0])/2 * w), int((thumb_tip[1]+index_tip[1])/2 * h)

            cv2.circle(frame, (cx, cy), 12, RED, -1)
            cv2.line(frame, (tx, ty), (ix, iy), RED, 3)

    return frame

def main():
    images = load_images()
    if not images:
        print("No images found in gallery/")
        return

    cap = cv2.VideoCapture(0)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6)

    angle = 0.0
    zoom = 1.0
    selected_idx = 0
    anim_t = 0

    move_history = deque(maxlen=MAX_HISTORY)
    prev_center = None
    base_pinch = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        hand_landmarks = results.multi_hand_landmarks

        pinch_dist, move, center, thumb_tip, index_tip = None, np.array([0,0]), None, None, None

        if hand_landmarks:
            pinch_dist, move, center, thumb_tip, index_tip = detect_gesture(hand_landmarks, move_history)

            if base_pinch is None and pinch_dist is not None:
                base_pinch = pinch_dist

            if base_pinch is not None and pinch_dist is not None:
                zoom = np.clip(1.0 + (pinch_dist - base_pinch) * 18.0, 0.3, 3.5)

            if prev_center is not None and center is not None:
                dx = center[0] - prev_center[0]
                angle += dx * 32.0

            prev_center = center

        hud_frame = draw_hud(frame.copy(), hand_landmarks, (pinch_dist, move, center, thumb_tip, index_tip), anim_t)
        cv2.imshow("Webcam HUD", hud_frame)

        gallery_frame = np.full((480, 640, 3), BG_COLOR, dtype=np.uint8)
        gallery_frame = draw_gallery(gallery_frame, images, angle, zoom, selected_idx, anim_t)
        cv2.imshow("3D Gallery", gallery_frame)

        anim_t += 0.045

        # ✅ EXIT WITH ENTER KEY (13)
        if cv2.waitKey(1) & 0xFF == 13:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
