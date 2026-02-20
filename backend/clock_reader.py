"""
clock_reader.py — Analog clock -> digital time using OpenCV
Works on 1-2 vCPU / 2GB RAM. No GPU, no AI model needed.
"""

import cv2
import numpy as np
import math


def read_clock(image_bytes: bytes) -> dict:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Could not decode image", "time": "N/A"}

    img = _resize(img, 600)
    circle = _detect_clock_face(img)
    if circle is None:
        return {"time": "N/A", "confidence": "LOW", "notes": "No clock face detected."}

    cx, cy, r = circle
    clock_img, offset = _crop_clock(img, cx, cy, r)
    local_cx = cx - offset[0]
    local_cy = cy - offset[1]

    hands = _detect_hands(clock_img, local_cx, local_cy, r)

    if len(hands) < 2:
        return {"time": "N/A", "confidence": "LOW", "notes": f"Only detected {len(hands)} hand(s). Need at least 2."}

    hour_angle, minute_angle, confidence = _classify_hands(hands, r)
    hour, minute = _angles_to_time(hour_angle, minute_angle)
    time_str = f"{hour:02d}:{minute:02d}"

    return {
        "time": time_str,
        "period": "AM/PM unknown (24h not determinable from analog)",
        "confidence": confidence,
        "notes": f"Hour hand: {hour_angle:.1f}, Minute hand: {minute_angle:.1f}",
        "hour": hour,
        "minute": minute,
    }


# ---------------------------------------------------------------------------

def _resize(img, max_dim):
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def _detect_clock_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    h, w = gray.shape
    min_r = int(min(h, w) * 0.2)
    max_r = int(min(h, w) * 0.55)

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=min(h, w) * 0.5,
        param1=80, param2=40,
        minRadius=min_r, maxRadius=max_r,
    )

    if circles is None:
        cy, cx = h // 2, w // 2
        r = int(min(h, w) * 0.45)
        return (cx, cy, r)

    circles = np.round(circles[0, :]).astype(int)
    best = sorted(circles, key=lambda c: c[2], reverse=True)[0]
    return (best[0], best[1], best[2])


def _crop_clock(img, cx, cy, r):
    pad = int(r * 0.15)
    h, w = img.shape[:2]
    x1 = max(0, cx - r - pad)
    y1 = max(0, cy - r - pad)
    x2 = min(w, cx + r + pad)
    y2 = min(h, cy + r + pad)
    return img[y1:y2, x1:x2], (x1, y1)


def _detect_hands(img, cx, cy, r):
    """
    Detect clock hands. Returns list of (angle_degrees, length) tuples.

    Improvements over v1:
    - Dual-pass Hough with different thresholds
    - CLAHE contrast enhancement so dark hands pop on light faces
    - Ray-walking to find the TRUE tip of each hand (not just the
      end of the detected line segment, which gets clipped)
    - Angle clustering before tip refinement
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Mask to clock face; blank hub
    mask = np.zeros_like(gray)
    cv2.circle(mask, (cx, cy), int(r * 0.95), 255, -1)
    cv2.circle(mask, (cx, cy), int(r * 0.07), 0, -1)
    masked = cv2.bitwise_and(gray, mask)

    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(masked)

    # Two edge passes merged
    edges1 = cv2.Canny(enhanced, 40, 120, apertureSize=3)
    edges2 = cv2.Canny(enhanced, 60, 180, apertureSize=3)
    edges = cv2.bitwise_or(edges1, edges2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Two Hough passes
    raw_candidates = []
    for min_len, gap in [(int(r * 0.35), int(r * 0.12)),
                         (int(r * 0.22), int(r * 0.20))]:
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180,
            threshold=15, minLineLength=min_len, maxLineGap=gap
        )
        if lines is not None:
            raw_candidates.extend(lines)

    if not raw_candidates:
        return []

    # Filter: one end near center, other toward rim
    hand_segments = []
    for line in raw_candidates:
        x1, y1, x2, y2 = line[0]
        d1 = math.dist((x1, y1), (cx, cy))
        d2 = math.dist((x2, y2), (cx, cy))
        near, far = min(d1, d2), max(d1, d2)
        if near > r * 0.38 or far < r * 0.22:
            continue
        tip_x, tip_y = (x1, y1) if d1 > d2 else (x2, y2)
        angle = _point_to_clock_angle(cx, cy, tip_x, tip_y)
        hand_segments.append((angle, far))

    if not hand_segments:
        return []

    # Cluster by angle (merge duplicate detections of same hand)
    clusters = []
    for angle, length in hand_segments:
        merged = False
        for cluster in clusters:
            if abs(_angle_diff(angle, cluster['angle'])) < 15:
                if length > cluster['length']:
                    cluster['angle'] = angle
                    cluster['length'] = length
                merged = True
                break
        if not merged:
            clusters.append({'angle': angle, 'length': length})

    # Ray-walk: for each cluster, walk outward from center along its angle
    # and find the TRUE farthest dark pixel — fixes truncated tip detection
    inv = cv2.bitwise_not(enhanced)   # dark hand pixels -> bright
    hands = []

    for cluster in clusters:
        angle_rad = math.radians(cluster['angle'])
        dx = math.sin(angle_rad)
        dy = -math.cos(angle_rad)

        true_len = cluster['length']
        for frac in [f / 100.0 for f in range(20, 98, 2)]:
            px = int(cx + dx * r * frac)
            py = int(cy + dy * r * frac)
            h_img, w_img = inv.shape
            if 0 <= px < w_img and 0 <= py < h_img:
                if inv[py, px] > 80:   # was dark in original
                    true_len = r * frac

        hands.append((cluster['angle'], true_len))

    # Final dedup
    deduped = []
    for angle, length in sorted(hands, key=lambda h: h[1], reverse=True):
        if not any(abs(_angle_diff(angle, fa)) < 12 for fa, _ in deduped):
            deduped.append((angle, length))

    # Filter second hand by stroke width.
    # Measure dark pixels perpendicular to each hand at mid-length.
    # Second hand: ~1-3px wide. Hour/minute hands: ~4-10px wide.
    # Only run filter when we have more than 2 candidates.
    if len(deduped) <= 2:
        return deduped

    def measure_width(angle_deg, length):
        angle_rad = math.radians(angle_deg)
        dx = math.sin(angle_rad)
        dy = -math.cos(angle_rad)
        # Sample at 60% along the hand
        sample_x = cx + dx * length * 0.6
        sample_y = cy + dy * length * 0.6
        # Perpendicular direction
        perp_x, perp_y = -dy, dx
        dark_count = 0
        h_img, w_img = inv.shape
        for offset in range(-8, 9):
            px = int(sample_x + perp_x * offset)
            py = int(sample_y + perp_y * offset)
            if 0 <= px < w_img and 0 <= py < h_img:
                if inv[py, px] > 60:
                    dark_count += 1
        return dark_count

    widths = [(a, l, measure_width(a, l)) for a, l in deduped]
    max_w = max(w for _, _, w in widths)
    # Drop any hand that is less than 35% as wide as the widest hand
    filtered = [(a, l) for a, l, w in widths if w >= max_w * 0.35]

    return filtered if len(filtered) >= 2 else deduped


def _classify_hands(hands, r):
    """
    Classify hands into hour and minute.

    Default: shorter hand = hour (always true on analog clocks).
    Override: only if the longer-as-hour assignment saves >15 degrees
    of angular error vs the nearest valid hour position.

    Why not use a consistency score?
    Detection noise of ~10-15deg in the hour hand breaks fraction-based
    scores — the wrong assignment always looks more self-consistent.
    """
    if len(hands) < 2:
        return 0, 0, "LOW"

    candidates = sorted(hands, key=lambda h: h[1], reverse=True)[:2]
    a1, l1 = candidates[0]   # longer
    a2, l2 = candidates[1]   # shorter

    def min_hour_error(hour_a, minute_a):
        """Angular distance from hour_a to the nearest valid hour position."""
        implied_minute = (minute_a / 6) % 60
        errors = []
        for h in range(12):
            expected = (h * 30 + implied_minute * 0.5) % 360
            err = abs((hour_a - expected + 180) % 360 - 180)
            errors.append(err)
        return min(errors)

    err_short_hour = min_hour_error(a2, a1)   # shorter = hour
    err_long_hour  = min_hour_error(a1, a2)   # longer  = hour

    # Only pick longer=hour if it saves more than 15° — a meaningful improvement
    OVERRIDE_THRESHOLD = 15
    if err_long_hour < err_short_hour - OVERRIDE_THRESHOLD:
        hour_angle, minute_angle = a1, a2
        err = err_long_hour
    else:
        hour_angle, minute_angle = a2, a1
        err = err_short_hour

    confidence = "HIGH" if err < 8 else "MEDIUM" if err < 18 else "LOW"
    return hour_angle, minute_angle, confidence


def _angles_to_time(hour_angle, minute_angle):
    minute = round(minute_angle / 6) % 60
    pure_hour_angle = (hour_angle - minute * 0.5) % 360
    hour = int(pure_hour_angle / 30) % 12
    if hour == 0:
        hour = 12
    return hour, minute


def _point_to_clock_angle(cx, cy, px, py):
    dx = px - cx
    dy = py - cy
    return math.degrees(math.atan2(dx, -dy)) % 360


def _angle_diff(a, b):
    return (a - b + 180) % 360 - 180