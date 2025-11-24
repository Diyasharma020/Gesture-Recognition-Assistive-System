import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time
import math
import argparse
import logging
from collections import deque

import cv2
import mediapipe as mp

HOLD_FRAMES = 8
VOTE_THRESHOLD = 5
PHRASE_TIME = 1.6

TIP_MCP_MIN = 0.04
TIP_PIP_Y_OFF = 0.06
PALM_SPREAD_MIN = 0.07
OK_DIST = 0.05
CALL_ME_MIN = 0.09
V_MIN_ANGLE = 18
V_MAX_ANGLE = 110
THUMB_WRIST_Y_OFFSET = 0.05

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("gesture")

def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def vec(a, b):
    return (b.x - a.x, b.y - a.y)

def mag(v):
    return math.hypot(v[0], v[1])

def angle_between(v1, v2):
    m1 = mag(v1)
    m2 = mag(v2)
    if m1 == 0 or m2 == 0:
        return 180.0
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    c = max(-1.0, min(1.0, dot/(m1*m2)))
    return math.degrees(math.acos(c))

def finger_extended(tip, pip, mcp, tip_pip_offset=TIP_PIP_Y_OFF, tip_mcp_min=TIP_MCP_MIN):
    return (tip.y < pip.y - tip_pip_offset) and (dist(tip, mcp) > tip_mcp_min)

def thumb_up_down(lm, tip_mcp_min=TIP_MCP_MIN, wrist_offset=THUMB_WRIST_Y_OFFSET):
    sep = dist(lm[4], lm[2]) > tip_mcp_min
    tip_y = lm[4].y
    wrist_y = lm[0].y
    return sep and (tip_y < wrist_y - wrist_offset), sep and (tip_y > wrist_y + wrist_offset)

def average_adjacent_tip_spread(lm):
    pts = [lm[8], lm[12], lm[16], lm[20]]
    ds = [dist(pts[i], pts[i+1]) for i in range(3)]
    return sum(ds)/len(ds) if ds else 0.0

def detect_gesture_from_landmarks(lm):
    idx = finger_extended(lm[8], lm[6], lm[5])
    mid = finger_extended(lm[12], lm[10], lm[9])
    ring = finger_extended(lm[16], lm[14], lm[13])
    pinky = finger_extended(lm[20], lm[18], lm[17])
    thumb_up, thumb_down = thumb_up_down(lm)
    flags = (thumb_up or thumb_down, idx, mid, ring, pinky)

    spread = average_adjacent_tip_spread(lm)
    d_ti = dist(lm[4], lm[8])
    d_im = dist(lm[8], lm[12])
    d_tp = dist(lm[4], lm[20])
    v_ang = angle_between(vec(lm[6], lm[8]), vec(lm[10], lm[12]))

    if not any(flags):
        return "FIST", flags, {"spread": spread, "t_i": d_ti, "i_m": d_im, "t_p": d_tp, "v_ang": v_ang}

    if idx and mid and ring and pinky and (spread > PALM_SPREAD_MIN):
        return "PALM", flags, {"spread": spread, "t_i": d_ti, "i_m": d_im, "t_p": d_tp, "v_ang": v_ang}

    if (d_ti < OK_DIST) and (not mid and not ring and not pinky):
        return "OK_SIGN", flags, {"spread": spread, "t_i": d_ti, "i_m": d_im, "t_p": d_tp, "v_ang": v_ang}

    if (flags[0] and pinky and (not idx and not mid and not ring) and (d_tp > CALL_ME_MIN)):
        return "CALL_ME", flags, {"spread": spread, "t_i": d_ti, "i_m": d_im, "t_p": d_tp, "v_ang": v_ang}

    if idx and mid and (not ring and not pinky) and (V_MIN_ANGLE <= v_ang <= V_MAX_ANGLE):
        return "V_SIGN", flags, {"spread": spread, "t_i": d_ti, "i_m": d_im, "t_p": d_tp, "v_ang": v_ang}

    if thumb_up and (not idx and not mid and not ring and not pinky):
        return "THUMB_UP", flags, {"spread": spread, "t_i": d_ti, "i_m": d_im, "t_p": d_tp, "v_ang": v_ang}

    if thumb_down and (not idx and not mid and not ring and not pinky):
        return "THUMB_DOWN", flags, {"spread": spread, "t_i": d_ti, "i_m": d_im, "t_p": d_tp, "v_ang": v_ang}

    if idx and (not mid and not ring and not pinky) and (not thumb_up and not thumb_down):
        return "INDEX_POINT", flags, {"spread": spread, "t_i": d_ti, "i_m": d_im, "t_p": d_tp, "v_ang": v_ang}

    return None, flags, {"spread": spread, "t_i": d_ti, "i_m": d_im, "t_p": d_tp, "v_ang": v_ang}

def phrase_for_state(s):
    return {
        "FIST": "Help",
        "PALM": "I'm okay",
        "THUMB_UP": "Thank you",
        "THUMB_DOWN": "No",
        "INDEX_POINT": "Wait",
        "V_SIGN": "Yes",
        "OK_SIGN": "OK",
        "CALL_ME": "Call me"
    }.get(s)

def run_detector(profile=False):
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    det_buffer = deque(maxlen=HOLD_FRAMES)
    current_phrase = ""
    phrase_until = 0.0

    fps_last = time.time()
    fps_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            fps_count += 1
            now = time.time()
            if profile and now - fps_last >= 2.0:
                fps = fps_count / (now - fps_last)
                log.info(f"FPS: {fps:.1f}")
                fps_last = now
                fps_count = 0

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            detected = None
            dbg = None

            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0].landmark
                state, flags, info = detect_gesture_from_landmarks(lm)
                detected = state
                dbg = (flags, info)
                mp_draw.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

            det_buffer.append(detected)
            accepted = None

            if len(det_buffer) == det_buffer.maxlen:
                counts = {}
                for s in det_buffer:
                    if s:
                        counts[s] = counts.get(s, 0) + 1
                if counts:
                    st, votes = max(counts.items(), key=lambda kv: kv[1])
                    if votes >= VOTE_THRESHOLD:
                        accepted = st

            if accepted:
                ph = phrase_for_state(accepted)
                if ph and (ph != current_phrase or now > phrase_until):
                    current_phrase = ph
                    phrase_until = now + PHRASE_TIME
                    log.info(f"{accepted} -> {ph}")

            if current_phrase and now > phrase_until:
                current_phrase = ""

            h, w, _ = frame.shape
            overlay_h = 130
            cv2.rectangle(frame, (0, h-overlay_h), (w, h), (0,0,0), -1)
            cv2.putText(frame, f"Recent: {list(det_buffer)}", (8, h-overlay_h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

            if dbg:
                flags, info = dbg
                cv2.putText(frame, "Flags:" + "".join(["1" if x else "0" for x in flags]),
                            (8, h-overlay_h+45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
                cv2.putText(frame, f"spread:{info['spread']:.3f} t-i:{info['t_i']:.3f}",
                            (8, h-overlay_h+70), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200,200,200), 1)
                cv2.putText(frame, f"i-m:{info['i_m']:.3f} t-p:{info['t_p']:.3f} ang:{info['v_ang']:.1f}",
                            (8, h-overlay_h+95), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200,200,200), 1)

            if current_phrase:
                size, baseline = cv2.getTextSize(current_phrase, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
                tw, th = size
                x = (w - tw)//2
                y = 70
                cv2.rectangle(frame, (x-10, y-th-10), (x+tw+10, y+10), (0,0,0), -1)
                cv2.putText(frame, current_phrase, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,255), 3)

            cv2.putText(frame, "Press Q to quit", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,180,180), 2)
            cv2.imshow("8-Gesture Detector", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--profile", action="store_true")
    args = p.parse_args()
    run_detector(profile=args.profile)

