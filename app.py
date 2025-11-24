import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import math
import threading
from collections import deque, namedtuple
from queue import Queue, Empty

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

st.set_page_config(layout="wide", page_title="Fast Gesture → Phrase (Fixed)")

st.sidebar.title("Controls & Performance")

input_width = st.sidebar.selectbox("Input width (px)", options=[160, 240, 320, 480, 640], index=2, help="Smaller = faster, less accurate.")
target_fps = st.sidebar.slider("Processing FPS (target)", 1, 30, 10)
low_power_mode = st.sidebar.checkbox("Fast mode (model_complexity=0)", value=True)

buffer_len = st.sidebar.slider("Stability frames (buffer length)", 3, 12, 6)
vote_needed = st.sidebar.slider("Votes needed to accept", 2, buffer_len, max(3, min(5, buffer_len-1)))

TIP_MCP_MIN = st.sidebar.slider("Tip↔MCP min (sep)", 0.01, 0.12, 0.04, 0.01)
TIP_PIP_Y_OFF = st.sidebar.slider("Tip vs PIP Y offset", 0.02, 0.20, 0.06, 0.01)
PALM_SPREAD_MIN = st.sidebar.slider("Palm spread threshold", 0.02, 0.18, 0.07, 0.01)
OK_DIST = st.sidebar.slider("OK touch distance", 0.01, 0.12, 0.05, 0.01)
CALL_ME_MIN = st.sidebar.slider("Call-me thumb↔pinky", 0.04, 0.16, 0.09, 0.01)
V_MIN_ANGLE = st.sidebar.slider("V min angle (deg)", 6, 30, 18, 1)
V_MAX_ANGLE = st.sidebar.slider("V max angle (deg)", 40, 140, 110, 1)
THUMB_WRIST_Y_OFFSET = st.sidebar.slider("Thumb wrist y offset", 0.01, 0.10, 0.05, 0.01)

show_debug = st.sidebar.checkbox("Show debug overlay (fps, values)", value=True)
enable_tts = st.sidebar.checkbox("Enable TTS (speak phrase)", value=False)
if enable_tts and pyttsx3 is None:
    st.sidebar.warning("Install pyttsx3 for TTS: pip install pyttsx3")

st.sidebar.markdown("")
st.sidebar.write("Tips: good frontal lighting, hold gesture ~0.6s, keep hand 30–60 cm from camera.")

col_vid, col_info = st.columns([3, 1])
video_slot = col_vid.empty()
info_slot = col_info.empty()
start_btn = col_info.button("Start")
stop_btn = col_info.button("Stop")
snapshot_btn = col_info.button("Snapshot")

class CameraThread:
    def __init__(self, src=0, queue_size=1):
        self.src = src
        self.cap = None
        self.running = False
        self.thread = None
        self.frame_q = Queue(maxsize=queue_size)
        self.actual_fps = 0.0

    def start(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(self.src)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        last = time.time()
        frames = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            frames += 1
            now = time.time()
            if now - last >= 1.0:
                self.actual_fps = frames / (now - last)
                last = now
                frames = 0
            try:
                if self.frame_q.full():
                    _ = self.frame_q.get_nowait()
                self.frame_q.put_nowait(frame)
            except Exception:
                pass
            time.sleep(0.001)

    def read_latest(self, timeout=0.01):
        try:
            return self.frame_q.get(timeout=timeout)
        except Empty:
            return None

    def stop(self):
        self.running = False
        try:
            if self.cap:
                self.cap.release()
        except:
            pass

if "camera" not in st.session_state:
    st.session_state.camera = None
if "running" not in st.session_state:
    st.session_state.running = False
if "buffer" not in st.session_state:
    st.session_state.buffer = deque(maxlen=buffer_len)
if "phrase" not in st.session_state:
    st.session_state.phrase = ""
if "phrase_expire" not in st.session_state:
    st.session_state.phrase_expire = 0.0
if "engine" not in st.session_state:
    st.session_state.engine = pyttsx3.init() if (pyttsx3 and enable_tts) else None

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
HandResult = namedtuple("HandResult", ["state", "flags", "info"])

def d(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def v(a, b):
    return (b.x - a.x, b.y - a.y)

def m(vv):
    return math.hypot(vv[0], vv[1])

def ang_between(u, w):
    mu = m(u); mw = m(w)
    if mu == 0 or mw == 0:
        return 180.0
    dot = u[0]*w[0] + u[1]*w[1]
    c = max(-1.0, min(1.0, dot/(mu*mw)))
    return math.degrees(math.acos(c))

def is_finger_up(tip, pip, mcp, tip_mcp_min, tip_pip_off):
    return (tip.y < pip.y - tip_pip_off) and (d(tip, mcp) > tip_mcp_min)

def detect_gesture(lm, params):
    TIP_MCP = params["TIP_MCP_MIN"]
    TIP_PIP = params["TIP_PIP_Y_OFF"]
    PALM_SP = params["PALM_SPREAD_MIN"]
    OK_D = params["OK_DIST"]
    CALL_MIN = params["CALL_ME_MIN"]
    V_MIN = params["V_MIN_ANGLE"]
    V_MAX = params["V_MAX_ANGLE"]
    THUMB_OFF = params["THUMB_WRIST_Y_OFFSET"]

    idx = is_finger_up(lm[8], lm[6], lm[5], TIP_MCP, TIP_PIP)
    mid = is_finger_up(lm[12], lm[10], lm[9], TIP_MCP, TIP_PIP)
    ring = is_finger_up(lm[16], lm[14], lm[13], TIP_MCP, TIP_PIP)
    pinky = is_finger_up(lm[20], lm[18], lm[17], TIP_MCP, TIP_PIP)

    sep = d(lm[4], lm[2]) > TIP_MCP
    tipy = lm[4].y; wriy = lm[0].y
    thumb_up = sep and (tipy < wriy - THUMB_OFF)
    thumb_down = sep and (tipy > wriy + THUMB_OFF)
    flags = (thumb_up or thumb_down, idx, mid, ring, pinky)

    spread = (d(lm[8], lm[12]) + d(lm[12], lm[16]) + d(lm[16], lm[20])) / 3.0
    t_i = d(lm[4], lm[8]); i_m = d(lm[8], lm[12]); t_p = d(lm[4], lm[20])
    v_ang = ang_between(v(lm[6], lm[8]), v(lm[10], lm[12]))

    if not any(flags):
        return HandResult("FIST", flags, {"t_i":t_i,"i_m":i_m,"t_p":t_p,"v_ang":v_ang,"spread":spread})
    if idx and mid and ring and pinky and (spread > PALM_SP):
        return HandResult("PALM", flags, {"t_i":t_i,"i_m":i_m,"t_p":t_p,"v_ang":v_ang,"spread":spread})
    if (t_i < OK_D) and (not mid and not ring and not pinky):
        return HandResult("OK_SIGN", flags, {"t_i":t_i,"i_m":i_m,"t_p":t_p,"v_ang":v_ang,"spread":spread})
    if ((thumb_up or thumb_down) and pinky and (not idx and not mid and not ring) and (t_p > CALL_MIN)):
        return HandResult("CALL_ME", flags, {"t_i":t_i,"i_m":i_m,"t_p":t_p,"v_ang":v_ang,"spread":spread})
    if idx and mid and (not ring and not pinky) and (V_MIN <= v_ang <= V_MAX):
        return HandResult("V_SIGN", flags, {"t_i":t_i,"i_m":i_m,"t_p":t_p,"v_ang":v_ang,"spread":spread})
    if thumb_up and (not idx and not mid and not ring and not pinky):
        return HandResult("THUMB_UP", flags, {"t_i":t_i,"i_m":i_m,"t_p":t_p,"v_ang":v_ang,"spread":spread})
    if thumb_down and (not idx and not mid and not ring and not pinky):
        return HandResult("THUMB_DOWN", flags, {"t_i":t_i,"i_m":i_m,"t_p":t_p,"v_ang":v_ang,"spread":spread})
    if idx and (not mid and not ring and not pinky) and (not thumb_up and not thumb_down):
        return HandResult("INDEX_POINT", flags, {"t_i":t_i,"i_m":i_m,"t_p":t_p,"v_ang":v_ang,"spread":spread})

    return HandResult(None, flags, {"t_i":t_i,"i_m":i_m,"t_p":t_p,"v_ang":v_ang,"spread":spread})

def phrase_map(s):
    return {
        "FIST":"Help","PALM":"I'm okay","THUMB_UP":"Thank you","THUMB_DOWN":"No",
        "INDEX_POINT":"Wait","V_SIGN":"Yes","OK_SIGN":"OK","CALL_ME":"Call me"
    }.get(s, None)

def start_camera():
    if st.session_state.running:
        return
    try:
        st.session_state.camera = CameraThread(src=0, queue_size=1)
        st.session_state.camera.start()
        st.session_state.running = True
        st.session_state.buffer = deque(maxlen=buffer_len)
    except Exception as e:
        st.error(f"Failed to open camera: {e}")
        st.session_state.running = False

def stop_camera():
    cam_obj = st.session_state.get("camera")
    if cam_obj:
        try:
            cam_obj.stop()
        except:
            pass
    st.session_state.camera = None
    st.session_state.running = False

if start_btn:
    start_camera()
if stop_btn:
    stop_camera()

model_complex = 0 if low_power_mode else 1
hands_proc = mp_hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=model_complex, min_detection_confidence=0.5, min_tracking_confidence=0.5)

last_time = 0.0
interval = 1.0 / max(1, target_fps)

try:
    if st.session_state.running and st.session_state.camera is not None:
        cam = st.session_state.camera
        frame = cam.read_latest(timeout=0.02)
        if frame is None:
            video_slot.image(np.zeros((240,320,3), dtype=np.uint8))
            info_slot.markdown("Waiting for camera frames...")
        else:
            h0, w0 = frame.shape[:2]
            scale = input_width / float(w0)
            small = cv2.resize(frame, (input_width, int(h0 * scale)), interpolation=cv2.INTER_LINEAR)
            now = time.time()
            if now - last_time >= interval:
                img_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                res = hands_proc.process(img_rgb)
                last_time = now

                detected_state = None
                debug_info = None
                if res.multi_hand_landmarks:
                    hand_lm = res.multi_hand_landmarks[0].landmark
                    hr = detect_gesture(hand_lm, {
                        "TIP_MCP_MIN": TIP_MCP_MIN,
                        "TIP_PIP_Y_OFF": TIP_PIP_Y_OFF,
                        "PALM_SPREAD_MIN": PALM_SPREAD_MIN,
                        "OK_DIST": OK_DIST,
                        "CALL_ME_MIN": CALL_ME_MIN,
                        "V_MIN_ANGLE": V_MIN_ANGLE,
                        "V_MAX_ANGLE": V_MAX_ANGLE,
                        "THUMB_WRIST_Y_OFFSET": THUMB_WRIST_Y_OFFSET
                    })
                    detected_state = hr.state
                    debug_info = (hr.flags, hr.info)
                    mp_draw.draw_landmarks(small, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

                st.session_state.buffer.append(detected_state)
                accepted = None
                buf = st.session_state.buffer
                if len(buf) == buf.maxlen:
                    counts = {}
                    for s in buf:
                        if s is None: continue
                        counts[s] = counts.get(s, 0) + 1
                    if counts:
                        state, votes = max(counts.items(), key=lambda kv: kv[1])
                        if votes >= vote_needed:
                            accepted = state

                if accepted:
                    ph = phrase_map(accepted)
                    if ph:
                        st.session_state.phrase = ph
                        st.session_state.phrase_expire = time.time() + 1.6
                        if st.session_state.engine and enable_tts:
                            try:
                                st.session_state.engine.say(ph)
                                st.session_state.engine.runAndWait()
                            except Exception:
                                pass

            display_frame = small.copy()
            if st.session_state.phrase and time.time() < st.session_state.phrase_expire:
                txt = st.session_state.phrase
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                cv2.rectangle(display_frame, (10, 10), (10 + tw + 12, 10 + th + 12), (0,0,0), -1)
                cv2.putText(display_frame, txt, (16, 10 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)

            if show_debug and debug_info is not None:
                flags, info = debug_info
                fh = display_frame.shape[0]
                cv2.rectangle(display_frame, (0, fh-90), (display_frame.shape[1], fh), (0,0,0), -1)
                cv2.putText(display_frame, f"Flags:{''.join(['1' if x else '0' for x in flags])}", (6, fh-66), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
                cv2.putText(display_frame, f"spread:{info['spread']:.3f} t-i:{info['t_i']:.3f}", (6, fh-48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
                cv2.putText(display_frame, f"i-m:{info['i_m']:.3f} t-p:{info['t_p']:.3f} v_ang:{info['v_ang']:.1f}", (6, fh-30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

            fps_text = f"Cam FPS: {cam.actual_fps:.1f}"
            info_slot.markdown(f"**Status**: running  \n**Target proc FPS**: {target_fps}  \n**{fps_text}**  \n**Buffer**: {list(st.session_state.buffer)}")

            display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            video_slot.image(display_rgb, use_column_width=True)

            if snapshot_btn:
                ts = int(time.time())
                filename = f"gesture_snapshot_{ts}.png"
                cv2.imwrite(filename, frame)
                st.sidebar.success(f"Saved {filename}")

    else:
        video_slot.image(np.zeros((240, 320, 3), dtype=np.uint8))
        info_slot.markdown("Camera stopped. Press **Start**.")

except Exception as e:
    st.error(f"Runtime error: {e}")

if stop_btn:
    stop_camera()
