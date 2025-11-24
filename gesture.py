import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2, time, math
import mediapipe as mp
from collections import deque

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

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1,
                      min_detection_confidence=0.6,
                      min_tracking_confidence=0.6)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

buffer = deque(maxlen=HOLD_FRAMES)
current_phrase = ""
phrase_until = 0.0

def dist(a,b):
    return math.hypot(a.x-b.x, a.y-b.y)

def vec(a,b):
    return (b.x-a.x, b.y-a.y)

def mag(v):
    return math.hypot(v[0], v[1])

def angle_between(v1, v2):
    m1 = mag(v1); m2 = mag(v2)
    if m1==0 or m2==0: return 180.0
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    c = max(-1.0, min(1.0, dot/(m1*m2)))
    return math.degrees(math.acos(c))

def finger_extended(tip, pip, mcp):
    return (tip.y < pip.y - TIP_PIP_Y_OFF) and (dist(tip, mcp) > TIP_MCP_MIN)

def thumb_up_down(lm):
    sep = dist(lm[4], lm[2]) > TIP_MCP_MIN
    tip_y = lm[4].y
    wrist_y = lm[0].y
    up = sep and (tip_y < wrist_y - THUMB_WRIST_Y_OFFSET)
    down = sep and (tip_y > wrist_y + THUMB_WRIST_Y_OFFSET)
    return up, down

def avg_adjacent_tip_spread(lm):
    pts = [lm[8], lm[12], lm[16], lm[20]]
    ds = []
    for i in range(3):
        ds.append(dist(pts[i], pts[i+1]))
    return sum(ds)/3.0 if ds else 0.0

def detect(lm):
    idx = finger_extended(lm[8], lm[6], lm[5])
    mid = finger_extended(lm[12], lm[10], lm[9])
    ring = finger_extended(lm[16], lm[14], lm[13])
    pinky = finger_extended(lm[20], lm[18], lm[17])
    thumb_up, thumb_down = thumb_up_down(lm)
    flags = (thumb_up or thumb_down, idx, mid, ring, pinky)
    spread = avg_adjacent_tip_spread(lm)

  
    t_i = dist(lm[4], lm[8])
    i_m = dist(lm[8], lm[12])
    t_p = dist(lm[4], lm[20])

    v1 = vec(lm[6], lm[8])
    v2 = vec(lm[10], lm[12])
    v_ang = angle_between(v1, v2)

    if not any(flags):
        return "FIST", flags, {"t_i":t_i,"i_m":i_m,"t_p":t_p,"v_ang":v_ang,"spread":spread}

    if idx and mid and ring and pinky and (spread > PALM_SPREAD_MIN):
        return "PALM", flags, {"t_i":t_i,"i_m":i_m,"t_p":t_p,"v_ang":v_ang,"spread":spread}

    if (t_i < OK_DIST) and (not mid and not ring and not pinky):
        return "OK_SIGN", flags, {"t_i":t_i,"i_m":i_m,"t_p":t_p,"v_ang":v_ang,"spread":spread}

    if ( (flags[0]) and pinky and (not idx and not mid and not ring) and (t_p > CALL_ME_MIN) ):
        return "CALL_ME", flags, {"t_i":t_i,"i_m":i_m,"t_p":t_p,"v_ang":v_ang,"spread":spread}

    if idx and mid and (not ring and not pinky) and (V_MIN_ANGLE <= v_ang <= V_MAX_ANGLE):
        return "V_SIGN", flags, {"t_i":t_i,"i_m":i_m,"t_p":t_p,"v_ang":v_ang,"spread":spread}

    if thumb_up and (not idx and not mid and not ring and not pinky):
        return "THUMB_UP", flags, {"t_i":t_i,"i_m":i_m,"t_p":t_p,"v_ang":v_ang,"spread":spread}

    if thumb_down and (not idx and not mid and not ring and not pinky):
        return "THUMB_DOWN", flags, {"t_i":t_i,"i_m":i_m,"t_p":t_p,"v_ang":v_ang,"spread":spread}

    if idx and (not mid and not ring and not pinky) and (not thumb_up and not thumb_down):
        return "INDEX_POINT", flags, {"t_i":t_i,"i_m":i_m,"t_p":t_p,"v_ang":v_ang,"spread":spread}

    return None, flags, {"t_i":t_i,"i_m":i_m,"t_p":t_p,"v_ang":v_ang,"spread":spread}

def phrase_of(s):
    return {
        "FIST":"Help","PALM":"I'm fine","THUMB_UP":"Thank you","THUMB_DOWN":"No",
        "INDEX_POINT":"Wait","V_SIGN":"Yes","OK_SIGN":"OK","CALL_ME":"Call me"
    }.get(s, None)

while True:
    ok, frame = cap.read()
    if not ok: break
    h,w,_ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    detected = None
    dbg = None

    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0].landmark
        state, flags, info = detect(lm)
        detected = state
        dbg = (flags, info)
        mp_draw.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

    buffer.append(detected)
    accepted = None
    if len(buffer) == buffer.maxlen:
        counts = {}
        for s in buffer:
            if s is None: continue
            counts[s] = counts.get(s,0) + 1
        if counts:
            st, votes = max(counts.items(), key=lambda kv: kv[1])
            if votes >= VOTE_THRESHOLD:
                accepted = st

    now = time.time()
    if accepted:
        ph = phrase_of(accepted)
        if ph and (ph != current_phrase or now > phrase_until):
            current_phrase = ph
            phrase_until = now + PHRASE_TIME
            print(f"[{time.strftime('%H:%M:%S')}] Accepted: {accepted} -> {ph}")

    if current_phrase and now > phrase_until:
        current_phrase = ""

    overlay_h = 140
    cv2.rectangle(frame, (0,h-overlay_h),(w,h),(0,0,0),-1)
    cv2.putText(frame, f"Recent:{list(buffer)}", (8,h-overlay_h+18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200),1)

    if dbg:
        flags, info = dbg
        flags_text = "Thumb,Idx,Mid,Rng,Pky: " + "".join(["1" if x else "0" for x in flags])
        cv2.putText(frame, flags_text, (8,h-overlay_h+45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200),1)
        cv2.putText(frame, f"spread:{info['spread']:.3f} t-i:{info['t_i']:.3f} i-m:{info['i_m']:.3f} t-p:{info['t_p']:.3f}", (8,h-overlay_h+72), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200,200,200),1)
        cv2.putText(frame, f"v_ang:{info['v_ang']:.1f}deg", (8,h-overlay_h+100), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200,200,200),1)

    if current_phrase:
        (size, baseline) = cv2.getTextSize(current_phrase, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        tw, th = size[0], size[1]
        x = (w - tw)//2; y = 80
        cv2.rectangle(frame, (x-10,y-th-10),(x+tw+10,y+10),(0,0,0),-1)
        cv2.putText(frame, current_phrase, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)

    cv2.putText(frame, "Press Q to quit", (8,26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,180,180), 2)
    cv2.imshow("8 gestures - follow the on-screen hints", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
