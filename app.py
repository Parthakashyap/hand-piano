import cv2
import mediapipe as mp
import numpy as np
import pygame
import os
import time
import base64
from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO
import threading
import json

# Initialize Flask app
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize pygame mixer for audio playback with special handling for headless environments
os.environ['SDL_AUDIODRIVER'] = 'dummy'  # Use dummy audio driver when no sound card is available
try:
    pygame.mixer.init()
    pygame.init()
except pygame.error:
    print("Warning: Pygame initialization failed. Audio playback may not work.")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

simplified_fingers = {
    0: "Wrist",
    1: "Thumb",
    2: "Index",
    3: "Middle",
    4: "Ring",
    5: "Pinky"
}

finger_landmark_indices = {
    1: [1, 2, 3, 4],  # Thumb
    2: [5, 6, 7, 8],  # Index
    3: [9, 10, 11, 12],  # Middle
    4: [13, 14, 15, 16],  # Ring
    5: [17, 18, 19, 20]   # Pinky
}

piano_chords = {
    # Major chords
    "C_Major": ["C4", "E4", "G4"],
    "D_Major": ["D4", "Gb4", "A4"],  # Gb is F#
    "E_Major": ["E4", "Ab4", "B4"],  # Ab is G#
    "F_Major": ["F4", "A4", "C5"],
    "G_Major": ["G4", "B4", "D5"],
    "A_Major": ["A4", "Db5", "E5"],  # Db is C#
    "B_Major": ["B4", "Eb5", "Gb5"],  # Eb is D#, Gb is F#
    
    # Minor chords
    "C_Minor": ["C4", "Eb4", "G4"],
    "D_Minor": ["D4", "F4", "A4"],
    "E_Minor": ["E4", "G4", "B4"],
    "F_Minor": ["F4", "Ab4", "C5"],
    "G_Minor": ["G4", "Bb4", "D5"],
    "A_Minor": ["A4", "C5", "E5"],
    "B_Minor": ["B4", "D5", "Gb5"]  # Gb is F#
}

right_palm_chord_mapping = {
    2: "A_Major",  # Index -> A Major
    3: "B_Major",  # Middle -> B Major
    4: "C_Major",  # Ring -> C Major
    5: "D_Major"   # Pinky -> D Major
}

right_back_chord_mapping = {
    2: "E_Major",  # Index -> E Major
    3: "F_Major",  # Middle -> F Major
    4: "G_Major",  # Ring -> G Major
    # No mapping for pinky on back side
}

left_palm_chord_mapping = {
    2: "A_Minor",  # Index -> A Minor
    3: "B_Minor",  # Middle -> B Minor
    4: "C_Minor",  # Ring -> C Minor
    5: "D_Minor"   # Pinky -> D Minor
}

left_back_chord_mapping = {
    2: "E_Minor",  # Index -> E Minor
    3: "F_Minor",  # Middle -> F Minor
    4: "G_Minor",  # Ring -> G Minor
    # No mapping for pinky on back side
}

sound_cache = {}

finger_states = {}
last_chord_time = 0
chord_cooldown = 0.5  # seconds
camera_running = False
camera_thread = None

def load_piano_note(note_name):
    """Load a piano note from the piano-mp3 folder"""
    # Convert note name format (e.g., "C4" to "C4", "Gb4" to "Gb4")
    # Handle special cases for sharps/flats
    if 'b' in note_name:  # Handle flat notes
        note_letter = note_name[0]
        octave = note_name[2] if len(note_name) > 2 else ""
        formatted_note = f"{note_letter}b{octave}"
    else:
        note_letter = note_name[0]
        octave = note_name[1] if len(note_name) > 1 else ""
        formatted_note = f"{note_letter}{octave}"
    
    if formatted_note in sound_cache:
        return sound_cache[formatted_note]
    
    file_path = os.path.join("static", "piano-mp3", f"{formatted_note}.mp3")
    if not os.path.exists(file_path):
        print(f"Warning: Sound file {file_path} not found")
        return None
    
    try:
        sound = pygame.mixer.Sound(file_path)
        sound_cache[formatted_note] = sound
        return sound
    except Exception as e:
        print(f"Error loading sound {file_path}: {e}")
        return None

def play_chord(chord_name, volume=0.5):
    """Play a chord by playing its component notes"""
    if chord_name not in piano_chords:
        print(f"Unknown chord: {chord_name}")
        return
    
    notes = piano_chords[chord_name]
    
    for note in notes:
        sound = load_piano_note(note)
        if sound:
            sound.set_volume(volume)
            sound.play()
    
    print(f"Playing {chord_name} chord: {', '.join(notes)}")
    
    chord_info = {
        'chord_name': chord_name.replace('_', ' '),
        'notes': notes
    }
    socketio.emit('chord_played', chord_info)

def play_chord_for_finger(finger_idx, orientation, handedness, should_play=True):
    """Play a chord based on which finger is extended, hand orientation, and handedness"""
    chord_name = None
    
    if finger_idx == 1:
        return None
    
    if handedness == "Right":
        if orientation == "Palm":
            chord_mapping = right_palm_chord_mapping
        else:  # "Back"
            chord_mapping = right_back_chord_mapping
    else:  # Left hand
        if orientation == "Palm":
            chord_mapping = left_palm_chord_mapping
        else:  # "Back"
            chord_mapping = left_back_chord_mapping
    
    if finger_idx in chord_mapping:
        chord_name = chord_mapping[finger_idx]
    
    if chord_name and should_play:
        play_chord(chord_name)
    
    return chord_name

def is_finger_extended(landmarks, finger_idx, handedness):
    """
    Check if a finger is extended based on its landmarks
    """
    if finger_idx == 1:  # Thumb
        thumb_tip = landmarks.landmark[4]
        thumb_base = landmarks.landmark[2]
        
        if handedness == "Right":
            return thumb_tip.x < thumb_base.x
        else:  # Left hand
            return thumb_tip.x > thumb_base.x
    else:
        finger_indices = finger_landmark_indices[finger_idx]
        tip_idx = finger_indices[-1]
        pip_idx = finger_indices[1]  # PIP joint (middle joint)
        mcp_idx = finger_indices[0]  # MCP joint (base)
        
        tip = landmarks.landmark[tip_idx]
        pip = landmarks.landmark[pip_idx]
        mcp = landmarks.landmark[mcp_idx]
        
        return tip.y < pip.y and pip.y < mcp.y

def detect_hand_orientation(landmarks, handedness):
    """
    Determine if the palm or back of the hand is showing.
    
    This uses the relative positions of the wrist and knuckles to determine orientation.
    Returns: "Palm" or "Back"
    """
    wrist = landmarks.landmark[0]
    index_mcp = landmarks.landmark[5]  # Index finger MCP (knuckle)
    middle_mcp = landmarks.landmark[9]  # Middle finger MCP
    ring_mcp = landmarks.landmark[13]  # Ring finger MCP
    pinky_mcp = landmarks.landmark[17]  # Pinky MCP
    
    thumb_cmc = landmarks.landmark[1]  # Thumb CMC joint
    thumb_mcp = landmarks.landmark[2]  # Thumb MCP joint
    
    v_wrist_to_middle = np.array([
        middle_mcp.x - wrist.x,
        middle_mcp.y - wrist.y,
        middle_mcp.z - wrist.z
    ])
    
    # Vector from index to pinky (across the hand)
    v_index_to_pinky = np.array([
        pinky_mcp.x - index_mcp.x,
        pinky_mcp.y - index_mcp.y,
        pinky_mcp.z - index_mcp.z
    ])
    
    normal = np.cross(v_wrist_to_middle, v_index_to_pinky)
    
    normal = normal / np.linalg.norm(normal)
    
    if handedness == "Right":
        return "Palm" if normal[2] > 0 else "Back"
    else:  # Left hand
        return "Palm" if normal[2] < 0 else "Back"

def draw_hand_info(image, hand_landmarks, handedness, hand_idx, image_width, image_height, 
                  finger_states, last_chord_time, chord_cooldown):
    """
    Draw hand information on the image including landmarks, orientation, and extended fingers
    Returns the chord to play (if any) and updated finger states
    """
    # Set color based on handedness
    hand_color = (0, 255, 0) if handedness == "Left" else (0, 0, 255)
    
    # Draw hand landmarks
    mp_drawing.draw_landmarks(
        image,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style()
    )
    
    # Detect hand orientation (palm or back)
    orientation = detect_hand_orientation(hand_landmarks, handedness)
    
    # Draw wrist (0)
    wrist = hand_landmarks.landmark[0]
    wrist_x = int(wrist.x * image_width)
    wrist_y = int(wrist.y * image_height)
    cv2.putText(image, "0", (wrist_x, wrist_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, hand_color, 2)
    
    # Check which fingers are extended and count them
    extended_fingers = []
    for finger_idx in range(1, 6):  # 1 to 5 (thumb to pinky)
        if is_finger_extended(hand_landmarks, finger_idx, handedness):
            # Only add non-thumb fingers to the extended fingers list
            if finger_idx != 1:  # Skip thumb
                extended_fingers.append(finger_idx)
            
        # Draw finger indices (1-5) and chord names
        landmark_indices = finger_landmark_indices[finger_idx]
        tip_idx = landmark_indices[-1]
        tip = hand_landmarks.landmark[tip_idx]
        tip_x = int(tip.x * image_width)
        tip_y = int(tip.y * image_height)
        
        # Draw finger index
        cv2.putText(image, str(finger_idx), (tip_x, tip_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, hand_color, 2)
        
        # Get chord name based on handedness and orientation
        chord_name = None
        if finger_idx != 1:  # Skip thumb
            if handedness == "Right":
                if orientation == "Palm" and finger_idx in right_palm_chord_mapping:
                    chord_name = right_palm_chord_mapping[finger_idx].replace("_", " ")
                elif orientation == "Back" and finger_idx in right_back_chord_mapping:
                    chord_name = right_back_chord_mapping[finger_idx].replace("_", " ")
            else:  # Left hand
                if orientation == "Palm" and finger_idx in left_palm_chord_mapping:
                    chord_name = left_palm_chord_mapping[finger_idx].replace("_", " ")
                elif orientation == "Back" and finger_idx in left_back_chord_mapping:
                    chord_name = left_back_chord_mapping[finger_idx].replace("_", " ")
        
        if chord_name:
            # Position the chord name to the side of the finger
            offset_x = 20  # Offset to the right
            offset_y = 0   # Same height as finger tip
            
            # Adjust position based on handedness to avoid text overlap
            if handedness == "Left":
                offset_x = -80  # Offset to the left for left hand
            
            # Draw a small semi-transparent background for better readability
            text_size = cv2.getTextSize(chord_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.rectangle(image, 
                         (tip_x + offset_x - 5, tip_y + offset_y - text_size[1] - 5),
                         (tip_x + offset_x + text_size[0] + 5, tip_y + offset_y + 5),
                         (0, 0, 0, 128), -1)
            
            # Draw the chord name
            cv2.putText(image, chord_name, (tip_x + offset_x, tip_y + offset_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    chord_type = "Minor" if handedness == "Left" else "Major"
    hand_text = f"{handedness} {orientation} ({chord_type} chords)"
    text_x = 10
    text_y = 30 + (hand_idx * 90)
    cv2.putText(image, hand_text, (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, hand_color, 2)
    
    if extended_fingers:
        finger_names_text = ", ".join([simplified_fingers[idx] for idx in extended_fingers])
        count_text = f"Showing {len(extended_fingers)} fingers: {finger_names_text}"
    else:
        count_text = "No fingers extended"
        
    cv2.putText(image, count_text, (text_x, text_y + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)
    
    chord_to_play = None
    current_time = time.time()
    can_play = (current_time - last_chord_time) >= chord_cooldown
    
    hand_key = f"{handedness}_{hand_idx}"
    
    if hand_key not in finger_states:
        finger_states[hand_key] = {
            "last_extended": set(),
            "current_extended": set(),
            "last_orientation": None
        }
    
    finger_states[hand_key]["current_extended"] = set(extended_fingers)
    current_orientation = orientation
    
    newly_extended = set()
    for finger in finger_states[hand_key]["current_extended"]:
        if (finger not in finger_states[hand_key]["last_extended"] or 
            finger_states[hand_key]["last_orientation"] != current_orientation):
            newly_extended.add(finger)
    
    if newly_extended and can_play:
        first_finger = min(newly_extended)
        chord_to_play = play_chord_for_finger(first_finger, orientation, handedness, should_play=False)
        
        if chord_to_play:
            chord_text = f"Playing: {chord_to_play.replace('_', ' ')} chord"
            cv2.putText(image, chord_text, (text_x, text_y + 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)
        else:
            if orientation == "Back" and first_finger == 5:
                no_chord_text = "No chord mapped for pinky on back side"
                cv2.putText(image, no_chord_text, (text_x, text_y + 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    finger_states[hand_key]["last_extended"] = finger_states[hand_key]["current_extended"].copy()
    finger_states[hand_key]["last_orientation"] = current_orientation
    
    return chord_to_play, finger_states

def process_camera():
    global finger_states, last_chord_time, camera_running
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Set higher resolution if supported
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while camera_running:
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a selfie-view display
        image = cv2.flip(image, 1)
        
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # To improve performance, optionally mark the image as not writeable
        image_rgb.flags.writeable = False
        
        # Process the image and detect hands
        results = hands.process(image_rgb)
        
        # Make the image writeable again for drawing
        image_rgb.flags.writeable = True
        
        # Get image dimensions
        image_height, image_width, _ = image.shape
         
        # Draw a title on the image
        cv2.putText(image, "Hand Piano - Extend a finger to play a chord", (10, image_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display chord mappings
        cv2.putText(image, "Right Hand: Major Chords (Palm: A-D, Back: E-G)", 
                   (10, image_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.putText(image, "Left Hand: Minor Chords (Palm: Am-Dm, Back: Em-Gm)", 
                   (10, image_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Determine if it's a left or right hand
                handedness = results.multi_handedness[hand_idx].classification[0].label
                
                # Draw hand information and check if we should play a chord
                chord_to_play, finger_states = draw_hand_info(
                    image, hand_landmarks, handedness, hand_idx, 
                    image_width, image_height, finger_states,
                    last_chord_time, chord_cooldown
                )
                
                # If we should play a chord
                if chord_to_play:
                    play_chord(chord_to_play)
                    last_chord_time = time.time()
        
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        
        encoded_frame = base64.b64encode(frame).decode('utf-8')
        socketio.emit('video_frame', {'frame': encoded_frame})
        
        time.sleep(0.03)
    
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_chord_mappings')
def get_chord_mappings():
    mappings = {
        'right_palm': {str(k): v.replace('_', ' ') for k, v in right_palm_chord_mapping.items()},
        'right_back': {str(k): v.replace('_', ' ') for k, v in right_back_chord_mapping.items()},
        'left_palm': {str(k): v.replace('_', ' ') for k, v in left_palm_chord_mapping.items()},
        'left_back': {str(k): v.replace('_', ' ') for k, v in left_back_chord_mapping.items()}
    }
    return jsonify(mappings)

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('start_camera')
def handle_start_camera():
    global camera_running, camera_thread
    
    if not camera_running:
        camera_running = True
        camera_thread = threading.Thread(target=process_camera)
        camera_thread.daemon = True
        camera_thread.start()
        return {'status': 'success', 'message': 'Camera started'}
    
    return {'status': 'error', 'message': 'Camera already running'}

@socketio.on('stop_camera')
def handle_stop_camera():
    global camera_running
    
    if camera_running:
        camera_running = False
        return {'status': 'success', 'message': 'Camera stopped'}
    
    return {'status': 'error', 'message': 'Camera not running'}

if __name__ == '__main__':
    if os.environ.get('RENDER', False):
        port = int(os.environ.get("PORT", 10000))
        socketio.run(app, host='0.0.0.0', port=port)
    else:
        socketio.run(app, debug=True, allow_unsafe_werkzeug=True)