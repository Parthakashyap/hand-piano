# Hand Piano 🎹👋

Turn your hands into musical instruments with computer vision! Hand Piano uses your webcam to detect hand gestures and plays piano chords based on which fingers you extend.


## ✨ Features

- **Gesture-Based Music**: Play different piano chords by extending different fingers
- **Hand Position Detection**: Different sounds based on palm or back of hand orientation
- **Dual Hand Support**: Right hand plays major chords, left hand plays minor chords
- **Real-time Feedback**: Visual indicators show which fingers are detected and what chords are playing
- **Web Interface**: Access through any browser with a webcam

## 🎮 How to Play

1. Allow camera access when prompted
2. Position your hand(s) in front of the camera
3. Extend a finger to play a chord:
   - **Right Hand, Palm Side**:
     - Index → A Major
     - Middle → B Major
     - Ring → C Major
     - Pinky → D Major
   - **Right Hand, Back Side**:
     - Index → E Major
     - Middle → F Major
     - Ring → G Major
   - **Left Hand, Palm Side**:
     - Index → A Minor
     - Middle → B Minor
     - Ring → C Minor
     - Pinky → D Minor
   - **Left Hand, Back Side**:
     - Index → E Minor
     - Middle → F Minor
     - Ring → G Minor

## 🛠️ Technology Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Flask, Flask-SocketIO
- **Computer Vision**: OpenCV, MediaPipe Hands
- **Audio**: Pygame


## 🚀 Installation & Setup

### Prerequisites
- Python 3.9+
- Webcam
- Speakers/headphones

### Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/hand-piano.git
   cd hand-piano
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   python app.py
   ```

5. Open your browser and navigate to `http://localhost:5000`


## 🧠 How It Works

Hand Piano uses MediaPipe Hands to detect hand landmarks in real-time video. The application tracks which fingers are extended and determines the hand orientation (palm or back). Based on these inputs, it maps specific gestures to piano chords and plays them using Pygame's audio capabilities.

The web interface communicates with the backend using WebSockets (Flask-SocketIO) to provide real-time feedback without page refreshes.

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add some amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [MediaPipe](https://google.github.io/mediapipe/) for hand tracking capabilities
- [Piano MP3 samples](https://github.com/googlecreativelab/aiexperiments-sound-maker/tree/master/app/public/sounds/piano) for the audio files
- [Flask-SocketIO](https://flask-socketio.readthedocs.io/) for real-time communication
- [OpenCV](https://opencv.org/) for computer vision processing

