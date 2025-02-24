import cv2
import base64
import numpy as np
import pyttsx3
import speech_recognition as sr
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model
model = YOLO("yolov8n.pt")

# Initialize Text-to-Speech
engine = pyttsx3.init()
engine.setProperty("rate", 160)

# Open Camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Optical Flow Initialization
prev_gray = None
turn_direction = "No turn"
detected_objects = []


async def process_video(websocket: WebSocket):
    """Captures video, detects objects, and provides navigation & voice commands."""
    await websocket.accept()
    global prev_gray, turn_direction, detected_objects
    prev_direction = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Object Detection
        results = model(frame)
        detected_objects = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0].item())
                label = model.names[cls]
                width = x2 - x1
                distance = round(500 / width, 2)  # Approximate distance
                detected_objects.append(f"{label} {distance} meters")

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({distance}m)", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Optical Flow for Turn Detection
        turn_direction = "No turn"
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 
                                                0.5, 3, 15, 3, 5, 1.2, 0)
            flow_x = np.mean(flow[..., 0])  

            if flow_x > 1.5:
                turn_direction = "right"
            elif flow_x < -1.5:
                turn_direction = "left"

        prev_gray = gray  

        # Announce Turns
        if turn_direction and turn_direction != prev_direction:
            engine.say(f"Turn {turn_direction} ahead")
            engine.runAndWait()
            prev_direction = turn_direction

        # Convert frame to base64
        _, buffer = cv2.imencode(".jpg", frame)
        frame_base64 = base64.b64encode(buffer).decode()

        # Send frame & detected objects over WebSocket
        await websocket.send_json({
            "image": frame_base64,
            "objects": detected_objects,
            "direction": turn_direction
        })


class VoiceCommand(BaseModel):
    command: str


@app.post("/voice-command/")
async def handle_voice_command(data: VoiceCommand):
    """Handles voice commands from API requests."""
    command = data.command.lower()
    response = virtual_assistant(command)
    print(f"Received command: {command} → Response: {response}")  # Debugging log
    return {"response": response}


@app.get("/detected-objects/")
async def get_detected_objects():
    """API to fetch the last detected objects."""
    return {"objects": detected_objects}


@app.get("/navigation-direction/")
async def get_navigation_direction():
    """API to fetch the current navigation direction."""
    return {"direction": turn_direction}


def virtual_assistant(command):
    """Processes user queries and responds accordingly."""
    if "describe objects" in command:
        return ", ".join(detected_objects) if detected_objects else "No objects detected"
    elif "where should i go" in command:
        return f"You should turn {turn_direction}" if turn_direction else "No turn detected, keep going straight"
    elif "what is your name" in command:
        return "I am your smart assistant, here to guide you."
    elif "how are you" in command:
        return "I'm functioning well, thank you for asking!"
    elif "tell me a joke" in command:
        return "Why did the scarecrow win an award? Because he was outstanding in his field!"
    else:
        return "I'm not sure, but I can help you navigate!"


@app.get("/", response_class=HTMLResponse)
async def serve_html():
    """Serves the web interface for video streaming and voice-controlled navigation."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>Smart Vision & Voice Commands</title>
    </head>
    <body>
        <h1>Real-Time Object Detection & Navigation</h1>
        <img id="videoFeed" width="640" height="480">
        <p id="detectedObjects"></p>
        <p id="navigationInfo"></p>
        <input type="text" id="voiceCommand" placeholder="Enter voice command">
        <button onclick="sendVoiceCommand()">Send Command</button>
        <p id="assistantResponse"></p>

        <script>
            let socket = new WebSocket("ws://localhost:8000/ws");

            socket.onmessage = function(event) {
                let data = JSON.parse(event.data);
                document.getElementById("videoFeed").src = "data:image/jpeg;base64," + data.image;
                document.getElementById("detectedObjects").innerText = "Detected: " + (data.objects || []).join(", ");
                document.getElementById("navigationInfo").innerText = "Navigation: " + data.direction;
            };

            function sendVoiceCommand() {
                let command = document.getElementById("voiceCommand").value;
                fetch("http://localhost:8000/voice-command/", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({"command": command})
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById("assistantResponse").innerText = "Assistant: " + data.response;
                })
                .catch(error => console.error("Fetch error:", error));
            }
        </script>
    </body>
    </html>
    """
    return html_content


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handles WebSocket for real-time streaming, navigation, and voice control."""
    await process_video(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
