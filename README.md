# Rescue360 Body Parts Detection

Computer vision system for detecting trapped human body parts in disaster environments using YOLOv8.

## Features

- Real-time body-part detection
- Classes:
  - hand
  - arm
  - head
  - leg
  - foot
  - person
- Insta360 camera support
- FastAPI inference server
- WebRTC streaming prototype
- Local and cloud deployment options

## Tech Stack

- Python
- YOLOv8
- OpenCV
- FastAPI
- WebRTC

## Project Structure

- server.py – inference server
- client_video_live_async.py – live camera client
- preview_cameras.py – camera discovery
- webrtc_poc/ – browser streaming prototype

## Author

Omer Bender
