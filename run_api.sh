#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
echo "Đang khởi động OmniVoice Studio tại http://localhost:8000..."
python server.py
