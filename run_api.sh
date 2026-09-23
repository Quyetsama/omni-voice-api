#!/bin/bash
cd "$(dirname "$0")"
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "../.venv/bin/activate" ]; then
    source ../.venv/bin/activate
else
    echo "Lỗi: Không tìm thấy virtual environment (.venv)!"
    exit 1
fi

echo "Đang khởi động OmniVoice Studio tại http://localhost:8000..."
python server.py
