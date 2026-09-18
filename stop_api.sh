#!/bin/bash
PID=$(lsof -t -i:8000)
if [ -n "$PID" ]; then
  kill -9 $PID
  echo "--> Đã dừng OmniVoice Server (PID: $PID)"
else
  echo "--> Server hiện không chạy trên cổng 8000."
fi
