#!/bin/bash
# Opens the FreqSSM-Lite enhancer at http://localhost:8000 (Ctrl+C in this window to stop).
cd "$(dirname "$0")"
(sleep 1; open "http://localhost:8000/enhancer.html") &
python3 -m http.server 8000
