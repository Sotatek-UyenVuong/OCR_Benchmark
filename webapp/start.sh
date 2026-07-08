#!/bin/bash
# Start OCR Benchmark webapp (backend + frontend)
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Starting backend on :8000 ..."
cd "$ROOT"
.venv/bin/uvicorn webapp.backend.main:app --reload --port 8000 &
BACKEND_PID=$!

echo "Starting frontend on :5173 ..."
cd "$ROOT/webapp/frontend"
npm run dev &
FRONTEND_PID=$!

echo ""
echo "  Backend:  http://localhost:8000/docs"
echo "  Frontend: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both"

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT
wait
