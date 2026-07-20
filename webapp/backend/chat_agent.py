"""
chat_agent.py
-------------
Benchmark Chat Agent — Gemini 2.5 Flash via OpenRouter with tool calling.

POST /api/chat/message  — SSE streaming, agentic loop ≤3 iterations
GET  /api/chat/history  — last 50 conversation turns
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import sys
from pathlib import Path
from typing import AsyncGenerator, Optional

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from .config import RESULT_ROOT, GT_ROOT
from .upload_scorer import (
    _collect_model_stats,
    _result_path,
    _parse_doc_id,
    _MODELS_FILE,
    _VALID_METRICS,
    get_page_evidence,
    collect_evidence,
)

logger = logging.getLogger(__name__)

# ── Startup guard ─────────────────────────────────────────────────────────────
OPEN_ROUTER_KEY = os.environ.get("OPEN_ROUTER", "")
if not OPEN_ROUTER_KEY:
    logger.error(
        "OPEN_ROUTER environment variable is not set. "
        "Chat Agent router will NOT be registered. "
        "Set OPEN_ROUTER=sk-or-v1-... in .env to enable."
    )

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL          = "google/gemini-2.5-flash"
APP_REFERER    = "http://localhost:8000"
APP_TITLE      = "OCR Benchmark Analyst"

HISTORY_FILE = RESULT_ROOT / ".chat_history.jsonl"

router = APIRouter(prefix="/api/chat", tags=["Chat Agent"])

# ── Tool definitions (OpenAI function-calling format) ─────────────────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_page_evidence",
            "description": (
                "Lấy chi tiết phân tích một trang cụ thể: GT text, pred text, "
                "các đoạn văn bản bị thêm/mất, structural signals. "
                "Dùng khi user hỏi về lý do điểm thấp, lỗi OCR, hay so sánh GT vs prediction của một trang."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id":   {"type": "string", "description": "Document ID, ví dụ: scan_en_001, table_vi_002"},
                    "page_num": {"type": "integer", "description": "Số trang (1-indexed)"},
                    "model":    {"type": "string",  "description": "Tên model. Để trống nếu dùng model đầu tiên có sẵn."},
                },
                "required": ["doc_id", "page_num"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_doc_summary",
            "description": (
                "Lấy tổng quan điểm số của tất cả model trên một document. "
                "Dùng khi user hỏi về performance tổng thể của một tài liệu cụ thể."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "Document ID, ví dụ: scan_en_001"},
                },
                "required": ["doc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_model_comparison",
            "description": (
                "So sánh các model với nhau. Trả về bảng average metrics theo model. "
                "Dùng khi user hỏi model nào tốt hơn, leaderboard, ranking tổng quan."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "uc_type": {
                        "type": "string",
                        "enum": ["scan", "table", "text_layer", "all"],
                        "description": "Lọc theo loại tài liệu. 'all' = tất cả.",
                    },
                    "lang": {
                        "type": "string",
                        "enum": ["en", "vi", "ja", "all"],
                        "description": "Lọc theo ngôn ngữ. 'all' = tất cả.",
                    },
                    "models": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Danh sách model muốn so sánh. Để trống = tất cả model.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_worst_pages",
            "description": (
                "Tìm các trang có điểm kém nhất của một model theo một metric cụ thể. "
                "Dùng khi user muốn biết model yếu nhất ở đâu, trang nào cần cải thiện."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model":  {"type": "string", "description": "Tên model cần phân tích"},
                    "metric": {
                        "type": "string",
                        "enum": sorted(_VALID_METRICS),
                        "description": "Metric cần sort (lower=worse: cer/wer; higher=better: char_f1/teds/...)",
                    },
                    "doc_id": {"type": "string", "description": "Giới hạn trong một document. Trống = tất cả docs."},
                    "top_k":  {"type": "integer", "description": "Số trang trả về (mặc định 5)"},
                },
                "required": ["model", "metric"],
            },
        },
    },
]


# ── Tool implementations ───────────────────────────────────────────────────────

def _tool_get_page_evidence(doc_id: str, page_num: int, model: Optional[str] = None) -> dict:
    return get_page_evidence(doc_id, page_num, model)


def _tool_get_doc_summary(doc_id: str) -> dict:
    """Return all models' per-page metric scores for a document."""
    uc_type, lang = _parse_doc_id(doc_id)
    result: dict[str, list] = {}

    pattern = f"*/{uc_type}/{lang}/{doc_id}_eval.json"
    found_files = list(RESULT_ROOT.glob(pattern))

    if not found_files:
        all_docs = sorted({p.stem.replace("_eval", "") for p in RESULT_ROOT.rglob("*_eval.json")})
        return {
            "error": "document_not_found",
            "details": f"No eval files found for doc_id='{doc_id}'",
            "available_docs": all_docs[:30],
        }

    for eval_path in found_files:
        model_name = eval_path.relative_to(RESULT_ROOT).parts[0]
        try:
            data = json.loads(eval_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        pages = (data.get("text") or {}).get("pages") or []
        result[model_name] = [
            {k: v for k, v in p.items() if k not in ("_evidence", "_meta")}
            for p in pages
        ]

    return {"doc_id": doc_id, "models": result}


def _tool_get_model_comparison(
    uc_type: Optional[str] = None,
    lang: Optional[str] = None,
    models: Optional[list[str]] = None,
) -> dict:
    """Return cross-model comparison table of average metrics."""
    stats = _collect_model_stats()

    rows = []
    for name, s in stats.items():
        if models and name not in models:
            continue

        # Filter by uc_type/lang via by_uc breakdown if needed
        # (simplified: use overall averages, trust _collect_model_stats)
        rows.append({
            "model":       name,
            "docs":        s["docs"],
            "avg_char_f1": s["avg_char_f1"],
            "avg_cer":     s["avg_cer"],
            "avg_teds":    s["avg_teds"],
            "avg_cell_f1": s["avg_cell_f1"],
            "source":      s["source"],
        })

    rows.sort(key=lambda r: (r["avg_char_f1"] or 0), reverse=True)

    return {
        "filters": {"uc_type": uc_type or "all", "lang": lang or "all",
                    "models": models or "all"},
        "comparison": rows,
    }


def _tool_find_worst_pages(
    model: str,
    metric: str,
    doc_id: Optional[str] = None,
    top_k: int = 5,
) -> dict:
    """Return pages ranked by worst score for a given metric and model."""
    if metric not in _VALID_METRICS:
        return {
            "error": "unknown_metric",
            "details": f"'{metric}' is not a valid metric",
            "valid_metrics": sorted(_VALID_METRICS),
        }

    model_dir = RESULT_ROOT / model
    if not model_dir.exists():
        return {"error": "model_not_found", "details": f"No results for model '{model}'"}

    lower_is_worse = metric in ("cer", "wer")
    pages_ranked = []

    eval_files = list(model_dir.rglob("*_eval.json"))
    for eval_path in eval_files:
        d_id = eval_path.stem.replace("_eval", "")
        if doc_id and d_id != doc_id:
            continue
        try:
            data = json.loads(eval_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for p in (data.get("text") or {}).get("pages") or []:
            val = p.get(metric)
            if val is None:
                continue
            pages_ranked.append({
                "doc_id":   d_id,
                "page_num": p.get("page_num"),
                "value":    val,
                "metric":   metric,
                "model":    model,
            })

    if not pages_ranked:
        return {"error": "no_data", "details": f"No pages with metric '{metric}' for model '{model}'"}

    # Sort: for lower_is_worse metrics, highest value = worst; else lowest = worst
    pages_ranked.sort(key=lambda x: x["value"], reverse=lower_is_worse)
    return {
        "model":  model,
        "metric": metric,
        "lower_is_worse": lower_is_worse,
        "worst_pages": pages_ranked[:top_k],
    }


TOOL_DISPATCH = {
    "get_page_evidence":   _tool_get_page_evidence,
    "get_doc_summary":     _tool_get_doc_summary,
    "get_model_comparison": _tool_get_model_comparison,
    "find_worst_pages":    _tool_find_worst_pages,
}


# ── System prompt ─────────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    """Build system prompt with dynamic benchmark context."""
    # Load current model list
    try:
        known_models = json.loads(_MODELS_FILE.read_text()) if _MODELS_FILE.exists() else []
    except Exception:
        known_models = []

    stats = _collect_model_stats()
    all_models = sorted(set(list(stats.keys()) + known_models))
    model_list_str = ", ".join(
        f"{m} ({stats[m]['docs']}/24 docs)" if m in stats else m
        for m in all_models
    )

    return f"""Bạn là OCR Benchmark Analyst — chuyên gia phân tích chất lượng các mô hình OCR.

## Benchmark này có gì
- 24 tài liệu: scan (en/vi/ja), table (en/vi/ja), text_layer (en/vi/ja)
- Models: {model_list_str if model_list_str else "chưa có model nào"}

## Metrics (tất cả tính trên normalized text, sau khi bỏ ảnh và HTML tags)
- **CER** (Character Error Rate): thấp hơn = tốt hơn. >100% = pred dài hơn GT nhiều (insertions)
- **WER** (Word Error Rate): thấp hơn = tốt hơn. Tương tự CER nhưng đơn vị từ.
- **Char F1 / Word F1**: cao hơn = tốt hơn. Đo content overlap, ít nhạy với thứ tự hơn CER.
- **Edit Sim** (Normalized Edit Similarity): cao hơn = tốt hơn. 100% = hoàn toàn giống.
- **TEDS** (Tree Edit Distance Similarity): cao hơn = tốt hơn. Chỉ cho bảng — đo cấu trúc hàng/cột.
- **Cell F1**: cao hơn = tốt hơn. Chỉ cho bảng — đo ô bảng theo vị trí (row, col).

## Patterns phổ biến
- Char F1 cao + CER cao → nội dung đúng nhưng thứ tự đọc khác (reading order error)
- Pred chars >> GT chars → GT thiếu scope HOẶC model đọc thêm caption/label hình
- TEDS thấp + Char F1 cao → text đúng nhưng cấu trúc bảng sai (merge cell, rowspan)
- TABLE_NOT_DETECTED → model đổ nội dung bảng vào text, làm CER tăng vọt

## Tools có sẵn
1. **get_page_evidence(doc_id, page_num, model?)** — lấy GT text, pred text, diff chunks của một trang cụ thể. Dùng khi user hỏi "tại sao điểm thấp", "lỗi gì", "GT vs pred khác nhau chỗ nào".
2. **get_doc_summary(doc_id)** — điểm số tổng quan của tất cả model trên một document.
3. **get_model_comparison(uc_type?, lang?, models?)** — bảng so sánh model.
4. **find_worst_pages(model, metric, doc_id?, top_k?)** — tìm trang yếu nhất.

## Quy tắc trả lời
- Trả lời bằng **cùng ngôn ngữ** với câu hỏi của user (Vietnamese hoặc English).
- **Luôn cite** giá trị metric cụ thể, tên model, doc_id khi đưa ra nhận xét.
- Khi user hỏi về lỗi OCR hay điểm thấp → **gọi get_page_evidence** để có evidence cụ thể.
- Phân biệt "lỗi model" vs "GT không đầy đủ scope" khi phân tích.
- Ngắn gọn, rõ ràng, có ví dụ text thực tế khi available.
"""


# ── OpenRouter API client ─────────────────────────────────────────────────────

async def _call_openrouter_sync(messages: list[dict]) -> dict:
    """Non-streaming call — for tool-selection iterations."""
    headers = {
        "Authorization":  f"Bearer {OPEN_ROUTER_KEY}",
        "Content-Type":   "application/json",
        "HTTP-Referer":   APP_REFERER,
        "X-Title":        APP_TITLE,
    }
    body = {
        "model":    MODEL,
        "messages": messages,
        "tools":    TOOL_DEFINITIONS,
        "stream":   False,
        "temperature": 0.2,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(OPENROUTER_URL, headers=headers, json=body)
        resp.raise_for_status()
        return resp.json()


async def _stream_openrouter_final(messages: list[dict]) -> AsyncGenerator[str, None]:
    """Streaming call for the final answer — no tools."""
    headers = {
        "Authorization":  f"Bearer {OPEN_ROUTER_KEY}",
        "Content-Type":   "application/json",
        "HTTP-Referer":   APP_REFERER,
        "X-Title":        APP_TITLE,
    }
    body = {
        "model":       MODEL,
        "messages":    messages,
        "stream":      True,
        "temperature": 0.3,
        "max_tokens":  2048,
    }
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream("POST", OPENROUTER_URL, headers=headers, json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    chunk = line[6:]
                    if chunk.strip() == "[DONE]":
                        return
                    try:
                        delta = json.loads(chunk)["choices"][0]["delta"]
                        content = delta.get("content") or ""
                        if content:
                            yield content
                    except Exception:
                        pass


# ── Agentic loop ──────────────────────────────────────────────────────────────

async def _run_agentic_loop(
    messages: list[dict],
) -> tuple[str, list[str], int, AsyncGenerator[str, None]]:
    """
    Run the agentic loop (max 3 iterations).
    Returns (tools_called, iteration_count, final_answer_stream_generator).

    Yields SSE events via the returned generator for tool_call signals.
    """
    tools_called: list[str] = []
    iteration = 0
    MAX_ITER = 3

    # Collected SSE non-text events to emit before streaming
    pre_events: list[str] = []

    while iteration < MAX_ITER:
        iteration += 1

        try:
            response = await _call_openrouter_sync(messages)
        except httpx.TimeoutException:
            raise TimeoutError("OpenRouter request timed out after 30s")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"OpenRouter HTTP {e.response.status_code}: {e.response.text[:200]}")

        choice = response.get("choices", [{}])[0]
        finish_reason = choice.get("finish_reason", "stop")
        message = choice.get("message", {})

        # ── Tool call ────────────────────────────────────────────────────
        if finish_reason == "tool_calls" or message.get("tool_calls"):
            tool_calls = message.get("tool_calls") or []

            if iteration >= MAX_ITER:
                # Hit limit — stop, ask LLM to answer with what it has
                messages.append({
                    "role": "system",
                    "content": "Đã đạt giới hạn tool calls. Hãy trả lời dựa trên thông tin hiện có.",
                })
                break

            # Append assistant tool_use message
            messages.append({"role": "assistant", "tool_calls": tool_calls, "content": ""})

            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                tools_called.append(fn_name)

                # SSE signal for UI
                pre_events.append(
                    f"event: tool_call\ndata: {json.dumps({'name': fn_name}, ensure_ascii=False)}\n\n"
                )

                # Dispatch tool
                try:
                    fn_args = json.loads(tc["function"]["arguments"])
                    fn = TOOL_DISPATCH.get(fn_name)
                    if fn:
                        result = fn(**fn_args)
                    else:
                        result = {"error": "unknown_tool", "name": fn_name}
                except Exception as exc:
                    logger.error("Tool %s failed: %s", fn_name, exc)
                    result = {"error": "tool_execution_error", "details": str(exc)[:200]}

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc["id"],
                    "name":         fn_name,
                    "content":      json.dumps(result, ensure_ascii=False, default=str),
                })

            continue  # next iteration

        # ── Final text answer ────────────────────────────────────────────
        # If LLM returned text directly without tool calls
        direct_content = message.get("content", "")
        break

    # Build the final streaming generator
    async def _final_stream():
        # Emit buffered tool_call events first
        for ev in pre_events:
            yield ev

        # Stream final answer
        # Re-call with stream=True for actual streaming output
        final_messages = messages.copy()
        if not any(m.get("role") == "assistant" and m.get("content") for m in final_messages):
            # No final answer yet — need to get one
            async for chunk in _stream_openrouter_final(final_messages):
                yield f"data: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
        else:
            # LLM already produced text directly — emit it character-by-character simulation
            direct_msg = next(
                (m["content"] for m in reversed(final_messages)
                 if m.get("role") == "assistant" and m.get("content")), ""
            )
            # Yield in small chunks to simulate streaming
            chunk_size = 10
            for i in range(0, len(direct_msg), chunk_size):
                yield f"data: {json.dumps({'text': direct_msg[i:i+chunk_size]}, ensure_ascii=False)}\n\n"

        yield "event: done\ndata: {}\n\n"

    return tools_called, iteration, _final_stream()


# ── History helpers ───────────────────────────────────────────────────────────

def _append_history(entry: dict) -> None:
    """Append a conversation turn to the JSONL history file."""
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.error("Failed to write chat history: %s", exc)


# ── API Endpoints ─────────────────────────────────────────────────────────────

@router.post("/message")
async def chat_message(request: Request):
    """
    SSE streaming chat endpoint.
    Body: {"session_id": str, "message": str, "history": list[dict]}
    Returns: text/event-stream with data:, event:tool_call, event:done, event:error events.
    """
    if not OPEN_ROUTER_KEY:
        async def _err():
            yield 'event: error\ndata: {"message": "OPEN_ROUTER key not configured"}\n\n'
        return StreamingResponse(_err(), media_type="text/event-stream")

    body = await request.json()
    session_id = body.get("session_id", "anon")
    user_message = body.get("message", "").strip()
    history: list[dict] = body.get("history", [])

    if not user_message:
        async def _empty():
            yield 'event: error\ndata: {"message": "Empty message"}\n\n'
        return StreamingResponse(_empty(), media_type="text/event-stream")

    # Build messages
    messages: list[dict] = [{"role": "system", "content": _build_system_prompt()}]
    # Include last 6 turns of history
    for turn in history[-6:]:
        if turn.get("role") in ("user", "assistant"):
            messages.append({"role": turn["role"], "content": turn.get("content", "")})
    messages.append({"role": "user", "content": user_message})

    async def _event_stream():
        full_response = ""
        tools_called: list[str] = []
        iterations = 0
        error_occurred = False

        try:
            tools_called, iterations, final_gen = await _run_agentic_loop(messages)

            async for chunk in final_gen:
                # Extract text content from data events for history
                if chunk.startswith("data: ") and "event:" not in chunk:
                    try:
                        payload = json.loads(chunk[6:])
                        full_response += payload.get("text", "")
                    except Exception:
                        pass
                yield chunk

        except TimeoutError:
            error_occurred = True
            yield f'event: error\ndata: {json.dumps({"message": "Request timed out. Please try again."})}\n\n'
            yield "event: done\ndata: {}\n\n"
        except Exception as exc:
            error_occurred = True
            logger.error("Chat agent error: %s", exc)
            safe_msg = str(exc)[:200]
            yield f'event: error\ndata: {json.dumps({"message": f"Agent error: {safe_msg}"})}\n\n'
            yield "event: done\ndata: {}\n\n"

        # Persist to history
        if full_response or error_occurred:
            _append_history({
                "session_id":        session_id,
                "timestamp":         datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "user_message":      user_message,
                "assistant_response": full_response,
                "tools_called":      tools_called,
                "iterations":        iterations,
            })

    return StreamingResponse(_event_stream(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/history")
def get_chat_history(limit: int = 50) -> list[dict]:
    """Return the most recent conversation turns, newest first."""
    if not HISTORY_FILE.exists():
        return []
    try:
        lines = HISTORY_FILE.read_text(encoding="utf-8").strip().splitlines()
        entries = []
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except Exception:
                pass
            if len(entries) >= limit:
                break
        return entries
    except Exception:
        return []
