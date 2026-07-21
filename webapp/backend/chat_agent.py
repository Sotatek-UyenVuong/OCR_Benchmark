"""
chat_agent.py
-------------
Benchmark Chat Agent — Gemini 2.5 Flash via OpenRouter with tool calling.

POST /api/chat/message  — SSE streaming, agentic loop ≤3 iterations
GET  /api/chat/history  — last 50 conversation turns
"""
from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import sys
import time
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
OPENAI_KEY      = os.environ.get("OPENAI_API_KEY", "")

# Use OpenAI directly if available, otherwise fall back to OpenRouter
if OPENAI_KEY:
    _API_KEY     = OPENAI_KEY
    _API_URL     = "https://api.openai.com/v1/chat/completions"
    _API_HEADERS_EXTRA = {}
    MODEL        = "gpt-4o-mini"
    _ENABLED     = True
elif OPEN_ROUTER_KEY:
    _API_KEY     = OPEN_ROUTER_KEY
    _API_URL     = "https://openrouter.ai/api/v1/chat/completions"
    _API_HEADERS_EXTRA = {"HTTP-Referer": APP_REFERER, "X-Title": APP_TITLE}
    MODEL        = "google/gemini-2.5-flash"
    _ENABLED     = True
else:
    _API_KEY = _API_URL = MODEL = ""
    _API_HEADERS_EXTRA = {}
    _ENABLED = False
    logger.error(
        "Neither OPENAI_API_KEY nor OPEN_ROUTER is set. "
        "Chat Agent router will NOT be registered."
    )

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
    {
        "type": "function",
        "function": {
            "name": "get_model_recommendation",
            "description": (
                "Gợi ý model tốt nhất cho một use case cụ thể: loại tài liệu (scan/table/text_layer) "
                "và/hoặc ngôn ngữ (en/vi/ja). Trả về ranking và phân tích điểm mạnh/yếu từng model. "
                "Dùng khi user hỏi: 'nên dùng model nào cho scan tiếng Việt', 'model nào tốt nhất cho "
                "table tiếng Nhật', 'model phù hợp nhất cho text_layer tiếng Anh', v.v."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "uc_type": {
                        "type": "string",
                        "enum": ["scan", "table", "text_layer"],
                        "description": "Loại tài liệu cần gợi ý. Bỏ trống = so sánh tất cả uc_type.",
                    },
                    "lang": {
                        "type": "string",
                        "enum": ["en", "vi", "ja"],
                        "description": "Ngôn ngữ cần gợi ý. Bỏ trống = so sánh tất cả ngôn ngữ.",
                    },
                    "metric": {
                        "type": "string",
                        "enum": ["char_f1", "cer", "teds", "cell_f1"],
                        "description": "Metric chính để xếp hạng. Mặc định: char_f1 cho scan/text_layer, teds cho table.",
                    },
                },
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
    """Return cross-model comparison table, filtered by uc_type/lang when provided."""
    stats = _collect_model_stats()
    use_breakdown = (uc_type and uc_type != "all") or (lang and lang != "all")

    rows = []
    for name, s in stats.items():
        if models and name not in models:
            continue

        if use_breakdown:
            # Pull metrics from granular by_uc breakdown
            by_uc = s.get("by_uc", {})
            uc_keys  = [uc_type] if (uc_type and uc_type != "all") else list(by_uc.keys())
            lang_keys = [lang]   if (lang and lang != "all") else None

            cf1_vals, cer_vals, teds_vals, cell_vals, docs = [], [], [], [], 0
            for uc in uc_keys:
                for lg, b in (by_uc.get(uc) or {}).items():
                    if lang_keys and lg not in lang_keys:
                        continue
                    docs += b.get("docs", 0)
                    if b.get("avg_char_f1") is not None: cf1_vals.append(b["avg_char_f1"])
                    if b.get("avg_cer")     is not None: cer_vals.append(b["avg_cer"])
                    if b.get("avg_teds")    is not None: teds_vals.append(b["avg_teds"])
                    if b.get("avg_cell_f1") is not None: cell_vals.append(b["avg_cell_f1"])

            def _m(lst): return round(sum(lst)/len(lst), 4) if lst else None
            row = {
                "model": name, "docs": docs,
                "avg_char_f1": _m(cf1_vals), "avg_cer": _m(cer_vals),
                "avg_teds": _m(teds_vals), "avg_cell_f1": _m(cell_vals),
                "source": s["source"],
            }
        else:
            row = {
                "model":       name,
                "docs":        s["docs"],
                "avg_char_f1": s["avg_char_f1"],
                "avg_cer":     s["avg_cer"],
                "avg_teds":    s["avg_teds"],
                "avg_cell_f1": s["avg_cell_f1"],
                "source":      s["source"],
            }

        if row["docs"] > 0:  # skip models with no data for this filter
            rows.append(row)

    rows.sort(key=lambda r: (r["avg_char_f1"] or 0), reverse=True)
    return {
        "filters": {"uc_type": uc_type or "all", "lang": lang or "all",
                    "models": models or "all"},
        "comparison": rows,
    }


def _tool_get_model_recommendation(
    uc_type: Optional[str] = None,
    lang: Optional[str] = None,
    metric: Optional[str] = None,
) -> dict:
    """Return ranked models + analysis for a specific uc_type × lang combination."""
    stats = _collect_model_stats()

    # Default metric: teds for table, char_f1 for others
    if not metric:
        metric = "teds" if uc_type == "table" else "char_f1"

    lower_is_better = metric == "cer"
    metric_key = f"avg_{metric}"

    # Determine which uc_type×lang combos to cover
    all_ucs  = ["scan", "table", "text_layer"]
    all_langs = ["en", "vi", "ja"]
    uc_filter   = [uc_type] if uc_type else all_ucs
    lang_filter = [lang]    if lang    else all_langs

    results = {}  # (uc, lang) -> list of {model, value, docs}

    for uc in uc_filter:
        for lg in lang_filter:
            combo_rows = []
            for name, s in stats.items():
                b = (s.get("by_uc") or {}).get(uc, {}).get(lg)
                if not b or b.get("docs", 0) == 0:
                    continue
                val = b.get(metric_key)
                if val is None:
                    continue
                combo_rows.append({
                    "model": name,
                    "value": round(val * 100, 1),  # percent
                    "docs":  b["docs"],
                    "char_f1": round((b.get("avg_char_f1") or 0) * 100, 1),
                    "cer":     round((b.get("avg_cer")     or 0) * 100, 1),
                    "teds":    round((b.get("avg_teds")    or 0) * 100, 1) if b.get("avg_teds") else None,
                    "cell_f1": round((b.get("avg_cell_f1") or 0) * 100, 1) if b.get("avg_cell_f1") else None,
                })
            if combo_rows:
                combo_rows.sort(key=lambda r: r["value"], reverse=not lower_is_better)
                results[f"{uc}/{lg}"] = {
                    "ranked": combo_rows,
                    "best":   combo_rows[0]["model"],
                    "metric": metric,
                }

    if not results:
        return {
            "error": "no_data",
            "details": f"Không có dữ liệu cho uc_type={uc_type!r} lang={lang!r}",
        }

    return {
        "query":   {"uc_type": uc_type or "all", "lang": lang or "all", "metric": metric},
        "results": results,
        "note": (
            "Giá trị tính theo %. "
            f"{'CER thấp hơn = tốt hơn.' if lower_is_better else 'Điểm cao hơn = tốt hơn.'} "
            "Chỉ hiện model có dữ liệu cho combination đó."
        ),
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
    "get_page_evidence":        _tool_get_page_evidence,
    "get_doc_summary":          _tool_get_doc_summary,
    "get_model_comparison":     _tool_get_model_comparison,
    "get_model_recommendation": _tool_get_model_recommendation,
    "find_worst_pages":         _tool_find_worst_pages,
}


# ── System prompt ─────────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    """Build concise system prompt with dynamic benchmark context."""
    try:
        known_models = json.loads(_MODELS_FILE.read_text()) if _MODELS_FILE.exists() else []
    except Exception:
        known_models = []

    stats = _collect_model_stats()
    all_models = sorted(set(list(stats.keys()) + known_models))
    model_list_str = ", ".join(
        f"{m}({stats[m]['docs']}/24)" if m in stats else m
        for m in all_models
    ) or "none yet"

    return f"""OCR Benchmark Analyst. Answer in user's language (Vietnamese/English).

BENCHMARK: 24 docs (scan/table/text_layer × en/vi/ja). Models: {model_list_str}

METRICS (cite values in answers):
- CER/WER: lower=better. >100% means pred longer than GT.
- Char F1/Word F1/Edit Sim: higher=better.
- TEDS/Cell F1: table structure, higher=better.

TOOLS — call autonomously when needed:
- get_page_evidence(doc_id,page_num,model?): GT vs pred text + diff. USE for "why low score" questions.
  Returns pred_md_full (full page markdown) when available (upload models) for deep content comparison.
- get_doc_summary(doc_id): all models' scores per page for one doc.
- get_model_comparison(uc_type?,lang?,models?): cross-model table, filtered by uc_type/lang.
- get_model_recommendation(uc_type?,lang?,metric?): best model for a specific use case (scan/table/text_layer × en/vi/ja). USE when user asks "which model is best for Vietnamese scan", "recommend model for Japanese table", etc.
- find_worst_pages(model,metric,doc_id?,top_k?): worst scoring pages.

RULES:
- Always cite metric values + model names.
- Distinguish "model error" vs "GT scope issue".
- For "which model is best for X": use get_model_recommendation, NOT get_model_comparison.
- When asked about high CER or low scores: call get_page_evidence for the SINGLE worst page only."""



# ── OpenRouter API client ─────────────────────────────────────────────────────

async def _call_openrouter_sync(messages: list[dict]) -> dict:
    """
    Single streaming call that handles both tool calls and direct answers.
    Accumulates the full streaming response into a dict mimicking the non-streaming format.
    This reduces latency by ~30-40% vs two separate calls.
    """
    headers = {
        "Authorization": f"Bearer {_API_KEY}",
        "Content-Type":  "application/json",
        **_API_HEADERS_EXTRA,
    }
    body = {
        "model":       MODEL,
        "messages":    messages,
        "tools":       TOOL_DEFINITIONS,
        "stream":      True,   # use streaming even for tool selection
        "temperature": 0.2,
        "max_tokens":  1024,
    }

    accumulated_content = ""
    accumulated_tool_calls: dict[int, dict] = {}  # index → {id, name, arguments_buf}
    finish_reason = "stop"

    async with httpx.AsyncClient(timeout=30) as client:
        async with client.stream("POST", _API_URL, headers=headers, json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                chunk = line[6:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    d = json.loads(chunk)
                    choice = d.get("choices", [{}])[0]
                    delta  = choice.get("delta", {})
                    fr     = choice.get("finish_reason")
                    if fr:
                        finish_reason = fr

                    # Accumulate text content
                    if delta.get("content"):
                        accumulated_content += delta["content"]

                    # Accumulate tool calls (streamed as fragments)
                    for tc_delta in delta.get("tool_calls") or []:
                        idx = tc_delta.get("index", 0)
                        if idx not in accumulated_tool_calls:
                            accumulated_tool_calls[idx] = {
                                "id": tc_delta.get("id", ""),
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        tc = accumulated_tool_calls[idx]
                        fn = tc_delta.get("function", {})
                        if fn.get("name"):
                            tc["function"]["name"] += fn["name"]
                        if fn.get("arguments"):
                            tc["function"]["arguments"] += fn["arguments"]
                        if tc_delta.get("id"):
                            tc["id"] = tc_delta["id"]

                except Exception:
                    pass

    # Build response dict in non-streaming format
    message: dict = {"role": "assistant", "content": accumulated_content or None}
    if accumulated_tool_calls:
        message["tool_calls"] = list(accumulated_tool_calls.values())
        finish_reason = "tool_calls"

    return {
        "choices": [{"message": message, "finish_reason": finish_reason}]
    }


async def _stream_openrouter_final(messages: list[dict]) -> AsyncGenerator[str, None]:
    """Fallback streaming call for final answer synthesis — no tools."""
    headers = {
        "Authorization":  f"Bearer {_API_KEY}",
        "Content-Type":   "application/json",
        **_API_HEADERS_EXTRA,
    }
    body = {
        "model":       MODEL,
        "messages":    messages,
        "stream":      True,
        "temperature": 0.3,
        "max_tokens":  2048,
    }
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream("POST", _API_URL, headers=headers, json=body) as resp:
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
    final_direct_content: str = ""  # holds LLM's last text response

    while iteration < MAX_ITER:
        iteration += 1
        t0 = time.time()

        try:
            response = await _call_openrouter_sync(messages)
        except httpx.TimeoutException:
            raise TimeoutError("OpenRouter request timed out after 30s")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"OpenRouter HTTP {e.response.status_code}: {e.response.text[:200]}")

        elapsed = time.time() - t0
        choice = response.get("choices", [{}])[0]
        finish_reason = choice.get("finish_reason", "stop")
        message = choice.get("message", {})
        tool_calls = message.get("tool_calls") or []
        logger.debug("[LOOP iter=%d] %.2fs finish=%r tools=%s", iteration, elapsed,
                     finish_reason, [tc['function']['name'] for tc in tool_calls])

        # ── Tool call ────────────────────────────────────────────────────
        if finish_reason == "tool_calls" or message.get("tool_calls"):
            tool_calls = message.get("tool_calls") or []

            if iteration >= MAX_ITER:
                # Hit limit — stop; final_direct_content stays empty,
                # fallback _stream_openrouter_final will synthesize
                break

            # Append assistant tool_use message — content must be None for OpenAI compat
            messages.append({"role": "assistant", "tool_calls": tool_calls, "content": None})

            # Emit SSE signals + collect tool names
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                tools_called.append(fn_name)
                pre_events.append(
                    f"event: tool_call\ndata: {json.dumps({'name': fn_name}, ensure_ascii=False)}\n\n"
                )

            # Execute all tools in parallel (faster than sequential)
            async def _exec_tool(tc):
                fn_name = tc["function"]["name"]
                try:
                    fn_args = json.loads(tc["function"]["arguments"])
                    fn = TOOL_DISPATCH.get(fn_name)
                    res = fn(**fn_args) if fn else {"error": "unknown_tool", "name": fn_name}
                except Exception as exc:
                    logger.error("Tool %s failed: %s", fn_name, exc)
                    res = {"error": "tool_execution_error", "details": str(exc)[:200]}
                return tc, res

            results = await asyncio.gather(*[_exec_tool(tc) for tc in tool_calls])
            for tc, res in results:
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc["id"],
                    "name":         tc["function"]["name"],
                    "content":      json.dumps(res, ensure_ascii=False, default=str),
                })

            continue  # next iteration

        # ── Final text answer ────────────────────────────────────────────
        # LLM returned text directly (no tool calls in this iteration)
        final_direct_content = message.get("content") or ""
        break

    # Build the final streaming generator
    async def _final_stream():
        # Emit buffered tool_call events first
        for ev in pre_events:
            yield ev

        logger.debug("[STREAM] has_tools=%s direct_len=%d", bool(pre_events), len(final_direct_content))

        if final_direct_content:
            # Stream the LLM's direct response (already generated — no extra API call needed)
            chunk_size = 16
            for i in range(0, len(final_direct_content), chunk_size):
                yield f"data: {json.dumps({'text': final_direct_content[i:i+chunk_size]}, ensure_ascii=False)}\n\n"
        else:
            # Fallback: ask LLM to synthesize (shouldn't normally happen)
            logger.warning("[STREAM] fallback synthesis — no direct content from loop")
            chunk_count = 0
            async for chunk in _stream_openrouter_final(messages):
                chunk_count += 1
                yield f"data: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"

        yield "event: done\ndata: {}\n\n"

    return tools_called, iteration, _final_stream()


# ── Query Rewrite ─────────────────────────────────────────────────────────────

async def _rewrite_query(message: str, history: list) -> str:
    """
    Rewrite a short/ambiguous follow-up question into a specific question
    with full benchmark context resolved from conversation history.
    
    Only activates when:
    - message is short (< 80 chars) — likely a vague follow-up
    - history has a SCORE RESULT or COMPARISON block — context available
    
    Returns the original message if rewrite fails or is unnecessary.
    """
    if len(message.strip()) >= 80:
        return message  # already specific enough

    # Find the most recent score/comparison context block in history
    context_block = next(
        (m["content"] for m in reversed(history)
         if m.get("role") == "assistant"
         and any(tag in m.get("content", "") for tag in ["[SCORE RESULT", "[COMPARISON"])),
        None
    )
    if not context_block:
        return message  # no context to expand with

    rewrite_prompt = f"""You are a context resolver for an OCR benchmark chat system.

AVAILABLE CONTEXT:
{context_block[:600]}

SHORT QUESTION: "{message}"

Rewrite the short question into ONE complete, specific question that:
- Names the exact model, document, and page number when relevant
- Uses the same language as the question (Vietnamese or English)
- Is self-contained (no pronouns like "it", "that", "the above")
- Stays concise (1 sentence max)

Output ONLY the rewritten question, nothing else:"""

    try:
        headers = {
            "Authorization": f"Bearer {_API_KEY}",
            "Content-Type": "application/json",
            **_API_HEADERS_EXTRA,
        }
        body = {
            "model": MODEL,
            "messages": [{"role": "user", "content": rewrite_prompt}],
            "stream": False,
            "temperature": 0,
            "max_tokens": 120,
        }
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(_API_URL, headers=headers, json=body)
            if resp.status_code == 200:
                rewritten = resp.json()["choices"][0]["message"]["content"].strip()
                # Sanity check: rewrite shouldn't be empty or way longer than needed
                if rewritten and len(rewritten) < 300:
                    logger.info("Query rewrite: %r -> %r", message, rewritten)
                    return rewritten
    except Exception as exc:
        logger.warning("Query rewrite failed (using original): %s", exc)

    return message


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
    if not _ENABLED:
        async def _err():
            yield 'event: error\ndata: {"message": "No LLM API key configured (set OPENAI_API_KEY or OPEN_ROUTER in .env)"}\n\n'
        return StreamingResponse(_err(), media_type="text/event-stream")

    body = await request.json()
    session_id = body.get("session_id", "anon")
    user_message = body.get("message", "").strip()
    history: list[dict] = body.get("history", [])

    if not user_message:
        async def _empty():
            yield 'event: error\ndata: {"message": "Empty message"}\n\n'
        return StreamingResponse(_empty(), media_type="text/event-stream")

    # ── Logging ───────────────────────────────────────────────────────────────
    logger.info("[CHAT] session=%s query=%r history_turns=%d", session_id, user_message[:80], len(history))

    # ── Query rewrite: expand short/ambiguous follow-ups using history context ──
    effective_message = await _rewrite_query(user_message, history)
    if effective_message != user_message:
        logger.info("[REWRITE] %r → %r", user_message[:60], effective_message[:80])

    # Build messages — use last 8 turns (more context = better resolution)
    messages: list[dict] = [{"role": "system", "content": _build_system_prompt()}]
    for turn in history[-8:]:
        if turn.get("role") in ("user", "assistant"):
            messages.append({"role": turn["role"], "content": turn.get("content", "")})
    # Push both original (for history continuity) and rewritten (for agent understanding)
    if effective_message != user_message:
        # Agent sees rewritten; history logs original
        messages.append({"role": "user", "content": effective_message})
    else:
        messages.append({"role": "user", "content": user_message})

    async def _event_stream():
        full_response = ""
        tools_called: list[str] = []
        iterations = 0
        error_occurred = False

        try:
            tools_called, iterations, final_gen = await _run_agentic_loop(messages)
            logger.info("[CHAT] tools=%s iterations=%d session=%s", tools_called, iterations, session_id)

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
            logger.warning("[CHAT] timeout session=%s", session_id)
            yield f'event: error\ndata: {json.dumps({"message": "Request timed out. Please try again."})}\n\n'
            yield "event: done\ndata: {}\n\n"
        except Exception as exc:
            error_occurred = True
            logger.error("[CHAT] error session=%s %s: %s", session_id, type(exc).__name__, exc)
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
