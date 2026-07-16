"""
uet_metrics.py
--------------
Metrics extracted from UET benchmark notebooks:
  - uet/Copy_of_datalab_convert_benchmark_uet.ipynb
  - uet/Copy_of_mistral_ocr4_benchmark_uet.ipynb

Provides:
  compute_text_metrics(gt_text, pred_text) -> dict
  compute_table_metrics_from_html(gt_html_list, pred_html_list) -> dict
  compute_all_metrics(gt_json_page, pred_json_page) -> dict

Our data format:
  GT/Pred JSON page: {
    "page_num": int,
    "full_text": str,            # for scan/text_layer
    "tables": [                  # for table/mixed
      {"table_id": int, "html": str, "cells": [...]}
    ]
  }

Note: table is already HTML — no MD→HTML conversion needed.
"""

from __future__ import annotations

import re
import html
import math
import unicodedata
from typing import Any, Dict, List, Tuple, Optional
from collections import Counter

# ── Dependencies ──────────────────────────────────────────────
try:
    from rapidfuzz.distance import Levenshtein
    _LEV_AVAILABLE = True
except ImportError:
    _LEV_AVAILABLE = False
    import difflib

import numpy as np

try:
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False

try:
    from apted import APTED, Config as APTEDConfig
    _APTED_AVAILABLE = True
except ImportError:
    _APTED_AVAILABLE = False
    # Stub so TEDSConfig class definition doesn't fail at import time
    class APTEDConfig:
        pass
    class APTED:
        def __init__(self, *a, **kw): pass
        def compute_edit_distance(self): return 0

try:
    from scipy.optimize import linear_sum_assignment
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

try:
    from markdown_it import MarkdownIt
    _MDI_AVAILABLE = True
except ImportError:
    _MDI_AVAILABLE = False
    MarkdownIt = None

# ── Global config (mirrors notebook defaults) ────────────────
COLLAPSE_WHITESPACE = True
LOWERCASE_FOR_METRICS = False

# ── Import normalizers from normalize.py (single source of truth) ────────────
from pathlib import Path
from ocr_benchmark.normalize import (
    normalize_ocr_text,
    normalize_ws,
    normalize,
    normalize_cell,
    normalize_latex,
    flatten_markdown_tables_for_text,
    COLLAPSE_WHITESPACE,
    LOWERCASE_FOR_METRICS,
)


def edit_distance(a: str, b: str) -> int:
    return Levenshtein.distance(a or '', b or '')
def normalized_edit_similarity(ref: str, pred: str) -> float:
    ref = ref or ''
    pred = pred or ''
    denom = max(len(ref), len(pred), 1)
    return max(0.0, 1.0 - edit_distance(ref, pred) / denom)


def cer(ref: str, pred: str) -> float:
    ref = ref or ''
    pred = pred or ''
    return edit_distance(ref, pred) / max(len(ref), 1)


def tokenize_words(s: str) -> List[str]:
    return re.findall(r'\w+|[^\w\s]', s, flags=re.UNICODE)


def wer(ref: str, pred: str) -> float:
    r = tokenize_words(ref)
    p = tokenize_words(pred)
    return Levenshtein.distance(r, p) / max(len(r), 1)


def multiset_prf(ref_items: List[Any], pred_items: List[Any]) -> Dict[str, float]:
    ref_c = Counter(ref_items)
    pred_c = Counter(pred_items)
    overlap = sum((ref_c & pred_c).values())
    pred_n = sum(pred_c.values())
    ref_n = sum(ref_c.values())
    precision = overlap / pred_n if pred_n else (1.0 if ref_n == 0 else 0.0)
    recall = overlap / ref_n if ref_n else (1.0 if pred_n == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1}


def char_f1(ref: str, pred: str) -> float:
    return multiset_prf(list(ref), list(pred))['f1']


def word_f1(ref: str, pred: str) -> float:
    return multiset_prf(tokenize_words(ref), tokenize_words(pred))['f1']


def markdown_to_plain_text(md: str) -> str:
    # Render Markdown to HTML, then extract visible text. Fall back to regex stripping if rendering fails.
    try:
        rendered = MD.render(md or '')
        soup = BeautifulSoup(rendered, 'lxml')
        text = soup.get_text('\n')
    except Exception:
        text = md or ''
        text = re.sub(r'!\[([^\]]*)\]\([^)]*\)', r'\1', text)
        text = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', text)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'[`*_>#~-]+', ' ', text)
    return normalize_ws(text)


def extract_blocks_for_order(md: str) -> List[str]:
    # Coarse reading-order metric: sequence of non-empty visible text blocks.
    #text = markdown_to_plain_text(md)
    text = normalize(md)
    blocks = [normalize_ws(x) for x in re.split(r'\n+|(?<=[.!?。！？])\s+', text) if normalize_ws(x)]
    return blocks


def lcs_len(a: List[Any], b: List[Any]) -> int:
    # Memory-efficient LCS for moderate lists.
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j-1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def reading_order_lcs(ref_md: str, pred_md: str) -> float:
    ref_blocks = extract_blocks_for_order(ref_md)
    pred_blocks = extract_blocks_for_order(pred_md)
    if not ref_blocks and not pred_blocks:
        return 1.0
    # Exact block LCS is strict. This is useful as a low-noise order indicator.
    return lcs_len(ref_blocks, pred_blocks) / max(len(ref_blocks), 1)


def markdown_token_paths(md: str, include_text: bool = False) -> List[str]:
    tokens = MD.parse(md or '')
    stack = []
    paths = []

    for tok in tokens:
        t = tok.type
        if tok.nesting == 1:
            stack.append(t)
            paths.append('/'.join(stack))
        elif tok.nesting == -1:
            paths.append('/'.join(stack + [t]))
            if stack:
                stack.pop()
        else:
            path = '/'.join(stack + [t])
            paths.append(path)
            if include_text and tok.content and t in {'inline', 'text', 'code_inline', 'fence', 'code_block'}:
                content = normalize_ws(tok.content)
                if content:
                    paths.append(path + '::TEXT::' + content)
    return paths


def markdown_structure_metrics(ref_md: str, pred_md: str) -> Dict[str, float]:
    ref_struct = markdown_token_paths(ref_md, include_text=False)
    pred_struct = markdown_token_paths(pred_md, include_text=False)
    struct = multiset_prf(ref_struct, pred_struct)

    ref_struct_text = markdown_token_paths(ref_md, include_text=True)
    pred_struct_text = markdown_token_paths(pred_md, include_text=True)
    struct_text = multiset_prf(ref_struct_text, pred_struct_text)

    return {
        'ast_struct_precision': struct['precision'],
        'ast_struct_recall': struct['recall'],
        'ast_struct_f1': struct['f1'],
        'ast_struct_text_precision': struct_text['precision'],
        'ast_struct_text_recall': struct_text['recall'],
        'ast_struct_text_f1': struct_text['f1'],
        'reading_order_lcs': reading_order_lcs(ref_md, pred_md),
    }


def is_pipe_table_separator(line: str) -> bool:
    s = line.strip()
    if '|' not in s:
        return False
    s2 = s.strip('|').strip()
    if not s2:
        return False
    parts = [p.strip() for p in s2.split('|')]
    return all(re.fullmatch(r':?-{3,}:?', p or '') is not None for p in parts)


def looks_like_pipe_row(line: str) -> bool:
    s = line.strip()
    return '|' in s and bool(s.strip('|').strip())


def split_pipe_row(line: str) -> List[str]:
    s = line.strip()
    if s.startswith('|'):
        s = s[1:]
    if s.endswith('|'):
        s = s[:-1]

    cells = []
    cur = []
    escaped = False
    for ch in s:
        if escaped:
            cur.append(ch)
            escaped = False
        elif ch == '\\':
            escaped = True
        elif ch == '|':
            cells.append(''.join(cur))
            cur = []
        else:
            cur.append(ch)
    cells.append(''.join(cur))
    return [normalize_cell(c) for c in cells]


def rectangularize(rows: List[List[str]]) -> List[List[str]]:
    if not rows:
        return []
    width = max(len(r) for r in rows)
    return [r + [''] * (width - len(r)) for r in rows]


def extract_pipe_tables(md: str) -> List[List[List[str]]]:
    lines = (md or '').splitlines()
    tables = []
    i = 0
    while i < len(lines) - 1:
        if looks_like_pipe_row(lines[i]) and is_pipe_table_separator(lines[i + 1]):
            rows = [split_pipe_row(lines[i])]
            i += 2
            while i < len(lines) and looks_like_pipe_row(lines[i]):
                # Stop if the row is likely not table content.
                rows.append(split_pipe_row(lines[i]))
                i += 1
            tables.append(rectangularize(rows))
        else:
            i += 1
    return tables


def extract_html_tables(md: str) -> List[List[List[str]]]:
    soup = BeautifulSoup(md or '', 'lxml')
    out = []
    for table in soup.find_all('table'):
        rows = []
        for tr in table.find_all('tr'):
            cells = []
            for cell in tr.find_all(['th', 'td']):
                # Basic handling: rowspans/colspans are ignored in the grid metric but retained in TEDS if using raw HTML.
                cells.append(normalize_cell(cell.get_text(' ')))
            if cells:
                rows.append(cells)
        if rows:
            out.append(rectangularize(rows))
    return out


def extract_tables(md: str) -> List[List[List[str]]]:
    # If HTML tables exist, include them. Pipe tables are also included.
    return extract_pipe_tables(md) + extract_html_tables(md)


def table_to_html_grid(table: List[List[str]]) -> str:
    rows = []
    for r, row in enumerate(table):
        tag = 'th' if r == 0 else 'td'
        row_html = ''.join(f'<{tag}>{html.escape(str(c))}</{tag}>' for c in row)
        rows.append(f'<tr>{row_html}</tr>')
    return '<table>' + ''.join(rows) + '</table>'


class TreeNode:
    def __init__(self, label: str, children: Optional[List['TreeNode']] = None):
        self.label = label
        self.children = children or []


def soup_node_to_tree(node) -> Optional[TreeNode]:
    if getattr(node, 'name', None) is None:
        text = normalize_cell(str(node))
        if text:
            return TreeNode('text:' + text, [])
        return None

    children = []
    for child in getattr(node, 'children', []):
        t = soup_node_to_tree(child)
        if t is not None:
            children.append(t)

    if node.name in {'td', 'th'}:
        # Store all cell text in the cell node label. This makes content similarity part of rename cost.
        label = f'{node.name}:{normalize_cell(node.get_text(" "))}'
        return TreeNode(label, [])

    return TreeNode(str(node.name), children)


def html_table_to_tree(html_table: str) -> TreeNode:
    import re as _re
    # Strip whitespace between tags to avoid spurious text nodes from formatting
    html_clean = _re.sub(r'>\s+<', '><', html_table.strip())
    soup = BeautifulSoup(html_clean, 'lxml')
    table = soup.find('table')
    if table is None:
        return TreeNode('table', [])
    return soup_node_to_tree(table) or TreeNode('table', [])


def count_tree_nodes(node: TreeNode) -> int:
    return 1 + sum(count_tree_nodes(c) for c in node.children)


class TEDSConfig(APTEDConfig):
    def children(self, node):
        return node.children

    def rename(self, n1, n2):
        l1 = n1.label
        l2 = n2.label
        tag1 = l1.split(':', 1)[0]
        tag2 = l2.split(':', 1)[0]
        if tag1 != tag2:
            return 1.0
        if tag1 in {'td', 'th', 'text'}:
            text1 = l1.split(':', 1)[1] if ':' in l1 else ''
            text2 = l2.split(':', 1)[1] if ':' in l2 else ''
            return 1.0 - normalized_edit_similarity(text1, text2)
        return 0.0

    def insert(self, node):
        return 1.0

    def delete(self, node):
        return 1.0


def teds_similarity_table(ref_table: List[List[str]], pred_table: List[List[str]]) -> float:
    ref_tree = html_table_to_tree(table_to_html_grid(ref_table))
    pred_tree = html_table_to_tree(table_to_html_grid(pred_table))
    dist = APTED(ref_tree, pred_tree, TEDSConfig()).compute_edit_distance()
    denom = max(count_tree_nodes(ref_tree), count_tree_nodes(pred_tree), 1)
    return max(0.0, 1.0 - dist / denom)


def cell_exact_f1_aligned(ref_table: List[List[str]], pred_table: List[List[str]]) -> float:
    ref_items = []
    pred_items = []
    for r, row in enumerate(ref_table):
        for c, val in enumerate(row):
            ref_items.append((r, c, normalize_cell(val)))
    for r, row in enumerate(pred_table):
        for c, val in enumerate(row):
            pred_items.append((r, c, normalize_cell(val)))
    return multiset_prf(ref_items, pred_items)['f1']


def avg_cell_text_similarity_aligned(ref_table: List[List[str]], pred_table: List[List[str]]) -> float:
    ref_rows = len(ref_table)
    pred_rows = len(pred_table)
    ref_cols = max([len(r) for r in ref_table], default=0)
    pred_cols = max([len(r) for r in pred_table], default=0)
    rows = max(ref_rows, pred_rows)
    cols = max(ref_cols, pred_cols)
    if rows == 0 or cols == 0:
        return 1.0 if rows == 0 and cols == 0 else 0.0

    sims = []
    for r in range(rows):
        for c in range(cols):
            rv = ref_table[r][c] if r < ref_rows and c < len(ref_table[r]) else ''
            pv = pred_table[r][c] if r < pred_rows and c < len(pred_table[r]) else ''
            sims.append(normalized_edit_similarity(normalize_cell(rv), normalize_cell(pv)))
    return float(np.mean(sims)) if sims else 1.0


def table_shape_similarity(ref_table: List[List[str]], pred_table: List[List[str]]) -> Dict[str, float]:
    rr = len(ref_table)
    pr = len(pred_table)
    rc = max([len(r) for r in ref_table], default=0)
    pc = max([len(r) for r in pred_table], default=0)
    row_sim = 1.0 - abs(rr - pr) / max(rr, pr, 1)
    col_sim = 1.0 - abs(rc - pc) / max(rc, pc, 1)
    return {'row_count_similarity': row_sim, 'col_count_similarity': col_sim}


def match_tables(ref_tables: List[List[List[str]]], pred_tables: List[List[List[str]]]) -> Tuple[List[Tuple[int, int, float]], float]:
    if not ref_tables and not pred_tables:
        return [], 1.0
    if not ref_tables or not pred_tables:
        return [], 0.0

    sim = np.zeros((len(ref_tables), len(pred_tables)), dtype=float)
    for i, rt in enumerate(ref_tables):
        for j, pt in enumerate(pred_tables):
            sim[i, j] = teds_similarity_table(rt, pt)

    row_ind, col_ind = linear_sum_assignment(1.0 - sim)
    matches = [(int(i), int(j), float(sim[i, j])) for i, j in zip(row_ind, col_ind)]
    # Penalize missing or extra tables by normalizing over max count.
    corpus_like_score = sum(s for _, _, s in matches) / max(len(ref_tables), len(pred_tables), 1)
    return matches, corpus_like_score


def table_metrics(ref_md: str, pred_md: str) -> Dict[str, float]:
    ref_tables = extract_tables(ref_md)
    pred_tables = extract_tables(pred_md)
    matches, teds_doc = match_tables(ref_tables, pred_tables)

    matched_teds = []
    matched_cell_f1 = []
    matched_cell_sim = []
    matched_row_sim = []
    matched_col_sim = []

    for i, j, sim in matches:
        rt = ref_tables[i]
        pt = pred_tables[j]
        matched_teds.append(sim)
        matched_cell_f1.append(cell_exact_f1_aligned(rt, pt))
        matched_cell_sim.append(avg_cell_text_similarity_aligned(rt, pt))
        shape = table_shape_similarity(rt, pt)
        matched_row_sim.append(shape['row_count_similarity'])
        matched_col_sim.append(shape['col_count_similarity'])

    table_count_precision = min(len(ref_tables), len(pred_tables)) / max(len(pred_tables), 1) if pred_tables else (1.0 if not ref_tables else 0.0)
    table_count_recall = min(len(ref_tables), len(pred_tables)) / max(len(ref_tables), 1) if ref_tables else (1.0 if not pred_tables else 0.0)
    table_count_f1 = (2 * table_count_precision * table_count_recall / (table_count_precision + table_count_recall)
                      if table_count_precision + table_count_recall else 0.0)

    return {
        'ref_table_count': len(ref_tables),
        'pred_table_count': len(pred_tables),
        'table_count_precision': table_count_precision,
        'table_count_recall': table_count_recall,
        'table_count_f1': table_count_f1,
        'table_teds_doc': teds_doc,
        'table_teds_matched_mean': float(np.mean(matched_teds)) if matched_teds else (1.0 if not ref_tables and not pred_tables else 0.0),
        'table_cell_exact_f1_mean': float(np.mean(matched_cell_f1)) if matched_cell_f1 else (1.0 if not ref_tables and not pred_tables else 0.0),
        'table_cell_text_similarity_mean': float(np.mean(matched_cell_sim)) if matched_cell_sim else (1.0 if not ref_tables and not pred_tables else 0.0),
        'table_row_count_similarity_mean': float(np.mean(matched_row_sim)) if matched_row_sim else (1.0 if not ref_tables and not pred_tables else 0.0),
        'table_col_count_similarity_mean': float(np.mean(matched_col_sim)) if matched_col_sim else (1.0 if not ref_tables and not pred_tables else 0.0),
    }


FORMULA_PATTERNS = [
    re.compile(r'\$\$(.+?)\$\$', re.DOTALL),
    re.compile(r'(?<!\\)\$(?!\$)(.+?)(?<!\\)\$', re.DOTALL),
    re.compile(r'\\\[(.+?)\\\]', re.DOTALL),
    re.compile(r'\\\((.+?)\\\)', re.DOTALL),
]


def extract_formulas(md: str) -> List[str]:
    formulas = []
    text = md or ''
    for pat in FORMULA_PATTERNS:
        formulas.extend([m.group(1) for m in pat.finditer(text)])
    return [normalize_latex(x) for x in formulas if normalize_latex(x)]


def formula_metrics(ref_md: str, pred_md: str) -> Dict[str, float]:
    ref = extract_formulas(ref_md)
    pred = extract_formulas(pred_md)
    exact = multiset_prf(ref, pred)

    if not ref and not pred:
        mean_best_sim = 1.0
    elif not ref or not pred:
        mean_best_sim = 0.0
    else:
        sims = []
        for rf in ref:
            sims.append(max(normalized_edit_similarity(rf, pf) for pf in pred))
        mean_best_sim = float(np.mean(sims))

    return {
        'ref_formula_count': len(ref),
        'pred_formula_count': len(pred),
        'formula_exact_precision': exact['precision'],
        'formula_exact_recall': exact['recall'],
        'formula_exact_f1': exact['f1'],
        'formula_best_edit_similarity_mean': mean_best_sim,
    }


# ── compute_metrics_for_pair (from notebook cell 22) ────────
def compute_metrics_for_pair(ref_md: str, pred_md: str) -> Dict[str, float]:
    """
    Compute all metrics for a (ref, pred) pair given as Markdown strings.
    This is the original UET function — works on raw Markdown.
    """
    ref_text = normalize_ocr_text(ref_md)
    pred_text = normalize_ocr_text(pred_md)

    out = {
        'ref_text_len': len(ref_text),
        'pred_text_len': len(pred_text),
        'cer': cer(ref_text, pred_text),
        'wer': wer(ref_text, pred_text),
        'normalized_edit_similarity': normalized_edit_similarity(ref_text, pred_text),
        'char_f1': char_f1(ref_text, pred_text),
        'word_f1': word_f1(ref_text, pred_text),
    }
    out.update(markdown_structure_metrics(ref_md, pred_md))
    out.update(table_metrics(ref_md, pred_md))
    out.update(formula_metrics(ref_md, pred_md))
    return out


# ── Adapters for our JSON format ─────────────────────────────

def compute_text_metrics(gt_text: str, pred_text: str) -> Dict[str, float]:
    """
    Text-only metrics (CER, WER, etc.) for scan/text_layer UC.
    Inputs are plain text strings (full_text from our GT/pred JSON).
    """
    ref_norm  = normalize_ocr_text(gt_text)
    pred_norm = normalize_ocr_text(pred_text)
    return {
        'cer': cer(ref_norm, pred_norm),
        'wer': wer(ref_norm, pred_norm),
        'normalized_edit_similarity': normalized_edit_similarity(ref_norm, pred_norm),
        'char_f1': char_f1(ref_norm, pred_norm),
        'word_f1': word_f1(ref_norm, pred_norm),
    }


def compute_table_metrics_from_html(
    gt_html_list: List[str],
    pred_html_list: List[str],
) -> Dict[str, float]:
    """
    Table metrics for UC04-06.
    Inputs: list of HTML table strings (already in HTML format — no MD conversion needed).

    Strategy: build a pseudo-Markdown string by joining tables with blank lines.
    extract_html_tables() in the UET code handles HTML tables directly.
    """
    def _html_tables_to_pseudo_md(html_list: List[str]) -> str:
        return "\n\n".join(h for h in html_list if h and h.strip())

    ref_md  = _html_tables_to_pseudo_md(gt_html_list)
    pred_md = _html_tables_to_pseudo_md(pred_html_list)
    return table_metrics(ref_md, pred_md)


def compute_all_metrics(
    gt_page: dict,
    pred_page: dict,
) -> Dict[str, float]:
    """
    Compute all relevant metrics for a single page pair.

    gt_page / pred_page: {
        "page_num": int,
        "full_text": str,          # optional
        "tables": [                # optional
            {"table_id": int, "html": str, ...}
        ]
    }
    """
    result: Dict[str, float] = {'page_num': gt_page.get('page_num', 1)}

    # Text metrics
    gt_text   = gt_page.get('full_text', '') or ''
    pred_text = pred_page.get('full_text', '') or ''
    if gt_text or pred_text:
        result.update(compute_text_metrics(gt_text, pred_text))

    # Table metrics
    gt_tables   = [t['html'] for t in (gt_page.get('tables')   or []) if t.get('html')]
    pred_tables = [t['html'] for t in (pred_page.get('tables') or []) if t.get('html')]
    if gt_tables or pred_tables:
        tbl = compute_table_metrics_from_html(gt_tables, pred_tables)
        result.update(tbl)

    return result
