import { useState, useEffect, useRef } from 'react'
import { Upload, Zap, ChevronDown, AlertCircle, CheckCircle2, Loader2 } from 'lucide-react'

interface GTDoc {
  doc_id: string
  uc_type: string
  lang: string
}

interface PageMetrics {
  page_num: number
  cer?: number | null
  wer?: number | null
  char_f1?: number | null
  word_f1?: number | null
  normalized_edit_similarity?: number | null
  table_teds_doc?: number | null
  table_cell_exact_f1_mean?: number | null
  error?: string
}

interface ScoreReport {
  model: string
  doc_id: string
  uc_type: string
  lang: string
  error?: string
  results: {
    text: {
      summary: {
        n_pages: number
        n_matched_pages: number
        cer?: number | null
        wer?: number | null
        char_f1?: number | null
        word_f1?: number | null
        normalized_edit_similarity?: number | null
        table_teds_doc?: number | null
        table_cell_exact_f1_mean?: number | null
      }
      pages: PageMetrics[]
    }
  }
}

// Color-code CER/WER: <0.05 green, 0.05-0.15 orange, >0.15 red
function metricColor(value: number | null | undefined, lowerBetter = true): string {
  if (value == null) return 'text-slate-400'
  if (lowerBetter) {
    if (value < 0.05) return 'text-green-600'
    if (value <= 0.15) return 'text-orange-500'
    return 'text-red-600'
  } else {
    if (value > 0.9) return 'text-green-600'
    if (value > 0.7) return 'text-orange-500'
    return 'text-red-600'
  }
}

function fmt(v: number | null | undefined, pct = false): string {
  if (v == null) return '—'
  const n = pct ? v * 100 : v
  return pct ? `${n.toFixed(1)}%` : n.toFixed(4)
}

// Group docs by uc_type/lang
function groupDocs(docs: GTDoc[]): Record<string, GTDoc[]> {
  const groups: Record<string, GTDoc[]> = {}
  for (const d of docs) {
    const key = `${d.uc_type} / ${d.lang}`
    if (!groups[key]) groups[key] = []
    groups[key].push(d)
  }
  return groups
}

export function UploadScore() {
  const [gtDocs, setGtDocs] = useState<GTDoc[]>([])
  const [docsError, setDocsError] = useState('')
  const [selectedDocId, setSelectedDocId] = useState('')
  const [modelName, setModelName] = useState('')
  const [files, setFiles] = useState<FileList | null>(null)
  const [scoring, setScoring] = useState(false)
  const [report, setReport] = useState<ScoreReport | null>(null)
  const [error, setError] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Load GT docs on mount
  useEffect(() => {
    fetch('/api/upload/gt_docs')
      .then(r => r.json())
      .then((docs: GTDoc[]) => {
        setGtDocs(docs)
        if (docs.length > 0) setSelectedDocId(docs[0].doc_id)
      })
      .catch(() => setDocsError('Could not load document list from server.'))
  }, [])

  const groups = groupDocs(gtDocs)
  const canSubmit = !scoring && !!modelName.trim() && !!selectedDocId && !!files && files.length > 0

  async function handleScore() {
    if (!canSubmit || !files) return
    setScoring(true)
    setError('')
    setReport(null)

    const fd = new FormData()
    fd.append('model_name', modelName.trim())
    fd.append('doc_id', selectedDocId)
    for (const f of Array.from(files)) {
      fd.append('files', f)
    }

    try {
      const res = await fetch('/api/upload/score', { method: 'POST', body: fd })
      const data = await res.json()
      if (!res.ok) {
        throw new Error(data.detail || `HTTP ${res.status}`)
      }
      setReport(data as ScoreReport)
    } catch (e: any) {
      setError(e.message || 'Scoring failed')
    } finally {
      setScoring(false)
    }
  }

  const summary = report?.results?.text?.summary
  const pages = report?.results?.text?.pages ?? []

  return (
    <div className="space-y-6">
      {/* Form card */}
      <section className="bg-white rounded-xl border border-blue-100 p-5 shadow-sm">
        <h2 className="font-semibold text-blue-900 mb-4 text-sm uppercase tracking-wide flex items-center gap-2">
          <Upload size={14} /> Upload & Score
        </h2>

        {docsError && (
          <div className="mb-4 flex items-center gap-2 text-sm text-red-600 bg-red-50 rounded-lg px-3 py-2">
            <AlertCircle size={14} /> {docsError}
          </div>
        )}

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {/* Model name */}
          <div>
            <label className="text-xs font-medium text-slate-600 block mb-1.5">Model Name</label>
            <input
              type="text"
              maxLength={100}
              placeholder="e.g. PaddleOCR-v4, MyModel-2025…"
              value={modelName}
              onChange={e => setModelName(e.target.value)}
              className="w-full bg-blue-50 border border-blue-200 rounded-lg px-3 py-2 text-sm text-blue-900
                         focus:outline-none focus:ring-2 focus:ring-blue-400 placeholder:text-slate-400"
            />
          </div>

          {/* Document dropdown */}
          <div>
            <label className="text-xs font-medium text-slate-600 block mb-1.5">Document (with GT)</label>
            <div className="relative">
              <select
                value={selectedDocId}
                onChange={e => setSelectedDocId(e.target.value)}
                disabled={gtDocs.length === 0}
                className="w-full appearance-none bg-blue-50 border border-blue-200 rounded-lg px-3 py-2 text-sm
                           text-blue-900 cursor-pointer focus:outline-none focus:ring-2 focus:ring-blue-400
                           disabled:opacity-50"
              >
                {gtDocs.length === 0 && <option value="">Loading…</option>}
                {Object.entries(groups).map(([group, docs]) => (
                  <optgroup key={group} label={group}>
                    {docs.map(d => (
                      <option key={d.doc_id} value={d.doc_id}>{d.doc_id}</option>
                    ))}
                  </optgroup>
                ))}
              </select>
              <ChevronDown size={14} className="absolute right-3 top-3 text-blue-500 pointer-events-none" />
            </div>
          </div>

          {/* Folder picker */}
          <div>
            <label className="text-xs font-medium text-slate-600 block mb-1.5">
              Prediction Folder (.md files)
            </label>
            <div
              onClick={() => fileInputRef.current?.click()}
              className="flex items-center gap-2 border-2 border-dashed border-blue-200 rounded-lg px-3 py-2
                         cursor-pointer hover:border-blue-400 hover:bg-blue-50/50 transition-colors min-h-[42px]"
            >
              <Upload size={14} className="text-blue-400 shrink-0" />
              <span className="text-sm truncate text-slate-500">
                {files && files.length > 0
                  ? <span className="text-blue-700 font-medium">{files.length} file{files.length > 1 ? 's' : ''} selected</span>
                  : <span className="text-blue-400">Click to select folder</span>
                }
              </span>
              <input
                ref={fileInputRef}
                type="file"
                // @ts-ignore — webkitdirectory is non-standard but widely supported
                webkitdirectory=""
                multiple
                className="hidden"
                onChange={e => setFiles(e.target.files)}
              />
            </div>
            {files && files.length > 0 && (
              <p className="text-xs text-slate-400 mt-1">
                {Array.from(files).filter(f => f.name.endsWith('.md')).length} .md files detected
              </p>
            )}
          </div>
        </div>

        <button
          onClick={handleScore}
          disabled={!canSubmit}
          className="mt-4 px-6 py-2.5 bg-blue-800 text-white text-sm font-semibold rounded-lg
                     hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed
                     flex items-center gap-2 transition-colors cursor-pointer"
        >
          {scoring
            ? <><Loader2 size={15} className="animate-spin" /> Scoring…</>
            : <><Zap size={15} /> Score</>
          }
        </button>

        {error && (
          <div className="mt-3 flex items-start gap-2 text-sm text-red-600 bg-red-50 rounded-lg px-3 py-2">
            <AlertCircle size={14} className="shrink-0 mt-0.5" />
            <span>{error}</span>
          </div>
        )}
      </section>

      {/* Results */}
      {report && (
        <>
          {/* Header */}
          <div className="flex items-center gap-3">
            <CheckCircle2 size={18} className="text-green-600" />
            <div>
              <span className="font-semibold text-blue-900">{report.model}</span>
              <span className="text-slate-500 text-sm ml-2">→ {report.doc_id}</span>
              <span className="text-slate-400 text-xs ml-2">({report.uc_type} / {report.lang})</span>
            </div>
          </div>

          {report.error && (
            <div className="text-sm text-orange-700 bg-orange-50 rounded-lg px-3 py-2 border border-orange-200">
              ⚠️ {report.error}
            </div>
          )}

          {/* Summary metrics */}
          {summary && (
            <section className="bg-white rounded-xl border border-blue-100 shadow-sm overflow-hidden">
              <div className="px-5 py-3 bg-blue-50 border-b border-blue-100">
                <h3 className="text-xs font-semibold text-blue-900 uppercase tracking-wide">
                  Document Summary — {summary.n_matched_pages}/{summary.n_pages} pages matched
                </h3>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 divide-x divide-y divide-blue-50">
                {[
                  { label: 'CER', val: summary.cer, pct: true, lowerBetter: true },
                  { label: 'WER', val: summary.wer, pct: true, lowerBetter: true },
                  { label: 'Char F1', val: summary.char_f1, pct: true, lowerBetter: false },
                  { label: 'Word F1', val: summary.word_f1, pct: true, lowerBetter: false },
                  { label: 'Edit Sim', val: summary.normalized_edit_similarity, pct: true, lowerBetter: false },
                  { label: 'TEDS', val: summary.table_teds_doc, pct: true, lowerBetter: false },
                  { label: 'Cell F1', val: summary.table_cell_exact_f1_mean, pct: true, lowerBetter: false },
                ].map(({ label, val, pct, lowerBetter }) => (
                  <div key={label} className="px-4 py-4 text-center">
                    <div className="text-xs font-medium text-slate-500 mb-1">{label}</div>
                    <div className={`text-xl font-bold ${metricColor(val, lowerBetter)}`}>
                      {fmt(val, pct)}
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* Per-page table */}
          {pages.length > 0 && (
            <section className="bg-white rounded-xl border border-blue-100 shadow-sm overflow-hidden">
              <div className="px-5 py-3 bg-blue-50 border-b border-blue-100">
                <h3 className="text-xs font-semibold text-blue-900 uppercase tracking-wide">Per-Page Breakdown</h3>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-blue-100 bg-slate-50">
                      <th className="px-4 py-2 text-left text-xs font-semibold text-slate-500">Page</th>
                      <th className="px-4 py-2 text-right text-xs font-semibold text-slate-500">CER</th>
                      <th className="px-4 py-2 text-right text-xs font-semibold text-slate-500">WER</th>
                      <th className="px-4 py-2 text-right text-xs font-semibold text-slate-500">Char F1</th>
                      <th className="px-4 py-2 text-right text-xs font-semibold text-slate-500">Word F1</th>
                      <th className="px-4 py-2 text-right text-xs font-semibold text-slate-500">Edit Sim</th>
                      <th className="px-4 py-2 text-right text-xs font-semibold text-slate-500">TEDS</th>
                    </tr>
                  </thead>
                  <tbody>
                    {pages.map(p => (
                      <tr key={p.page_num} className="border-b border-slate-50 hover:bg-blue-50/30 transition-colors">
                        <td className="px-4 py-2 font-medium text-blue-900">
                          {p.page_num}
                          {p.error && <span className="ml-2 text-xs text-orange-500" title={p.error}>⚠️</span>}
                        </td>
                        <td className={`px-4 py-2 text-right font-mono text-xs ${metricColor(p.cer, true)}`}>
                          {fmt(p.cer, true)}
                        </td>
                        <td className={`px-4 py-2 text-right font-mono text-xs ${metricColor(p.wer, true)}`}>
                          {fmt(p.wer, true)}
                        </td>
                        <td className={`px-4 py-2 text-right font-mono text-xs ${metricColor(p.char_f1, false)}`}>
                          {fmt(p.char_f1, true)}
                        </td>
                        <td className={`px-4 py-2 text-right font-mono text-xs ${metricColor(p.word_f1, false)}`}>
                          {fmt(p.word_f1, true)}
                        </td>
                        <td className={`px-4 py-2 text-right font-mono text-xs ${metricColor(p.normalized_edit_similarity, false)}`}>
                          {fmt(p.normalized_edit_similarity, true)}
                        </td>
                        <td className={`px-4 py-2 text-right font-mono text-xs ${metricColor(p.table_teds_doc, false)}`}>
                          {fmt(p.table_teds_doc, true)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}
        </>
      )}
    </div>
  )
}
