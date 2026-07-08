import { useState, useEffect, useRef } from 'react'
import { Upload, Zap, FileText, ChevronDown } from 'lucide-react'
import { ScoreCard } from './components/ScoreCard'
import { DiffViewer } from './components/DiffViewer'
import { PdfViewer } from './components/PdfViewer'
import type { Model, UcOption, EvalResult } from './types'

export default function App() {
  const [models, setModels] = useState<Model[]>([])
  const [ucOptions, setUcOptions] = useState<UcOption[]>([])
  const [selectedModel, setSelectedModel] = useState('')
  const [selectedUc, setSelectedUc] = useState<UcOption | null>(null)
  const [gtFile, setGtFile] = useState<File | null>(null)
  const [predFile, setPredFile] = useState<File | null>(null)
  const [pdfFile, setPdfFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<EvalResult | null>(null)
  const [error, setError] = useState('')
  const [currentPage, setCurrentPage] = useState(1)

  useEffect(() => {
    fetch('/api/models').then(r => r.json()).then(setModels)
    fetch('/api/uc-options').then(r => r.json()).then((opts: UcOption[]) => {
      setUcOptions(opts)
      setSelectedUc(opts[0])
    })
  }, [])

  const currentPageResult = result?.pages.find(p => p.page_num === currentPage)

  async function handleEvaluate() {
    if (!gtFile || !predFile || !selectedModel || !selectedUc) return
    setLoading(true)
    setError('')
    setResult(null)

    const fd = new FormData()
    fd.append('gt_json', gtFile)
    fd.append('prediction_text', predFile)
    fd.append('model_id', selectedModel)
    fd.append('uc_type', selectedUc.type)
    fd.append('include_alignment', 'true')

    try {
      const res = await fetch('/api/evaluate', { method: 'POST', body: fd })
      if (!res.ok) {
        const msg = await res.json()
        throw new Error(msg.detail || 'Evaluation failed')
      }
      const data: EvalResult = await res.json()
      setResult(data)
      setCurrentPage(1)
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  // Group models by tier
  const tiers = [...new Set(models.map(m => m.tier))]

  return (
    <div className="min-h-screen bg-[#F8FAFC]">
      {/* Header */}
      <header className="bg-white border-b border-blue-100 sticky top-0 z-10 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-blue-800 flex items-center justify-center">
            <Zap size={16} className="text-white" />
          </div>
          <div>
            <h1 className="font-semibold text-blue-900 text-sm leading-none">OCR Benchmark</h1>
            <p className="text-xs text-slate-400 mt-0.5">Evaluate & visualize OCR quality</p>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-6 space-y-6">
        {/* Config panel */}
        <section className="bg-white rounded-xl border border-blue-100 p-5 shadow-sm">
          <h2 className="font-semibold text-blue-900 mb-4 text-sm uppercase tracking-wide">Configuration</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">

            {/* Model */}
            <div>
              <label className="text-xs font-medium text-slate-600 block mb-1.5">OCR Model</label>
              <div className="relative">
                <select
                  value={selectedModel}
                  onChange={e => setSelectedModel(e.target.value)}
                  className="w-full appearance-none bg-blue-50 border border-blue-200 rounded-lg px-3 py-2 text-sm text-blue-900 cursor-pointer focus:outline-none focus:ring-2 focus:ring-blue-400"
                >
                  <option value="">Select model…</option>
                  {tiers.map(tier => (
                    <optgroup key={tier} label={tier}>
                      {models.filter(m => m.tier === tier).map(m => (
                        <option key={m.id} value={m.id}>{m.name}</option>
                      ))}
                    </optgroup>
                  ))}
                </select>
                <ChevronDown size={14} className="absolute right-3 top-3 text-blue-500 pointer-events-none" />
              </div>
            </div>

            {/* Use case */}
            <div>
              <label className="text-xs font-medium text-slate-600 block mb-1.5">Use Case</label>
              <div className="relative">
                <select
                  value={selectedUc?.id ?? ''}
                  onChange={e => setSelectedUc(ucOptions.find(u => u.id === e.target.value) ?? null)}
                  className="w-full appearance-none bg-blue-50 border border-blue-200 rounded-lg px-3 py-2 text-sm text-blue-900 cursor-pointer focus:outline-none focus:ring-2 focus:ring-blue-400"
                >
                  {ucOptions.map(u => (
                    <option key={u.id} value={u.id}>{u.label}</option>
                  ))}
                </select>
                <ChevronDown size={14} className="absolute right-3 top-3 text-blue-500 pointer-events-none" />
              </div>
            </div>

            {/* GT JSON */}
            <FileDropzone
              label="Ground Truth JSON"
              accept=".json"
              file={gtFile}
              onFile={setGtFile}
            />

            {/* Prediction TXT */}
            <FileDropzone
              label="Prediction Text"
              accept=".txt,.json"
              file={predFile}
              onFile={setPredFile}
            />
          </div>

          {/* PDF (optional) */}
          <div className="mt-4">
            <FileDropzone
              label="PDF (optional — for visual diff overlay)"
              accept=".pdf"
              file={pdfFile}
              onFile={setPdfFile}
              wide
            />
          </div>

          <button
            onClick={handleEvaluate}
            disabled={loading || !gtFile || !predFile || !selectedModel}
            className="mt-4 px-6 py-2.5 bg-blue-800 text-white text-sm font-semibold rounded-lg
                       hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed
                       flex items-center gap-2 transition-colors cursor-pointer"
          >
            {loading ? (
              <>
                <span className="animate-spin w-4 h-4 border-2 border-white border-t-transparent rounded-full" />
                Evaluating…
              </>
            ) : (
              <><Zap size={15} /> Run Evaluation</>
            )}
          </button>

          {error && (
            <p className="mt-3 text-sm text-red-600 bg-red-50 rounded-lg px-3 py-2">{error}</p>
          )}
        </section>

        {/* Results */}
        {result && (
          <>
            {/* Summary score card */}
            <ScoreCard summary={result.summary} />

            {/* Per-page view */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Left: PDF viewer */}
              <section className="bg-white rounded-xl border border-blue-100 p-5 shadow-sm">
                <h2 className="font-semibold text-blue-900 mb-4 text-sm uppercase tracking-wide flex items-center gap-2">
                  <FileText size={14} /> PDF Viewer
                </h2>
                <PdfViewer
                  file={pdfFile}
                  alignments={currentPageResult?.char_alignment ?? []}
                  currentPage={currentPage}
                  totalPages={result.pages.length}
                  onPageChange={setCurrentPage}
                />
              </section>

              {/* Right: char diff */}
              <section className="bg-white rounded-xl border border-blue-100 p-5 shadow-sm">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="font-semibold text-blue-900 text-sm uppercase tracking-wide">
                    Character Diff — Page {currentPage}
                  </h2>
                  {/* Page selector */}
                  <div className="flex gap-1">
                    {result.pages.map(p => (
                      <button
                        key={p.page_num}
                        onClick={() => setCurrentPage(p.page_num)}
                        className={`w-7 h-7 rounded text-xs font-mono cursor-pointer transition-colors
                          ${currentPage === p.page_num
                            ? 'bg-blue-800 text-white'
                            : 'bg-blue-50 text-blue-700 hover:bg-blue-100'
                          }`}
                      >
                        {p.page_num}
                      </button>
                    ))}
                  </div>
                </div>

                {currentPageResult ? (
                  <DiffViewer page={currentPageResult} />
                ) : (
                  <p className="text-slate-400 text-sm italic">Select a page</p>
                )}
              </section>
            </div>
          </>
        )}
      </main>
    </div>
  )
}

// ---- File dropzone component ----
function FileDropzone({
  label, accept, file, onFile, wide,
}: {
  label: string
  accept: string
  file: File | null
  onFile: (f: File) => void
  wide?: boolean
}) {
  const ref = useRef<HTMLInputElement>(null)

  return (
    <div className={wide ? 'sm:col-span-2 lg:col-span-4' : ''}>
      <label className="text-xs font-medium text-slate-600 block mb-1.5">{label}</label>
      <div
        onClick={() => ref.current?.click()}
        className="flex items-center gap-2 border-2 border-dashed border-blue-200 rounded-lg px-3 py-2.5
                   cursor-pointer hover:border-blue-400 hover:bg-blue-50/50 transition-colors min-h-[42px]"
      >
        <Upload size={14} className="text-blue-400 shrink-0" />
        <span className="text-sm truncate text-slate-500">
          {file ? file.name : <span className="text-blue-400">Click to upload</span>}
        </span>
        <input
          ref={ref}
          type="file"
          accept={accept}
          className="hidden"
          onChange={e => e.target.files?.[0] && onFile(e.target.files[0])}
        />
      </div>
    </div>
  )
}
