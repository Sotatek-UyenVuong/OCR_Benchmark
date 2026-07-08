import { useState } from 'react'
import { Document, Page, pdfjs } from 'react-pdf'
import 'react-pdf/dist/Page/AnnotationLayer.css'
import 'react-pdf/dist/Page/TextLayer.css'
import type { CharAlignment } from '../types'

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url,
).toString()

interface Props {
  file: File | null
  alignments?: CharAlignment[]   // for current page
  currentPage: number
  totalPages: number
  onPageChange: (p: number) => void
}

// Map alignment type → CSS class
const CLS: Record<string, string> = {
  substitution: 'diff-sub',
  deletion:     'diff-del',
  insertion:    'diff-ins',
  match:        '',
}

export function PdfViewer({ file, alignments, currentPage, totalPages, onPageChange }: Props) {
  const [numPages, setNumPages] = useState<number>(0)

  if (!file) {
    return (
      <div className="flex items-center justify-center h-64 rounded-xl border-2 border-dashed border-blue-200 text-slate-400">
        <p className="text-sm">PDF preview will appear here</p>
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {/* Navigation */}
      <div className="flex items-center justify-between">
        <span className="text-xs text-slate-500">
          Page {currentPage} / {totalPages || numPages}
        </span>
        <div className="flex gap-2">
          <button
            onClick={() => onPageChange(Math.max(1, currentPage - 1))}
            disabled={currentPage <= 1}
            className="px-3 py-1 rounded text-sm bg-blue-50 text-blue-700 disabled:opacity-40 hover:bg-blue-100 cursor-pointer"
          >
            ← Prev
          </button>
          <button
            onClick={() => onPageChange(Math.min(numPages, currentPage + 1))}
            disabled={currentPage >= numPages}
            className="px-3 py-1 rounded text-sm bg-blue-50 text-blue-700 disabled:opacity-40 hover:bg-blue-100 cursor-pointer"
          >
            Next →
          </button>
        </div>
      </div>

      {/* PDF render */}
      <div className="relative overflow-auto border rounded-xl bg-white shadow-sm">
        <Document
          file={file}
          onLoadSuccess={({ numPages }) => setNumPages(numPages)}
          className="flex justify-center py-4"
        >
          <Page
            pageNumber={currentPage}
            width={520}
            renderTextLayer={true}
            renderAnnotationLayer={false}
          />
        </Document>

        {/* Overlay: character diff annotations */}
        {alignments && alignments.length > 0 && (
          <div
            className="absolute inset-0 pointer-events-none"
            style={{ fontFamily: 'Fira Code, monospace', fontSize: 13 }}
          >
            {/* We render a transparent text ribbon at bottom showing diff inline */}
            <div className="absolute bottom-4 left-4 right-4 bg-white/90 border rounded-lg p-3 text-xs font-mono leading-relaxed shadow max-h-32 overflow-y-auto">
              <p className="text-slate-400 mb-1 font-sans text-[10px] uppercase tracking-wide">
                Char-level diff (this page)
              </p>
              {alignments.filter(a => a.type !== 'match').map((a, i) => (
                <span key={i} className={`${CLS[a.type]} mr-px`}>
                  {a.type === 'deletion' ? a.gt : a.pred}
                </span>
              ))}
              {alignments.filter(a => a.type !== 'match').length === 0 && (
                <span className="text-green-600">✓ Perfect match</span>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Color legend */}
      <div className="flex gap-4 text-xs text-slate-500 px-1">
        <span><span className="diff-sub px-1">S</span> Substitution</span>
        <span><span className="diff-del px-1">D</span> Deletion</span>
        <span><span className="diff-ins px-1">I</span> Insertion</span>
      </div>
    </div>
  )
}
