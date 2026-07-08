import type { PageResult } from '../types'

interface Props { page: PageResult }

const COLOR: Record<string, string> = {
  substitution: 'bg-red-100 border-b-2 border-red-500 text-red-800',
  deletion:     'bg-yellow-100 border-b-2 border-yellow-500 text-yellow-800',
  insertion:    'bg-blue-100 border-b-2 border-blue-500 text-blue-800',
  match:        '',
}

const LABEL: Record<string, string> = {
  substitution: 'S',
  deletion: 'D',
  insertion: 'I',
}

export function DiffViewer({ page }: Props) {
  const { char_alignment, cer_detail } = page
  if (!char_alignment || char_alignment.length === 0) {
    return (
      <div className="text-sm text-slate-400 italic p-4">
        No character alignment data (set include_alignment=true)
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Stats row */}
      {cer_detail && (
        <div className="flex gap-4 text-xs font-mono">
          <span className="px-2 py-1 rounded bg-red-100 text-red-800">
            S={cer_detail.substitutions}
          </span>
          <span className="px-2 py-1 rounded bg-yellow-100 text-yellow-800">
            D={cer_detail.deletions}
          </span>
          <span className="px-2 py-1 rounded bg-blue-100 text-blue-800">
            I={cer_detail.insertions}
          </span>
          <span className="px-2 py-1 rounded bg-slate-100 text-slate-700">
            N={cer_detail.total_chars_gt}
          </span>
          <span className="px-2 py-1 rounded bg-purple-100 text-purple-800 font-semibold">
            CER={((page.cer ?? 0) * 100).toFixed(2)}%
          </span>
        </div>
      )}

      {/* Legend */}
      <div className="flex gap-3 text-xs">
        <span className="flex items-center gap-1">
          <span className="px-1 rounded bg-red-100 border-b-2 border-red-500 font-mono">S</span> Substitution
        </span>
        <span className="flex items-center gap-1">
          <span className="px-1 rounded bg-yellow-100 border-b-2 border-yellow-500 font-mono">D</span> Deletion
        </span>
        <span className="flex items-center gap-1">
          <span className="px-1 rounded bg-blue-100 border-b-2 border-blue-500 font-mono">I</span> Insertion
        </span>
      </div>

      {/* Two-col diff */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <p className="text-xs text-slate-500 font-medium mb-2 uppercase tracking-wide">Ground Truth</p>
          <div className="font-mono text-sm leading-relaxed break-words bg-slate-50 rounded-lg p-3 border">
            {char_alignment.map((a, i) => {
              if (a.type === 'insertion') return null  // only in pred
              const cls = COLOR[a.type]
              return (
                <span key={i} className={`${cls} relative group cursor-default`}>
                  {a.gt}
                  {a.type !== 'match' && (
                    <span className="absolute -top-5 left-0 text-[9px] font-bold px-0.5 opacity-80">
                      {LABEL[a.type]}
                    </span>
                  )}
                </span>
              )
            })}
          </div>
        </div>

        <div>
          <p className="text-xs text-slate-500 font-medium mb-2 uppercase tracking-wide">Prediction</p>
          <div className="font-mono text-sm leading-relaxed break-words bg-slate-50 rounded-lg p-3 border">
            {char_alignment.map((a, i) => {
              if (a.type === 'deletion') return null  // only in GT
              const cls = COLOR[a.type]
              return (
                <span key={i} className={`${cls} relative group cursor-default`}>
                  {a.pred}
                  {a.type !== 'match' && (
                    <span className="absolute -top-5 left-0 text-[9px] font-bold px-0.5 opacity-80">
                      {LABEL[a.type]}
                    </span>
                  )}
                </span>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}
