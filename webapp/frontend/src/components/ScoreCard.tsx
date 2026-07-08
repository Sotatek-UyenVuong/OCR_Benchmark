import type { Summary } from '../types'

interface Props { summary: Summary }

const MetricBadge = ({ label, value, lowerBetter }: { label: string; value?: number; lowerBetter?: boolean }) => {
  if (value === undefined) return null
  const pct = Math.round(value * 100)
  const good = lowerBetter ? value < 0.05 : value > 0.85
  const mid  = lowerBetter ? value < 0.15 : value > 0.60
  const color = good ? 'bg-green-100 text-green-800 border-green-300'
              : mid  ? 'bg-yellow-100 text-yellow-800 border-yellow-300'
              :         'bg-red-100 text-red-800 border-red-300'

  return (
    <div className={`flex flex-col items-center px-4 py-3 rounded-lg border ${color}`}>
      <span className="text-xs font-medium uppercase tracking-wide opacity-70">{label}</span>
      <span className="text-2xl font-bold font-mono mt-1">{pct}%</span>
    </div>
  )
}

export function ScoreCard({ summary }: Props) {
  const isTextUC = summary.uc_type !== 'table'

  return (
    <div className="bg-white rounded-xl border border-blue-100 p-5 shadow-sm">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="font-semibold text-blue-900">{summary.model_id}</h3>
          <p className="text-xs text-slate-500">{summary.doc_id} · {summary.n_pages} page(s)</p>
        </div>
        <div className="text-right">
          <span className="text-xs text-slate-400">Processing time</span>
          <p className="font-mono font-semibold text-amber-700">{summary.processing_time_ms.toFixed(1)} ms</p>
        </div>
      </div>

      {/* Metrics grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
        {isTextUC ? (
          <>
            <MetricBadge label="CER"        value={summary.avg_cer}        lowerBetter />
            <MetricBadge label="WER"        value={summary.avg_wer}        lowerBetter />
            <MetricBadge label="nWER"       value={summary.avg_nwer}       lowerBetter />
            <MetricBadge label="PCS"        value={summary.avg_pcs} />
            <MetricBadge label="Punct mIoU" value={summary.avg_punct_miou} />
            <MetricBadge label="Cap mIoU"   value={summary.avg_cap_miou} />
            {summary.avg_mean_iou !== undefined &&
              <MetricBadge label="Layout IoU" value={summary.avg_mean_iou} />}
          </>
        ) : (
          <MetricBadge label="TEDS" value={summary.avg_teds} />
        )}
      </div>

      {/* Legend */}
      <div className="mt-4 flex gap-4 text-xs text-slate-500">
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-green-400 inline-block"/> Good</span>
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-yellow-400 inline-block"/> Fair</span>
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-red-400 inline-block"/> Poor</span>
      </div>
    </div>
  )
}
