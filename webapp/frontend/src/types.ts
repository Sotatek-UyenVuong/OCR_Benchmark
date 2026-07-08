export interface Model {
  id: string
  name: string
  tier: string
}

export interface UcOption {
  id: string
  label: string
  type: 'scan' | 'table' | 'text_layer'
}

export interface CharAlignment {
  gt: string
  pred: string
  type: 'match' | 'substitution' | 'deletion' | 'insertion'
}

export interface CerDetail {
  substitutions: number
  deletions: number
  insertions: number
  total_chars_gt: number
}

export interface PageResult {
  doc_id: string
  page_num: number
  cer?: number
  cer_detail?: CerDetail
  wer?: number
  nwer?: number
  pcs?: number
  punct_miou?: number
  cap_miou?: number
  avg_teds?: number
  mean_iou?: number
  ground_truth?: string
  prediction?: string
  char_alignment?: CharAlignment[]
}

export interface Summary {
  model_id: string
  doc_id: string
  uc_type: string
  n_pages: number
  processing_time_ms: number
  avg_cer?: number
  avg_wer?: number
  avg_nwer?: number
  avg_pcs?: number
  avg_punct_miou?: number
  avg_cap_miou?: number
  avg_teds?: number
  avg_mean_iou?: number
}

export interface EvalResult {
  summary: Summary
  pages: PageResult[]
}
