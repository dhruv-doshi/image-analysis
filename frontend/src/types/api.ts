export interface ExifData {
  camera_make?: string
  camera_model?: string
  iso?: number
  shutter_speed?: string
  aperture?: number
  focal_length?: number
  lens_model?: string
  image_width?: number
  image_height?: number
}

export interface QualityTier {
  overall: 'excellent' | 'good' | 'average' | 'poor' | 'terrible'
  brisque_tier: 'excellent' | 'good' | 'average' | 'poor' | 'terrible'
  sharpness_tier: 'excellent' | 'good' | 'average' | 'poor' | 'terrible'
  noise_tier: 'excellent' | 'good' | 'average' | 'poor' | 'terrible'
  exposure_tier: 'excellent' | 'good' | 'average' | 'poor' | 'terrible'
  composition_tier?: 'excellent' | 'good' | 'average' | 'poor' | 'terrible'
}

export interface TechnicalScores {
  brisque: number
  nima_aesthetic?: number
  clip_iqa?: number
  musiq?: number | null
  niqe?: number | null
  sharpness_laplacian: number
  sharpness_regional: { top_left: number; top_right: number; bottom_left: number; bottom_right: number }
  noise_sigma: number
  exposure_clipped_highlights_pct: number
  exposure_clipped_shadows_pct: number
  histogram_mean: number
  histogram_std: number
  dynamic_range_stops: number
  contrast_rms: number
}

export interface CompositionScores {
  saliency_centroid_x: number
  saliency_centroid_y: number
  rot_alignment_score: number
  golden_ratio_alignment_score: number
  best_alignment: 'rule_of_thirds' | 'golden_ratio'
  negative_space_ratio: number
  visual_weight_quadrants: { top_left: number; top_right: number; bottom_left: number; bottom_right: number }
  visual_weight_balance: number
  symmetry_horizontal: number
  symmetry_vertical: number
  dominant_line_angles: number[]
  leading_lines_converge_to_subject: boolean
  line_pattern: 'diagonal' | 'horizontal' | 'vertical' | 'mixed' | 'none'
  horizon_tilt_degrees?: number | null
  scene_type?: string
  color_harmony_type?: string
  color_harmony_score?: number
}

export type AnalyseStreamEvent =
  | { type: 'metrics'; exif: ExifData; quality_tier: QualityTier; technical: TechnicalScores; composition: CompositionScores }
  | { type: 'chunk'; text: string }
  | { type: 'report'; report: AnalysisReport }
  | { type: 'error'; message: string }
  | { type: 'done' }

export interface AnalysisReport {
  summary: string
  composition?: string
  aesthetics?: string
  technical?: string
  improvements?: string
  editing?: string
  inspiration?: string
}

export interface AnalyseResponse {
  exif: ExifData
  quality_tier: QualityTier
  technical: TechnicalScores
  composition: CompositionScores
  report: AnalysisReport
}
