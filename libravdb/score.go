package libravdb

// publicScore converts an internal backend score into a consumer-facing
// relevance score where higher values are always better.
// Cosine collections expose cosine similarity semantics; other metrics expose
// monotone normalized relevance values suitable for thresholding and ranking.
func publicScore(metric DistanceMetric, raw float32) float32 {
	switch metric {
	case CosineDistance:
		return clampFloat32(1-raw, -1, 1)
	case L2Distance:
		if raw < 0 {
			raw = 0
		}
		return 1 / (1 + raw)
	case InnerProduct:
		return -raw
	default:
		return raw
	}
}

func normalizePublicSearchResults(metric DistanceMetric, results []*SearchResult) {
	for _, result := range results {
		result.Score = publicScore(metric, result.Score)
	}
}

func clampFloat32(v, minValue, maxValue float32) float32 {
	if v < minValue {
		return minValue
	}
	if v > maxValue {
		return maxValue
	}
	return v
}
