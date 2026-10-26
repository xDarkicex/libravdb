package libravdb

import (
	"strings"

	"github.com/xDarkicex/libravdb/internal/optimizer"
)

func (e *Executor) attachRecordMetadata(col *Collection, sr *SearchResult, plan *optimizer.PhysicalPlan, rec Record) *SearchResult {
	e.attachVectorFunctions(col, sr, plan, rec)
	attachVectorOperators(sr, plan, rec)
	attachFTSRanks(sr, plan, rec)
	attachFTSValues(sr, plan, rec)
	if len(plan.Projections) == 0 {
		sr.Metadata = rec.Metadata
		return sr
	}
	projected := make(map[string]interface{}, len(plan.Projections))
	for _, name := range plan.Projections {
		if value, ok := e.projectRecordValue(sr, rec, plan, name); ok {
			projected[name] = value
		}
	}
	sr.Metadata = projected
	return sr
}
func setProjectedValue(sr *SearchResult, name string, value interface{}) {
	if sr.Metadata == nil {
		sr.Metadata = make(map[string]interface{})
	}
	sr.Metadata[name] = value
}
func (e *Executor) attachVectorFunctions(col *Collection, sr *SearchResult, plan *optimizer.PhysicalPlan, rec Record) {
	if len(rec.Vector) == 0 {
		return
	}
	for _, projection := range plan.VectorFuncProjections {
		if len(projection.QueryVector) == 0 || len(projection.QueryVector) != len(rec.Vector) {
			continue
		}
		setProjectedValue(sr, projection.Name, computeVectorScore(col, projection, rec.Vector))
	}
}
func attachVectorOperators(sr *SearchResult, plan *optimizer.PhysicalPlan, rec Record) {
	if len(rec.Vector) == 0 {
		return
	}
	for _, projection := range plan.VectorOperatorProjections {
		if score, ok := vectorOperatorScore(projection.Operator, projection.QueryVector, rec.Vector); ok {
			setProjectedValue(sr, projection.Name, score)
		}
	}
}
func attachFTSRanks(sr *SearchResult, plan *optimizer.PhysicalPlan, rec Record) {
	for _, projection := range plan.FTSRankProjections {
		rank := float64(0)
		if value, ok := recordMetadataValue(rec.Metadata, projection.TextColumn); ok && value != nil {
			rank = ftsRankText(recordMetaToString(value), projection.TextQuery, "plain")
		}
		setProjectedValue(sr, projection.Name, rank)
	}
}
func attachFTSValues(sr *SearchResult, plan *optimizer.PhysicalPlan, rec Record) {
	for _, projection := range plan.FTSProjections {
		text := projection.Query
		if projection.Column != "" {
			text = ""
			if value, ok := recordMetadataValue(rec.Metadata, projection.Column); ok && value != nil {
				text = recordMetaToString(value)
			}
		}
		if value, ok := ftsProjectionValue(projection, text); ok {
			setProjectedValue(sr, projection.Name, value)
		}
	}
}
func ftsProjectionValue(projection optimizer.FTSProjection, text string) (interface{}, bool) {
	switch projection.Kind {
	case optimizer.FTSProjectionVector:
		return ftsVectorStringConfig(text, projection.Config), true
	case optimizer.FTSProjectionQuery:
		return ftsQueryStringConfig(text, projection.QueryMode, projection.Config), true
	case optimizer.FTSProjectionRank:
		return ftsRankTextConfigNorm(text, projection.Query, projection.QueryMode, projection.Config, projection.Normalization), true
	default:
		return nil, false
	}
}
func projectionSourceName(plan *optimizer.PhysicalPlan, name string) string {
	for _, ref := range plan.ProjectionRefs {
		if strings.EqualFold(ref.OutputName, name) {
			return ref.SourceName
		}
	}
	return name
}
func (e *Executor) projectRecordValue(sr *SearchResult, rec Record, plan *optimizer.PhysicalPlan, name string) (interface{}, bool) {
	if name == "id" || name == "ID" {
		return sr.ID, true
	}
	source := projectionSourceName(plan, name)
	if e.isVectorColumn(plan.CollectionName, source) && len(rec.Vector) > 0 {
		return formatConflictVector(rec.Vector), true
	}
	if value, ok := recordMetadataValue(sr.Metadata, source); ok {
		return value, true
	}
	return recordMetadataValue(rec.Metadata, source)
}
