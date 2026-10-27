package libravdb

import (
	"fmt"
	"strings"
)

func orderTextValue(value interface{}) string {
	if text, ok := value.(string); ok {
		return text
	}
	return fmt.Sprint(value)
}
func numericStringCandidate(text string) bool {
	if text == "" {
		return false
	}
	first := text[0]
	if first >= '0' && first <= '9' {
		return true
	}
	if first == '+' || first == '-' || first == '.' {
		return true
	}
	return strings.EqualFold(text, "NaN") || strings.EqualFold(text, "Inf") || strings.EqualFold(text, "Infinity")
}

func orderedRowsLess(left, right *SearchResult, column string) bool {
	a, aok := left.Metadata[column]
	b, bok := right.Metadata[column]
	if !aok && !bok {
		return left.ID < right.ID
	}
	if !aok {
		return true
	}
	if !bok {
		return false
	}
	return presentOrderValuesLess(left.ID, a, right.ID, b)
}
func presentOrderValuesLess(leftID string, a interface{}, rightID string, b interface{}) bool {
	af, an := toFloat(a)
	bf, bn := toFloat(b)
	if an && bn {
		if af != bf {
			return af < bf
		}
		return leftID < rightID
	}
	as, bs := orderTextValue(a), orderTextValue(b)
	if as != bs {
		return as < bs
	}
	return leftID < rightID
}
