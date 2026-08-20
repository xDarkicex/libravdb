package libravdb

import (
	"fmt"
	"strconv"

	"github.com/xDarkicex/libravdb/internal/optimizer"
)

// scalarPredicateMatches evaluates a non-NULL comparison using the typed
// optimizer value. It is shared by live, epoch, graph, and hybrid operators so
// no execution path falls back to SQL source-string coercion.
func scalarPredicateMatches(actual interface{}, pred optimizer.RelationalPredicate) bool {
	expected := pred.PredicateValue()
	if pred.ValueIsNull || expected.IsNull() {
		return false
	}
	if pred.Like || pred.ILike {
		return sqlLikeMatch(actual, expected.BytesData, pred.ILike)
	}
	if pred.InList || len(pred.InValues) > 0 {
		sawNull := false
		for _, value := range pred.InValues {
			if value.IsNull() {
				sawNull = true
				continue
			}
			cmp, actualNull, err := optimizer.CompareScalar(actual, value)
			if err == nil && !actualNull && cmp == 0 {
				return !pred.Not
			}
		}
		if pred.Not && sawNull {
			return false
		}
		return pred.Not
	}
	cmp, actualNull, err := optimizer.CompareScalar(actual, expected)
	if err != nil || actualNull {
		return false
	}
	if pred.Inclusive {
		if pred.Operator == 13 && cmp >= 0 {
			return true
		}
		if pred.Operator == 14 && cmp <= 0 {
			return true
		}
		return false
	}
	if pred.Not {
		return !optimizer.MatchesOperator(cmp, pred.Operator)
	}
	return optimizer.MatchesOperator(cmp, pred.Operator)
}

// sqlLikeMatch evaluates the SQL LIKE wildcard subset without converting the
// pattern or source into an intermediate regular expression. '%' matches any
// byte sequence and '_' matches one byte; ILIKE folds ASCII letters only,
// matching the lexer/parser's byte-oriented identifier semantics.
func sqlLikeMatch(actual interface{}, pattern []byte, insensitive bool) bool {
	var value string
	switch v := actual.(type) {
	case string:
		value = v
	case []byte:
		value = string(v)
	case int:
		value = strconv.Itoa(v)
	case int64:
		value = strconv.FormatInt(v, 10)
	case uint64:
		value = strconv.FormatUint(v, 10)
	case float64:
		value = strconv.FormatFloat(v, 'f', -1, 64)
	case float32:
		value = strconv.FormatFloat(float64(v), 'f', -1, 32)
	case bool:
		value = strconv.FormatBool(v)
	default:
		value = fmt.Sprint(actual)
	}

	i, j := 0, 0
	star, match := -1, 0
	for i < len(value) {
		if j < len(pattern) && pattern[j] != '%' && (pattern[j] == '_' || likeByteEqual(value[i], pattern[j], insensitive)) {
			i++
			j++
			continue
		}
		if j < len(pattern) && pattern[j] == '%' {
			star = j
			match = i
			j++
			continue
		}
		if star >= 0 {
			j = star + 1
			match++
			i = match
			continue
		}
		return false
	}
	for j < len(pattern) && pattern[j] == '%' {
		j++
	}
	return j == len(pattern)
}

func likeByteEqual(a byte, b byte, insensitive bool) bool {
	if insensitive {
		if a >= 'A' && a <= 'Z' {
			a += 'a' - 'A'
		}
		if b >= 'A' && b <= 'Z' {
			b += 'a' - 'A'
		}
	}
	return a == b
}
