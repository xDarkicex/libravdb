package libravdb

import (
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/xDarkicex/libravdb/internal/optimizer"
)

type preparedFTSPredicate struct {
	column, text, config string
	root                 *ftsQueryNode
	terms                []string
}

func prepareFTSPredicates(predicates []optimizer.FTSPredicate) []preparedFTSPredicate {
	prepared := make([]preparedFTSPredicate, len(predicates))
	for i, predicate := range predicates {
		root := parseFTSQueryConfig(predicate.Query, predicate.QueryMode, predicate.Config)
		item := preparedFTSPredicate{column: predicate.Column, text: predicate.Text, config: predicate.Config, root: root}
		if predicate.Config == "" || strings.EqualFold(predicate.Config, "simple") {
			item.terms = simpleFTSConjunction(root)
		}
		prepared[i] = item
	}
	return prepared
}
func simpleFTSConjunction(root *ftsQueryNode) []string {
	if root == nil {
		return nil
	}
	terms := []string{}
	if !appendSimpleFTSTerms(root, &terms) {
		return nil
	}
	return terms
}
func appendSimpleFTSTerms(node *ftsQueryNode, terms *[]string) bool {
	if node.kind == ftsQueryTerm && !node.prefix {
		*terms = append(*terms, node.term)
		return true
	}
	if node.kind != ftsQueryAnd {
		return false
	}
	return appendSimpleFTSTerms(node.left, terms) && appendSimpleFTSTerms(node.right, terms)
}
func matchesPreparedFTS(record Record, predicates []preparedFTSPredicate) bool {
	for _, predicate := range predicates {
		text := predicate.text
		if predicate.column != "" {
			value, ok := recordMetadataValue(record.Metadata, predicate.column)
			if !ok || value == nil {
				return false
			}
			text = recordMetaToString(value)
		}
		if !predicate.matches(text) {
			return false
		}
	}
	return true
}
func (predicate preparedFTSPredicate) matches(text string) bool {
	if len(predicate.terms) > 0 {
		for _, term := range predicate.terms {
			if !containsSimpleFTSTerm(text, term) {
				return false
			}
		}
		return true
	}
	if predicate.root == nil || ftsPositiveTermCount(predicate.root) == 0 {
		return false
	}
	vector := buildFTSVectorConfig(text, predicate.config)
	return ftsQueryMatches(vector, predicate.root) && ftsNodeScore(vector, predicate.root) > 0
}

// Simple positive conjunctions only need lexeme membership, not a map of every
// document word and position. Lowercase-rune equality matches ToLower exactly;
// EqualFold would incorrectly equate long-s with s in the simple dictionary.
func containsSimpleFTSTerm(text, term string) bool {
	start := -1
	for i, r := range text {
		if ftsLexemeRune(r) {
			if start < 0 {
				start = i
			}
			continue
		}
		if start >= 0 && equalSimpleFTSLexeme(text[start:i], term) {
			return true
		}
		start = -1
	}
	return start >= 0 && equalSimpleFTSLexeme(text[start:], term)
}
func equalSimpleFTSLexeme(word, term string) bool {
	if word == term {
		return true
	}
	for len(word) > 0 && len(term) > 0 {
		wr, wn := utf8.DecodeRuneInString(word)
		tr, tn := utf8.DecodeRuneInString(term)
		if unicode.ToLower(wr) != tr {
			return false
		}
		word, term = word[wn:], term[tn:]
	}
	return len(word) == 0 && len(term) == 0
}

func countSimpleFTSTerm(text, term string) int {
	count, start := 0, -1
	for i, r := range text {
		if ftsLexemeRune(r) {
			if start < 0 {
				start = i
			}
			continue
		}
		if start >= 0 && equalSimpleFTSLexeme(text[start:i], term) {
			count++
		}
		start = -1
	}
	if start >= 0 && equalSimpleFTSLexeme(text[start:], term) {
		count++
	}
	return count
}

func ftsLexemeRune(r rune) bool { return unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' }
