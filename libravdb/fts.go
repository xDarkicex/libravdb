package libravdb

import (
	"math"
	"sort"
	"strings"
	"unicode"
)

// ftsVector is the scan-time representation of a PostgreSQL-like tsvector.
// Positions are retained so phrase queries and cover-density-style ranking can
// be evaluated without changing the durable record format.
type ftsVector struct {
	terms map[string][]int
}

type ftsQueryKind uint8

const (
	ftsQueryTerm ftsQueryKind = iota
	ftsQueryAnd
	ftsQueryOr
	ftsQueryNot
	ftsQueryPhrase
)

type ftsQueryNode struct {
	kind        ftsQueryKind
	term        string
	prefix      bool
	left, right *ftsQueryNode
	phrase      []string
}

func buildFTSVector(text string) ftsVector {
	return buildFTSVectorConfig(text, "simple")
}

func buildFTSVectorConfig(text, config string) ftsVector {
	vector := ftsVector{terms: make(map[string][]int)}
	position := 0
	for _, word := range ftsWords(text) {
		position++
		term := normalizeFTSTermConfig(word, config)
		if term == "" {
			continue
		}
		vector.terms[term] = append(vector.terms[term], position)
	}
	return vector
}

func ftsWords(text string) []string {
	return strings.FieldsFunc(text, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '_'
	})
}

func normalizeFTSTerm(term string) string {
	return strings.ToLower(strings.TrimSpace(term))
}

// normalizeFTSTermConfig implements the deterministic built-in dictionaries
// used by the scan-time fallback.  simple is an exact lowercase dictionary;
// english applies common stop-word removal and conservative stemming.  A
// durable PostgreSQL dictionary/configuration catalog can replace this layer
// without changing the SQL surface.
func normalizeFTSTermConfig(term, config string) string {
	term = normalizeFTSTerm(term)
	if term == "" {
		return ""
	}
	if !strings.EqualFold(config, "english") && !strings.EqualFold(config, "english_stem") {
		return term
	}
	if englishStopWords[term] {
		return ""
	}
	return englishStem(term)
}

var englishStopWords = map[string]bool{
	"a": true, "an": true, "and": true, "are": true, "as": true, "at": true,
	"be": true, "by": true, "for": true, "from": true, "in": true, "into": true,
	"is": true, "it": true, "of": true, "on": true, "or": true, "that": true,
	"the": true, "their": true, "there": true, "this": true, "to": true,
	"was": true, "were": true, "with": true,
}

func englishStem(term string) string {
	// This is intentionally conservative: it handles the common inflections
	// that occur in application search text while avoiding aggressive rewrites
	// that would create false positives in a scan-time dictionary.
	if len(term) > 5 && strings.HasSuffix(term, "ies") {
		return term[:len(term)-3] + "y"
	}
	if len(term) > 5 && strings.HasSuffix(term, "sses") {
		return term[:len(term)-2]
	}
	if len(term) > 4 && strings.HasSuffix(term, "ing") && hasVowel(term[:len(term)-3]) {
		return term[:len(term)-3]
	}
	if len(term) > 4 && strings.HasSuffix(term, "ed") && hasVowel(term[:len(term)-2]) {
		return term[:len(term)-2]
	}
	if len(term) > 4 && strings.HasSuffix(term, "es") {
		return term[:len(term)-2]
	}
	if len(term) > 3 && strings.HasSuffix(term, "s") && !strings.HasSuffix(term, "ss") {
		return term[:len(term)-1]
	}
	return term
}

func hasVowel(term string) bool {
	for i := 0; i < len(term); i++ {
		switch term[i] {
		case 'a', 'e', 'i', 'o', 'u', 'y':
			return true
		}
	}
	return false
}

func parseFTSQuery(query, mode string) *ftsQueryNode {
	return parseFTSQueryConfig(query, mode, "simple")
}

func parseFTSQueryConfig(query, mode, config string) *ftsQueryNode {
	if mode == "plain" {
		var root *ftsQueryNode
		for _, word := range ftsWords(query) {
			term := normalizeFTSTermConfig(word, config)
			if term == "" {
				continue
			}
			node := &ftsQueryNode{kind: ftsQueryTerm, term: term}
			if root == nil {
				root = node
			} else {
				root = &ftsQueryNode{kind: ftsQueryAnd, left: root, right: node}
			}
		}
		return root
	}
	if mode == "phrase" {
		phrase := make([]string, 0)
		for _, word := range ftsWords(query) {
			if term := normalizeFTSTermConfig(word, config); term != "" {
				phrase = append(phrase, term)
			}
		}
		if len(phrase) == 0 {
			return nil
		}
		return &ftsQueryNode{kind: ftsQueryPhrase, phrase: phrase}
	}
	if mode == "web" {
		return parseWebSearchQueryConfig(query, config)
	}
	return parseRawTSQueryConfig(query, config)
}

func parseWebSearchQuery(query string) *ftsQueryNode {
	return parseWebSearchQueryConfig(query, "simple")
}

func parseWebSearchQueryConfig(query, config string) *ftsQueryNode {
	var root *ftsQueryNode
	pendingOr := false
	words := []byte(query)
	for i := 0; i < len(words); {
		if words[i] == ' ' || words[i] == '\t' || words[i] == '\n' {
			i++
			continue
		}
		if words[i] == '"' {
			start := i + 1
			i = start
			for i < len(words) && words[i] != '"' {
				i++
			}
			phrase := make([]string, 0)
			for _, word := range ftsWords(string(words[start:i])) {
				if term := normalizeFTSTermConfig(word, config); term != "" {
					phrase = append(phrase, term)
				}
			}
			if len(phrase) > 0 {
				node := &ftsQueryNode{kind: ftsQueryPhrase, phrase: phrase}
				root = appendWebNode(root, node, pendingOr)
				pendingOr = false
			}
			if i < len(words) {
				i++
			}
			continue
		}
		start := i
		for i < len(words) && words[i] != ' ' && words[i] != '\t' && words[i] != '\n' {
			i++
		}
		token := string(words[start:i])
		if strings.EqualFold(token, "OR") {
			pendingOr = true
			continue
		}
		negated := strings.HasPrefix(token, "-")
		if negated {
			token = token[1:]
		}
		term := normalizeFTSTermConfig(token, config)
		if term == "" {
			continue
		}
		node := &ftsQueryNode{kind: ftsQueryTerm, term: term}
		if negated {
			node = &ftsQueryNode{kind: ftsQueryNot, left: node}
		}
		root = appendWebNode(root, node, pendingOr)
		pendingOr = false
	}
	return root
}

func appendWebNode(root, node *ftsQueryNode, useOr bool) *ftsQueryNode {
	if root == nil {
		return node
	}
	if useOr {
		return &ftsQueryNode{kind: ftsQueryOr, left: root, right: node}
	}
	return &ftsQueryNode{kind: ftsQueryAnd, left: root, right: node}
}

func parseRawTSQuery(query string) *ftsQueryNode {
	return parseRawTSQueryConfig(query, "simple")
}

func parseRawTSQueryConfig(query, config string) *ftsQueryNode {
	parser := ftsQueryParser{tokens: scanTSQuery(query), config: config}
	return parser.parseOr()
}

type ftsQueryParser struct {
	tokens []string
	pos    int
	config string
}

func scanTSQuery(query string) []string {
	var tokens []string
	for i := 0; i < len(query); {
		if query[i] == ' ' || query[i] == '\t' || query[i] == '\n' {
			i++
			continue
		}
		if strings.ContainsRune("&|!()", rune(query[i])) {
			tokens = append(tokens, query[i:i+1])
			i++
			continue
		}
		if i+2 < len(query) && query[i:i+3] == "<->" {
			tokens = append(tokens, "<->")
			i += 3
			continue
		}
		start := i
		for i < len(query) && !strings.ContainsRune(" \t\n&|!()", rune(query[i])) {
			i++
		}
		tokens = append(tokens, query[start:i])
	}
	return tokens
}

func (p *ftsQueryParser) parseOr() *ftsQueryNode {
	left := p.parseAnd()
	for p.pos < len(p.tokens) && p.tokens[p.pos] == "|" {
		p.pos++
		left = &ftsQueryNode{kind: ftsQueryOr, left: left, right: p.parseAnd()}
	}
	return left
}

func (p *ftsQueryParser) parseAnd() *ftsQueryNode {
	left := p.parseUnary()
	for p.pos < len(p.tokens) && (p.tokens[p.pos] == "&" || p.tokens[p.pos] == "<->") {
		op := p.tokens[p.pos]
		p.pos++
		right := p.parseUnary()
		if op == "<->" {
			left = &ftsQueryNode{kind: ftsQueryPhrase, phrase: queryTerms(left, right)}
		} else {
			left = &ftsQueryNode{kind: ftsQueryAnd, left: left, right: right}
		}
	}
	return left
}

func (p *ftsQueryParser) parseUnary() *ftsQueryNode {
	if p.pos < len(p.tokens) && p.tokens[p.pos] == "!" {
		p.pos++
		return &ftsQueryNode{kind: ftsQueryNot, left: p.parseUnary()}
	}
	if p.pos < len(p.tokens) && p.tokens[p.pos] == "(" {
		p.pos++
		node := p.parseOr()
		if p.pos < len(p.tokens) && p.tokens[p.pos] == ")" {
			p.pos++
		}
		return node
	}
	if p.pos >= len(p.tokens) {
		return nil
	}
	token := p.tokens[p.pos]
	prefix := strings.HasSuffix(token, ":*")
	term := normalizeFTSTermConfig(strings.TrimSuffix(token, ":*"), p.config)
	p.pos++
	if term == "" {
		return nil
	}
	return &ftsQueryNode{kind: ftsQueryTerm, term: term, prefix: prefix}
}

func queryTerms(left, right *ftsQueryNode) []string {
	terms := make([]string, 0, 2)
	collectQueryTerms(&terms, left)
	collectQueryTerms(&terms, right)
	return terms
}

func collectQueryTerms(out *[]string, node *ftsQueryNode) {
	if node == nil {
		return
	}
	if node.kind == ftsQueryTerm {
		*out = append(*out, node.term)
		return
	}
	if node.kind == ftsQueryPhrase {
		*out = append(*out, node.phrase...)
		return
	}
	collectQueryTerms(out, node.left)
	collectQueryTerms(out, node.right)
}

func ftsQueryMatches(vector ftsVector, node *ftsQueryNode) bool {
	if node == nil {
		return false
	}
	switch node.kind {
	case ftsQueryTerm:
		if positions, ok := vector.terms[node.term]; ok && len(positions) > 0 {
			return true
		}
		if node.prefix {
			for term := range vector.terms {
				if strings.HasPrefix(term, node.term) {
					return true
				}
			}
		}
		return false
	case ftsQueryAnd:
		return ftsQueryMatches(vector, node.left) && ftsQueryMatches(vector, node.right)
	case ftsQueryOr:
		return ftsQueryMatches(vector, node.left) || ftsQueryMatches(vector, node.right)
	case ftsQueryNot:
		return !ftsQueryMatches(vector, node.left)
	case ftsQueryPhrase:
		if len(node.phrase) == 0 {
			return false
		}
		for _, start := range vector.terms[node.phrase[0]] {
			matched := true
			for i := 1; i < len(node.phrase); i++ {
				if !containsPosition(vector.terms[node.phrase[i]], start+i) {
					matched = false
					break
				}
			}
			if matched {
				return true
			}
		}
	}
	return false
}

func containsPosition(positions []int, target int) bool {
	for _, position := range positions {
		if position == target {
			return true
		}
	}
	return false
}

func ftsRankText(text, query, mode string) float64 {
	return ftsRankTextConfig(text, query, mode, "simple")
}

func ftsRankTextConfig(text, query, mode, config string) float64 {
	return ftsRankTextConfigNorm(text, query, mode, config, 0)
}

func ftsRankTextConfigNorm(text, query, mode, config string, normalization uint32) float64 {
	vector := buildFTSVectorConfig(text, config)
	root := parseFTSQueryConfig(query, mode, config)
	if root == nil || !ftsQueryMatches(vector, root) {
		return 0
	}
	denominator := ftsPositiveTermCount(root)
	if denominator == 0 {
		return 0
	}
	score := ftsNodeScore(vector, root)
	if score <= 0 {
		return 0
	}
	// PostgreSQL's rank functions give repeated lexemes more weight than a
	// single occurrence. Keep this scan-time implementation deterministic
	// while preserving that useful ordering property.
	score /= float64(denominator)
	if normalization&1 != 0 {
		score /= 1 + math.Log(float64(totalFTSTermCount(vector)))
	}
	if normalization&2 != 0 {
		score /= float64(totalFTSTermCount(vector))
	}
	if normalization&8 != 0 {
		score /= float64(len(vector.terms))
	}
	if normalization&16 != 0 {
		score /= 1 + math.Log(float64(len(vector.terms)))
	}
	return score
}

func totalFTSTermCount(vector ftsVector) int {
	total := 0
	for _, positions := range vector.terms {
		total += len(positions)
	}
	if total == 0 {
		return 1
	}
	return total
}

func ftsPositiveTermCount(node *ftsQueryNode) int {
	if node == nil {
		return 0
	}
	switch node.kind {
	case ftsQueryTerm:
		return 1
	case ftsQueryPhrase:
		return len(node.phrase)
	case ftsQueryNot:
		return 0
	case ftsQueryAnd, ftsQueryOr:
		return ftsPositiveTermCount(node.left) + ftsPositiveTermCount(node.right)
	default:
		return 0
	}
}

func ftsNodeScore(vector ftsVector, node *ftsQueryNode) float64 {
	if node == nil {
		return 0
	}
	switch node.kind {
	case ftsQueryTerm:
		if positions := vector.terms[node.term]; len(positions) > 0 {
			return float64(len(positions))
		}
		if node.prefix {
			var count int
			for term, positions := range vector.terms {
				if strings.HasPrefix(term, node.term) {
					count += len(positions)
				}
			}
			return float64(count)
		}
		return 0
	case ftsQueryPhrase:
		if !ftsQueryMatches(vector, node) {
			return 0
		}
		return float64(len(node.phrase))
	case ftsQueryAnd:
		return ftsNodeScore(vector, node.left) + ftsNodeScore(vector, node.right)
	case ftsQueryOr:
		left, right := ftsNodeScore(vector, node.left), ftsNodeScore(vector, node.right)
		if left > right {
			return left
		}
		return right
	case ftsQueryNot:
		return 0
	default:
		return 0
	}
}

func ftsVectorString(text string) string {
	return ftsVectorStringConfig(text, "simple")
}

func ftsVectorStringConfig(text, config string) string {
	vector := buildFTSVectorConfig(text, config)
	terms := make([]string, 0, len(vector.terms))
	for term := range vector.terms {
		terms = append(terms, term)
	}
	sort.Strings(terms)
	var out strings.Builder
	for i, term := range terms {
		if i > 0 {
			out.WriteByte(' ')
		}
		out.WriteString(term)
		out.WriteByte(':')
		for j, position := range vector.terms[term] {
			if j > 0 {
				out.WriteByte(',')
			}
			out.WriteString(strconvItoa(position))
		}
	}
	return out.String()
}

func ftsQueryString(query, mode string) string {
	return ftsQueryStringConfig(query, mode, "simple")
}

func ftsQueryStringConfig(query, mode, config string) string {
	root := parseFTSQueryConfig(query, mode, config)
	return formatFTSQuery(root)
}

func formatFTSQuery(node *ftsQueryNode) string {
	return formatFTSQueryPrec(node, -1)
}

func formatFTSQueryPrec(node *ftsQueryNode, parentPrecedence int) string {
	if node == nil {
		return ""
	}
	precedence := 3
	var value string
	switch node.kind {
	case ftsQueryTerm:
		if node.prefix {
			value = node.term + ":*"
		} else {
			value = node.term
		}
	case ftsQueryPhrase:
		value = strings.Join(node.phrase, " <-> ")
	case ftsQueryNot:
		precedence = 2
		value = "!" + formatFTSQueryPrec(node.left, precedence)
	case ftsQueryAnd:
		precedence = 1
		value = formatFTSQueryPrec(node.left, precedence) + " & " + formatFTSQueryPrec(node.right, precedence)
	case ftsQueryOr:
		precedence = 0
		value = formatFTSQueryPrec(node.left, precedence) + " | " + formatFTSQueryPrec(node.right, precedence)
	default:
		return ""
	}
	if precedence < parentPrecedence {
		return "(" + value + ")"
	}
	return value
}

func strconvItoa(value int) string {
	if value == 0 {
		return "0"
	}
	var buf [20]byte
	pos := len(buf)
	for value > 0 {
		pos--
		buf[pos] = byte('0' + value%10)
		value /= 10
	}
	return string(buf[pos:])
}
