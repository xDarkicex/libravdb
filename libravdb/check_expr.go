package libravdb

import (
	"fmt"
	"strings"
)

// checkTruth is SQL's three-valued boolean domain. CHECK rejects only False;
// Unknown (which is normally caused by a NULL operand) satisfies the
// constraint unless a separate NOT NULL constraint rejects the row.
type checkTruth uint8

const (
	checkFalse checkTruth = iota
	checkTrue
	checkUnknown
)

type checkTokenKind uint8

const (
	checkTokenEOF checkTokenKind = iota
	checkTokenIdentifier
	checkTokenNumber
	checkTokenString
	checkTokenOperator
	checkTokenLeftParen
	checkTokenRightParen
)

type checkToken struct {
	kind checkTokenKind
	text string
}

func lexCheckExpr(expr string) ([]checkToken, error) {
	tokens := make([]checkToken, 0, 8)
	for i := 0; i < len(expr); {
		if expr[i] == ' ' || expr[i] == '\t' || expr[i] == '\r' || expr[i] == '\n' {
			i++
			continue
		}
		switch expr[i] {
		case '(':
			tokens = append(tokens, checkToken{kind: checkTokenLeftParen, text: "("})
			i++
			continue
		case ')':
			tokens = append(tokens, checkToken{kind: checkTokenRightParen, text: ")"})
			i++
			continue
		case '\'':
			start := i
			i++
			var b strings.Builder
			for i < len(expr) {
				if expr[i] != '\'' {
					b.WriteByte(expr[i])
					i++
					continue
				}
				// SQL escapes a quote inside a string by doubling it.
				if i+1 < len(expr) && expr[i+1] == '\'' {
					b.WriteByte('\'')
					i += 2
					continue
				}
				i++
				tokens = append(tokens, checkToken{kind: checkTokenString, text: b.String()})
				_ = start
				goto nextToken
			}
			return nil, fmt.Errorf("unterminated string literal in CHECK expression")
		case '=', '!', '<', '>':
			start := i
			i++
			if i < len(expr) && (expr[i] == '=' || (expr[start] == '<' && expr[i] == '>')) {
				i++
			}
			tokens = append(tokens, checkToken{kind: checkTokenOperator, text: expr[start:i]})
			continue
		}
		if (expr[i] >= '0' && expr[i] <= '9') || (expr[i] == '-' && i+1 < len(expr) && expr[i+1] >= '0' && expr[i+1] <= '9') {
			start := i
			if expr[i] == '-' {
				i++
			}
			for i < len(expr) && ((expr[i] >= '0' && expr[i] <= '9') || expr[i] == '.') {
				i++
			}
			tokens = append(tokens, checkToken{kind: checkTokenNumber, text: expr[start:i]})
			continue
		}
		if isCheckIdentStart(expr[i]) {
			start := i
			i++
			for i < len(expr) && isCheckIdentPart(expr[i]) {
				i++
			}
			tokens = append(tokens, checkToken{kind: checkTokenIdentifier, text: expr[start:i]})
			continue
		}
		return nil, fmt.Errorf("unexpected character %q in CHECK expression", expr[i])
	nextToken:
	}
	tokens = append(tokens, checkToken{kind: checkTokenEOF})
	return tokens, nil
}

func isCheckIdentStart(c byte) bool {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'
}

func isCheckIdentPart(c byte) bool {
	return isCheckIdentStart(c) || (c >= '0' && c <= '9') || c == '.'
}

type checkExprParser struct {
	tokens []checkToken
	pos    int
	row    map[string]interface{}
}

func (p *checkExprParser) current() checkToken {
	if p.pos >= len(p.tokens) {
		return checkToken{kind: checkTokenEOF}
	}
	return p.tokens[p.pos]
}

func (p *checkExprParser) advance() checkToken {
	t := p.current()
	if p.pos < len(p.tokens) {
		p.pos++
	}
	return t
}

func checkWord(t checkToken, word string) bool {
	return t.kind == checkTokenIdentifier && strings.EqualFold(t.text, word)
}

func (p *checkExprParser) parse() (checkTruth, error) {
	result, err := p.parseOr()
	if err != nil {
		return checkFalse, err
	}
	if p.current().kind != checkTokenEOF {
		return checkFalse, fmt.Errorf("unsupported CHECK expression near %q", p.current().text)
	}
	return result, nil
}

func (p *checkExprParser) parseOr() (checkTruth, error) {
	left, err := p.parseAnd()
	if err != nil {
		return checkFalse, err
	}
	for checkWord(p.current(), "OR") {
		p.advance()
		right, rightErr := p.parseAnd()
		if rightErr != nil {
			return checkFalse, rightErr
		}
		left = checkOr(left, right)
	}
	return left, nil
}

func (p *checkExprParser) parseAnd() (checkTruth, error) {
	left, err := p.parseUnary()
	if err != nil {
		return checkFalse, err
	}
	for checkWord(p.current(), "AND") {
		p.advance()
		right, rightErr := p.parseUnary()
		if rightErr != nil {
			return checkFalse, rightErr
		}
		left = checkAnd(left, right)
	}
	return left, nil
}

func (p *checkExprParser) parseUnary() (checkTruth, error) {
	if checkWord(p.current(), "NOT") {
		p.advance()
		value, err := p.parseUnary()
		if err != nil {
			return checkFalse, err
		}
		if value == checkUnknown {
			return checkUnknown, nil
		}
		if value == checkTrue {
			return checkFalse, nil
		}
		return checkTrue, nil
	}
	if p.current().kind == checkTokenLeftParen {
		p.advance()
		value, err := p.parseOr()
		if err != nil {
			return checkFalse, err
		}
		if p.current().kind != checkTokenRightParen {
			return checkFalse, fmt.Errorf("malformed CHECK expression: missing ')'")
		}
		p.advance()
		return value, nil
	}
	return p.parsePredicate()
}

func (p *checkExprParser) parsePredicate() (checkTruth, error) {
	left, err := p.parseOperand()
	if err != nil {
		return checkFalse, err
	}
	if checkWord(p.current(), "IS") {
		p.advance()
		not := false
		if checkWord(p.current(), "NOT") {
			not = true
			p.advance()
		}
		if !checkWord(p.current(), "NULL") {
			return checkFalse, fmt.Errorf("expected NULL after IS in CHECK expression")
		}
		p.advance()
		isNull := left == nil
		if not {
			isNull = !isNull
		}
		if isNull {
			return checkTrue, nil
		}
		return checkFalse, nil
	}
	if checkWord(p.current(), "BETWEEN") {
		p.advance()
		low, lowErr := p.parseOperand()
		if lowErr != nil {
			return checkFalse, lowErr
		}
		if !checkWord(p.current(), "AND") {
			return checkFalse, fmt.Errorf("malformed BETWEEN: missing AND")
		}
		p.advance()
		high, highErr := p.parseOperand()
		if highErr != nil {
			return checkFalse, highErr
		}
		if left == nil || low == nil || high == nil {
			return checkUnknown, nil
		}
		if compareVals(left, low) >= 0 && compareVals(left, high) <= 0 {
			return checkTrue, nil
		}
		return checkFalse, nil
	}
	if p.current().kind == checkTokenOperator {
		op := p.advance().text
		right, rightErr := p.parseOperand()
		if rightErr != nil {
			return checkFalse, rightErr
		}
		if left == nil || right == nil {
			return checkUnknown, nil
		}
		cmp := compareVals(left, right)
		switch op {
		case "=":
			return boolTruth(cmp == 0), nil
		case "!=", "<>":
			return boolTruth(cmp != 0), nil
		case "<":
			return boolTruth(cmp < 0), nil
		case ">":
			return boolTruth(cmp > 0), nil
		case "<=":
			return boolTruth(cmp <= 0), nil
		case ">=":
			return boolTruth(cmp >= 0), nil
		default:
			return checkFalse, fmt.Errorf("unsupported CHECK operator %q", op)
		}
	}
	if left == nil {
		return checkUnknown, nil
	}
	if value, ok := left.(bool); ok {
		return boolTruth(value), nil
	}
	return checkFalse, fmt.Errorf("CHECK expression is not boolean")
}

func (p *checkExprParser) parseOperand() (interface{}, error) {
	t := p.advance()
	switch t.kind {
	case checkTokenString:
		return t.text, nil
	case checkTokenNumber:
		return parseDefaultLiteral(t.text), nil
	case checkTokenIdentifier:
		switch strings.ToUpper(t.text) {
		case "NULL":
			return nil, nil
		case "TRUE":
			return true, nil
		case "FALSE":
			return false, nil
		}
		if value, ok := p.row[t.text]; ok {
			return value, nil
		}
		for key, value := range p.row {
			if strings.EqualFold(key, t.text) {
				return value, nil
			}
		}
		// An absent nullable column has SQL NULL semantics.
		return nil, nil
	default:
		return nil, fmt.Errorf("expected literal or column in CHECK expression near %q", t.text)
	}
}

func boolTruth(value bool) checkTruth {
	if value {
		return checkTrue
	}
	return checkFalse
}

func checkAnd(a, b checkTruth) checkTruth {
	if a == checkFalse || b == checkFalse {
		return checkFalse
	}
	if a == checkUnknown || b == checkUnknown {
		return checkUnknown
	}
	return checkTrue
}

func checkOr(a, b checkTruth) checkTruth {
	if a == checkTrue || b == checkTrue {
		return checkTrue
	}
	if a == checkUnknown || b == checkUnknown {
		return checkUnknown
	}
	return checkFalse
}

func evaluateCheckBooleanExpr(expr string, metadata map[string]interface{}) (bool, error) {
	tokens, err := lexCheckExpr(expr)
	if err != nil {
		return false, err
	}
	truth, err := (&checkExprParser{tokens: tokens, row: metadata}).parse()
	if err != nil {
		return false, err
	}
	return truth != checkFalse, nil
}
