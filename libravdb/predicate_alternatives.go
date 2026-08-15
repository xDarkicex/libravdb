package libravdb

import "github.com/xDarkicex/libravdb/internal/optimizer"

func planHasPredicates(plan *optimizer.PhysicalPlan) bool {
	return plan != nil && (len(plan.Predicates) > 0 || len(plan.PredicateAlternatives) > 0)
}

func planMatchesRecord(plan *optimizer.PhysicalPlan, record Record) bool {
	if plan == nil {
		return true
	}
	if len(plan.PredicateAlternatives) > 0 {
		for _, clause := range plan.PredicateAlternatives {
			if recordMatchesPredicates(record, clause) {
				return true
			}
		}
		return false
	}
	return recordMatchesPredicates(record, plan.Predicates)
}

func planMatchesSnapshotRecord(plan *optimizer.PhysicalPlan, record *Record) bool {
	if plan == nil || record == nil {
		return true
	}
	if len(plan.PredicateAlternatives) > 0 {
		for _, clause := range plan.PredicateAlternatives {
			if recordMatchesPredicatesSnapshot(record, clause) {
				return true
			}
		}
		return false
	}
	return recordMatchesPredicatesSnapshot(record, plan.Predicates)
}

func searchResultMatchesPlan(plan *optimizer.PhysicalPlan, result *SearchResult) bool {
	if plan == nil || result == nil {
		return true
	}
	if len(plan.PredicateAlternatives) > 0 {
		for _, clause := range plan.PredicateAlternatives {
			matched := true
			for _, predicate := range clause {
				if !predicateMatches(result, predicate) {
					matched = false
					break
				}
			}
			if matched {
				return true
			}
		}
		return false
	}
	for _, predicate := range plan.Predicates {
		if !predicateMatches(result, predicate) {
			return false
		}
	}
	return true
}

// graphJoinMatchesAlternatives evaluates the complete WHERE expression once
// both sides of a graph join are known. This is required for predicates such
// as `src.name LIKE $1 OR tgt.name LIKE $2`: neither side can be filtered
// independently without changing the meaning of the OR.
func graphJoinMatchesAlternatives(plan *optimizer.PhysicalPlan, aliases map[string]Record, defaultAlias string) bool {
	if plan == nil || len(plan.PredicateAlternatives) == 0 {
		return true
	}
	for _, clause := range plan.PredicateAlternatives {
		matched := true
		for _, predicate := range clause {
			alias := predicate.Alias
			if alias == "" {
				alias = defaultAlias
			}
			record, ok := recordForAlias(aliases, alias)
			if !ok || !recordMatchesPredicates(record, []optimizer.RelationalPredicate{predicate}) {
				matched = false
				break
			}
		}
		if matched {
			return true
		}
	}
	return false
}

func joinedRowMatchesAlternatives(row sqlJoinRow, alternatives optimizer.PredicateAlternatives, defaultAlias string) bool {
	if len(alternatives) == 0 {
		return true
	}
	for _, clause := range alternatives {
		matched := true
		for _, predicate := range clause {
			alias := predicate.Alias
			if alias == "" {
				alias = defaultAlias
			}
			record, ok := row.Sources[alias]
			if !ok {
				for name, candidate := range row.Sources {
					if equalFold(name, alias) {
						record, ok = candidate, true
						break
					}
				}
			}
			if !ok || record == nil || !recordMatchesPredicates(*record, []optimizer.RelationalPredicate{predicate}) {
				matched = false
				break
			}
		}
		if matched {
			return true
		}
	}
	return false
}

func recordForAlias(aliases map[string]Record, alias string) (Record, bool) {
	if record, ok := aliases[alias]; ok {
		return record, true
	}
	for name, record := range aliases {
		if equalFold(name, alias) {
			return record, true
		}
	}
	return Record{}, false
}

func equalFold(left, right string) bool {
	if len(left) != len(right) {
		return false
	}
	for i := range left {
		l, r := left[i], right[i]
		if l >= 'A' && l <= 'Z' {
			l += 'a' - 'A'
		}
		if r >= 'A' && r <= 'Z' {
			r += 'a' - 'A'
		}
		if l != r {
			return false
		}
	}
	return true
}
