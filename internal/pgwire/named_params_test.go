package pgwire

import "testing"

func TestNamedAtParameterProtocolHelpers(t *testing.T) {
	query := "SELECT VECTOR_DISTANCE(d.embedding, @query_vec) FROM documents d WHERE d.id = $2"
	if got := countParams(query); got != 2 {
		t.Fatalf("countParams=%d, want 2", got)
	}
	bound, err := substituteParams(query, [][]byte{[]byte("[1,0,0]"), []byte("doc-2")})
	if err != nil {
		t.Fatalf("substituteParams: %v", err)
	}
	want := "SELECT VECTOR_DISTANCE(d.embedding, '[1,0,0]') FROM documents d WHERE d.id = 'doc-2'"
	if bound != want {
		t.Fatalf("substituted query=%q, want %q", bound, want)
	}
}
