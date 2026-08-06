package singlefile

// SetTestCommitFailpoint installs a hook that CommitTx calls after building
// all WAL frames but before writing the commit marker. Return a non-nil error
// to simulate a WAL write failure. Call ClearTestCommitFailpoint to remove.
func SetTestCommitFailpoint(fn func() error) {
	testCommitFailpoint = fn
}

// ClearTestCommitFailpoint removes the test failpoint hook.
func ClearTestCommitFailpoint() {
	testCommitFailpoint = nil
}
