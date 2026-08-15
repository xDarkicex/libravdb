package pgwire

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"testing"
	"time"
)

func TestContextPasswordLookupSuccessAndLegacyFallback(t *testing.T) {
	credential, err := deriveSCRAMCredential("alice", "secret", []byte("fixed-scram-salt"), DefaultSCRAMIterations)
	if err != nil {
		t.Fatal(err)
	}
	called := false
	contextConfig := ServerConfig{
		PasswordLookupContext: func(ctx context.Context, username string) (SCRAMCredential, bool) {
			called = true
			if err := ctx.Err(); err != nil {
				return SCRAMCredential{}, false
			}
			return credential, username == "alice"
		},
	}
	got, found, err := lookupSCRAMCredentialContext(context.Background(), contextConfig, "alice")
	if err != nil || !found || got.Username != "alice" || !called {
		t.Fatalf("context lookup failed: found=%v err=%v called=%v", found, err, called)
	}

	legacyConfig := ServerConfig{PasswordLookup: func(username string) (SCRAMCredential, bool) {
		return credential, username == "alice"
	}}
	got, found = lookupSCRAMCredential(legacyConfig, "alice")
	if !found || got.Username != "alice" {
		t.Fatalf("legacy lookup fallback failed: found=%v user=%q", found, got.Username)
	}
}

func TestContextPasswordLookupTimeoutInsideSCRAM(t *testing.T) {
	config := ServerConfig{
		PasswordLookupContext: func(ctx context.Context, _ string) (SCRAMCredential, bool) {
			<-ctx.Done()
			return SCRAMCredential{}, false
		},
		PasswordLookupTimeout: 10 * time.Millisecond,
	}
	rw := &startupTestReadWriter{reader: bytes.NewReader(scramInitialWire("n,,n=alice,r=client-nonce"))}
	started := time.Now()
	err := authenticateSCRAMContext(context.Background(), rw, "alice", config)
	if err == nil || !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected lookup deadline error, got %v", err)
	}
	if elapsed := time.Since(started); elapsed > time.Second {
		t.Fatalf("lookup timeout took too long: %v", elapsed)
	}
}

func TestContextPasswordLookupParentCancellationInsideSCRAM(t *testing.T) {
	started := make(chan struct{})
	config := ServerConfig{
		PasswordLookupContext: func(ctx context.Context, _ string) (SCRAMCredential, bool) {
			close(started)
			<-ctx.Done()
			return SCRAMCredential{}, false
		},
		PasswordLookupTimeout: time.Minute,
	}
	rw := &startupTestReadWriter{reader: bytes.NewReader(scramInitialWire("n,,n=alice,r=client-nonce"))}
	ctx, cancel := context.WithCancel(context.Background())
	errCh := make(chan error, 1)
	go func() { errCh <- authenticateSCRAMContext(ctx, rw, "alice", config) }()
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("context-aware lookup was not invoked")
	}
	cancel()
	select {
	case err := <-errCh:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("expected parent cancellation, got %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("SCRAM lookup did not stop after parent cancellation")
	}
}

func TestContextCancellationIsNotCountedAsAuthFailure(t *testing.T) {
	if shouldRecordAuthFailure(context.Canceled) || shouldRecordAuthFailure(context.DeadlineExceeded) {
		t.Fatal("context cancellation was counted as an authentication failure")
	}
}

func scramInitialWire(clientFirst string) []byte {
	initial := append([]byte("SCRAM-SHA-256\x00"), 0, 0, 0, 0)
	initial = append(initial, []byte(clientFirst)...)
	// The SASL initial-response length starts immediately after the mechanism NUL.
	binary.BigEndian.PutUint32(initial[len("SCRAM-SHA-256")+1:], uint32(len(clientFirst)))
	var wire bytes.Buffer
	_ = WriteMessage(&wire, msgPassword, initial)
	return wire.Bytes()
}
