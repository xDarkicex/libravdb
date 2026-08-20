package pgwire

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"io"
	"math/big"
	"net"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestStartupTLSNegotiationAndReady(t *testing.T) {
	cert := testTLSCertificate(t)
	server, client := net.Pipe()
	defer client.Close()
	defer server.Close()

	serverResult := make(chan error, 1)
	go func() {
		_, result, err := handleStartupWithConfig(server, nil, &tls.Config{
			Certificates: []tls.Certificate{cert},
			MinVersion:   tls.VersionTLS12,
		}, false, ServerConfig{})
		if err == nil && (result.User != "tls-user" || result.Database != "test") {
			serverResult <- &startupTestError{message: "unexpected TLS startup identity"}
			return
		}
		serverResult <- err
	}()

	writeSSLRequest(t, client)
	var response [1]byte
	if _, err := io.ReadFull(client, response[:]); err != nil {
		t.Fatalf("read SSL response: %v", err)
	}
	if response[0] != 'S' {
		t.Fatalf("expected SSL acceptance S, got %q", response[0])
	}
	secure := tls.Client(client, &tls.Config{InsecureSkipVerify: true, MinVersion: tls.VersionTLS12}) // test certificate is self-signed
	if err := secure.Handshake(); err != nil {
		t.Fatalf("TLS handshake: %v", err)
	}
	if err := sendStartupPacket(secure, "tls-user", "test"); err != nil {
		t.Fatalf("send TLS startup: %v", err)
	}
	assertStartupMessages(t, secure, 8)
	if err := <-serverResult; err != nil {
		t.Fatalf("server startup: %v", err)
	}
}

func TestStartupRequireTLSRejectsPlaintext(t *testing.T) {
	server, client := net.Pipe()
	defer client.Close()
	defer server.Close()

	serverResult := make(chan error, 1)
	go func() {
		_, _, err := handleStartupWithConfig(server, nil, nil, true, ServerConfig{RequireTLS: true})
		serverResult <- err
	}()
	if err := sendStartupPacket(client, "plain", "test"); err != nil {
		t.Fatalf("send plaintext startup: %v", err)
	}
	msgType, payload, err := ReadMessage(client)
	if err != nil {
		t.Fatalf("read TLS-required error: %v", err)
	}
	if msgType != msgErrorResponse || !strings.Contains(string(payload), "TLS is required") {
		t.Fatalf("expected TLS-required ErrorResponse, type=%q payload=%q", msgType, payload)
	}
	if err := <-serverResult; err == nil {
		t.Fatal("expected plaintext startup to fail")
	}
}

func TestStartupSCRAMSHA256(t *testing.T) {
	salt := []byte("fixed-scram-salt")
	credential, err := deriveSCRAMCredential("alice", "correct horse battery staple", salt, DefaultSCRAMIterations)
	if err != nil {
		t.Fatalf("derive credential: %v", err)
	}
	server, client := net.Pipe()
	defer client.Close()
	defer server.Close()
	config := ServerConfig{Credentials: map[string]SCRAMCredential{"alice": credential}}
	serverResult := make(chan error, 1)
	go func() {
		_, _, err := handleStartupWithConfig(server, nil, nil, false, config)
		serverResult <- err
	}()

	if err := sendStartupPacket(client, "alice", "test"); err != nil {
		t.Fatalf("send startup: %v", err)
	}
	msgType, payload, err := ReadMessage(client)
	if err != nil || msgType != msgAuth || len(payload) < 4 || binary.BigEndian.Uint32(payload[:4]) != 10 {
		t.Fatalf("expected AuthenticationSASL, type=%q err=%v payload=%x", msgType, err, payload)
	}
	nonce := "client-fixed-nonce"
	clientFirst := "n,,n=alice,r=" + nonce
	initial := append([]byte("SCRAM-SHA-256\x00"), 0, 0, 0, 0)
	binary.BigEndian.PutUint32(initial[len("SCRAM-SHA-256")+1:], uint32(len(clientFirst)))
	initial = append(initial, clientFirst...)
	if err := WriteMessage(client, msgPassword, initial); err != nil {
		t.Fatalf("send SCRAM client-first: %v", err)
	}
	msgType, payload, err = ReadMessage(client)
	if err != nil || msgType != msgAuth || len(payload) < 4 || binary.BigEndian.Uint32(payload[:4]) != 11 {
		t.Fatalf("expected AuthenticationSASLContinue, type=%q err=%v", msgType, err)
	}
	serverFirst := string(payload[4:])
	attrs, err := parseSCRAMAttributes(serverFirst)
	if err != nil {
		t.Fatalf("parse server-first: %v", err)
	}
	serverNonce := attrs['r']
	serverSalt, err := base64.StdEncoding.DecodeString(attrs['s'])
	if err != nil {
		t.Fatalf("decode server salt: %v", err)
	}
	iterations, err := strconv.Atoi(attrs['i'])
	if err != nil {
		t.Fatalf("parse iterations: %v", err)
	}
	clientFinalWithoutProof := "c=biws,r=" + serverNonce
	authMessage := clientFirst[3:] + "," + serverFirst + "," + clientFinalWithoutProof
	salted := testPBKDF2SHA256([]byte("correct horse battery staple"), serverSalt, iterations)
	clientKey := testHMACSHA256(salted, []byte("Client Key"))
	stored := sha256.Sum256(clientKey)
	clientSignature := testHMACSHA256(stored[:], []byte(authMessage))
	proof := make([]byte, len(clientKey))
	for i := range proof {
		proof[i] = clientKey[i] ^ clientSignature[i]
	}
	clientFinal := clientFinalWithoutProof + ",p=" + base64.StdEncoding.EncodeToString(proof)
	if err := WriteMessage(client, msgPassword, []byte(clientFinal)); err != nil {
		t.Fatalf("send SCRAM client-final: %v", err)
	}
	msgType, payload, err = ReadMessage(client)
	if err != nil || msgType != msgAuth || len(payload) < 4 || binary.BigEndian.Uint32(payload[:4]) != 12 {
		t.Fatalf("expected AuthenticationSASLFinal, type=%q err=%v", msgType, err)
	}
	msgType, payload, err = ReadMessage(client)
	if err != nil || msgType != msgAuth || len(payload) < 4 || binary.BigEndian.Uint32(payload[:4]) != 0 {
		t.Fatalf("expected AuthenticationOk, type=%q err=%v payload=%x", msgType, err, payload)
	}
	assertStartupMessages(t, client, 7)
	if err := <-serverResult; err != nil {
		t.Fatalf("SCRAM server startup: %v", err)
	}
}

func TestStartupSCRAMRejectsWrongPassword(t *testing.T) {
	credential, err := deriveSCRAMCredential("alice", "correct", []byte("fixed-scram-salt"), DefaultSCRAMIterations)
	if err != nil {
		t.Fatal(err)
	}
	server, client := net.Pipe()
	defer client.Close()
	defer server.Close()
	serverResult := make(chan error, 1)
	go func() {
		_, _, err := handleStartupWithConfig(server, nil, nil, false, ServerConfig{Credentials: map[string]SCRAMCredential{"alice": credential}})
		serverResult <- err
	}()
	if err := sendStartupPacket(client, "alice", "test"); err != nil {
		t.Fatal(err)
	}
	_, _, err = ReadMessage(client) // AuthenticationSASL
	if err != nil {
		t.Fatal(err)
	}
	first := "n,,n=alice,r=wrong-password-nonce"
	initial := append([]byte("SCRAM-SHA-256\x00"), 0, 0, 0, 0)
	binary.BigEndian.PutUint32(initial[len("SCRAM-SHA-256")+1:], uint32(len(first)))
	initial = append(initial, first...)
	if err := WriteMessage(client, msgPassword, initial); err != nil {
		t.Fatal(err)
	}
	_, serverPayload, err := ReadMessage(client)
	if err != nil {
		t.Fatal(err)
	}
	serverFirst := string(serverPayload[4:])
	attrs, err := parseSCRAMAttributes(serverFirst)
	if err != nil {
		t.Fatal(err)
	}
	final := "c=biws,r=" + attrs['r'] + ",p=" + base64.StdEncoding.EncodeToString(make([]byte, 32))
	if err := WriteMessage(client, msgPassword, []byte(final)); err != nil {
		t.Fatal(err)
	}
	msgType, payload, err := ReadMessage(client)
	if err != nil {
		t.Fatal(err)
	}
	if msgType != msgErrorResponse || !strings.Contains(string(payload), "password authentication failed") || strings.Contains(string(payload), "SCRAM authentication failed") {
		t.Fatalf("expected SCRAM failure, type=%q payload=%q", msgType, payload)
	}
	if err := <-serverResult; err == nil {
		t.Fatal("wrong password unexpectedly authenticated")
	}
}

func TestServerSecurityConfigValidation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := validateTransportSecurity(ServerConfig{Addr: "0.0.0.0:5432"}); err == nil || !strings.Contains(err.Error(), "AllowInsecure") {
		t.Fatalf("expected public insecure listener rejection, got %v", err)
	}
	if err := validateTransportSecurity(ServerConfig{Addr: "0.0.0.0:5432", AllowInsecure: true}); err != nil {
		t.Fatalf("explicit insecure listener opt-in was rejected: %v", err)
	}
	if err := validateTransportSecurity(ServerConfig{Addr: "127.0.0.1:5432"}); err != nil {
		t.Fatalf("loopback development listener was rejected: %v", err)
	}
	if err := NewServer(nil, ServerConfig{Addr: "127.0.0.1:0", RequireTLS: true}).Serve(ctx); err == nil || !strings.Contains(err.Error(), "RequireTLS") {
		t.Fatalf("expected missing TLS configuration error, got %v", err)
	}
	cert := testTLSCertificate(t)
	if err := NewServer(nil, ServerConfig{Addr: "127.0.0.1:0", RequireAuthentication: true, RequireTLS: true, TLSConfig: &tls.Config{Certificates: []tls.Certificate{cert}}}).Serve(ctx); err == nil || !strings.Contains(err.Error(), "SCRAM") {
		t.Fatalf("expected missing SCRAM credential error, got %v", err)
	}
	credential, err := NewSCRAMCredential("alice", "secret")
	if err != nil {
		t.Fatal(err)
	}
	if err := NewServer(nil, ServerConfig{Addr: "127.0.0.1:0", Credentials: map[string]SCRAMCredential{"alice": credential}}).Serve(ctx); err == nil || !strings.Contains(err.Error(), "RequireTLS") {
		t.Fatalf("expected SCRAM plaintext rejection, got %v", err)
	}
	if _, err := NewSCRAMCredentialWithIterations("alice", "secret", DefaultSCRAMIterations-1); err == nil {
		t.Fatal("expected SCRAM iteration floor validation")
	}
}

func TestServerResourceLimitsHaveSecureDefaults(t *testing.T) {
	server := NewServer(nil, ServerConfig{})
	if server.connSem == nil || cap(server.connSem) != DefaultMaxConnections {
		t.Fatalf("expected default connection limit %d, got %d", DefaultMaxConnections, cap(server.connSem))
	}
	if server.authLimiter == nil || server.authLimiter.attemptPolicy.burst != DefaultAuthAttemptBurst || server.authLimiter.attemptPolicy.refill != DefaultAuthAttemptRefill || server.authLimiter.attemptPolicy.maxConcurrent != DefaultMaxConcurrentAuth {
		t.Fatalf("unexpected default authentication admission policy: %#v", server.authLimiter.attemptPolicy)
	}

	ctx := context.Background()
	if err := NewServer(nil, ServerConfig{MaxConnections: -1}).Serve(ctx); err == nil || !strings.Contains(err.Error(), "MaxConnections") {
		t.Fatalf("expected negative MaxConnections rejection, got %v", err)
	}
	if err := NewServer(nil, ServerConfig{StartupTimeout: -time.Second}).Serve(ctx); err == nil || !strings.Contains(err.Error(), "StartupTimeout") {
		t.Fatalf("expected negative StartupTimeout rejection, got %v", err)
	}
	if err := NewServer(nil, ServerConfig{IdleTimeout: -time.Second}).Serve(ctx); err == nil || !strings.Contains(err.Error(), "IdleTimeout") {
		t.Fatalf("expected negative IdleTimeout rejection, got %v", err)
	}
	if err := NewServer(nil, ServerConfig{MaxPreparedStatements: -1}).Serve(ctx); err == nil || !strings.Contains(err.Error(), "MaxPreparedStatements") {
		t.Fatalf("expected negative MaxPreparedStatements rejection, got %v", err)
	}
	if err := NewServer(nil, ServerConfig{MaxPortals: -1}).Serve(ctx); err == nil || !strings.Contains(err.Error(), "MaxPortals") {
		t.Fatalf("expected negative MaxPortals rejection, got %v", err)
	}
	if err := NewServer(nil, ServerConfig{MaxPreparedStatementBytes: -1}).Serve(ctx); err == nil || !strings.Contains(err.Error(), "MaxPreparedStatementBytes") {
		t.Fatalf("expected negative MaxPreparedStatementBytes rejection, got %v", err)
	}
	if err := NewServer(nil, ServerConfig{MaxPortalBytes: -1}).Serve(ctx); err == nil || !strings.Contains(err.Error(), "MaxPortalBytes") {
		t.Fatalf("expected negative MaxPortalBytes rejection, got %v", err)
	}
	if err := NewServer(nil, ServerConfig{AuthAttemptBurst: -1}).Serve(ctx); err == nil || !strings.Contains(err.Error(), "AuthAttemptBurst") {
		t.Fatalf("expected negative AuthAttemptBurst rejection, got %v", err)
	}
	if err := NewServer(nil, ServerConfig{AuthGlobalFailureLimit: -1}).Serve(ctx); err == nil || !strings.Contains(err.Error(), "AuthGlobalFailureLimit") {
		t.Fatalf("expected negative AuthGlobalFailureLimit rejection, got %v", err)
	}
	if err := NewServer(nil, ServerConfig{AuthGlobalFailureWindow: -time.Second}).Serve(ctx); err == nil || !strings.Contains(err.Error(), "AuthGlobalFailureWindow") {
		t.Fatalf("expected negative AuthGlobalFailureWindow rejection, got %v", err)
	}
	if err := NewServer(nil, ServerConfig{AuthAttemptRefill: -time.Second}).Serve(ctx); err == nil || !strings.Contains(err.Error(), "AuthAttemptRefill") {
		t.Fatalf("expected negative AuthAttemptRefill rejection, got %v", err)
	}
	if err := NewServer(nil, ServerConfig{MaxConcurrentAuth: -1}).Serve(ctx); err == nil || !strings.Contains(err.Error(), "MaxConcurrentAuth") {
		t.Fatalf("expected negative MaxConcurrentAuth rejection, got %v", err)
	}
}

func TestTLSConfigDefaultsToApprovedCipherSuites(t *testing.T) {
	server := NewServer(nil, ServerConfig{TLSConfig: &tls.Config{
		Certificates: []tls.Certificate{testTLSCertificate(t)},
	}})
	cfg, err := server.buildTLSConfig()
	if err != nil {
		t.Fatalf("build TLS config: %v", err)
	}
	want := approvedTLS12CipherSuites()
	if len(cfg.CipherSuites) != len(want) {
		t.Fatalf("unexpected default TLS 1.2 cipher suites: got %v, want %v", cfg.CipherSuites, want)
	}
	for i := range want {
		if cfg.CipherSuites[i] != want[i] {
			t.Fatalf("unexpected default TLS 1.2 cipher suites: got %v, want %v", cfg.CipherSuites, want)
		}
	}
	for _, suite := range cfg.CipherSuites {
		if !secureTLS12CipherSuite(suite) {
			t.Fatalf("default TLS config contains unapproved suite %d", suite)
		}
	}
}

func TestExtendedProtocolResourceLimits(t *testing.T) {
	state := newConnState()
	state.maxPreparedStatements = 1
	var output bytes.Buffer
	parsePayload := func(name, query string) []byte {
		payload := WriteNullTerminated(nil, name)
		payload = WriteNullTerminated(payload, query)
		return append(payload, 0, 0) // no parameter type OIDs
	}
	if err := handleParse(&output, state, parsePayload("first", "SELECT 1")); err != nil {
		t.Fatalf("first Parse failed: %v", err)
	}
	if len(state.prepared) != 1 {
		t.Fatalf("expected one prepared statement, got %d", len(state.prepared))
	}
	output.Reset()
	if err := handleParse(&output, state, parsePayload("second", "SELECT 2")); err != nil {
		t.Fatalf("limited Parse returned write error: %v", err)
	}
	if len(state.prepared) != 1 {
		t.Fatalf("prepared statement limit was bypassed: %d objects", len(state.prepared))
	}
	msgType, _, err := ReadMessage(bytes.NewReader(output.Bytes()))
	if err != nil || msgType != msgErrorResponse {
		t.Fatalf("expected ErrorResponse at prepared limit, type=%q err=%v", msgType, err)
	}

	output.Reset()
	state.maxPortals = 1
	bindPayload := func(name string) []byte {
		payload := WriteNullTerminated(nil, name)
		payload = WriteNullTerminated(payload, "first")
		payload = append(payload, 0, 0) // no parameter format codes
		payload = append(payload, 0, 0) // no parameters
		return append(payload, 0, 0)    // no result format codes
	}
	if err := handleBind(&output, state, bindPayload("first-portal")); err != nil {
		t.Fatalf("first Bind failed: %v", err)
	}
	output.Reset()
	if err := handleBind(&output, state, bindPayload("second-portal")); err != nil {
		t.Fatalf("limited Bind returned write error: %v", err)
	}
	if len(state.portals) != 1 {
		t.Fatalf("portal limit was bypassed: %d objects", len(state.portals))
	}
	msgType, _, err = ReadMessage(bytes.NewReader(output.Bytes()))
	if err != nil || msgType != msgErrorResponse {
		t.Fatalf("expected ErrorResponse at portal limit, type=%q err=%v", msgType, err)
	}

	output.Reset()
	if err := handleBind(&output, state, bindPayload("first-portal")); err != nil {
		t.Fatalf("existing portal replacement failed: %v", err)
	}
	if len(state.portals) != 1 {
		t.Fatalf("existing portal replacement changed object count: %d", len(state.portals))
	}
}

func TestServerIdleTimeoutClosesQuietSession(t *testing.T) {
	serverConn, clientConn := net.Pipe()
	defer clientConn.Close()
	done := make(chan struct{})
	server := NewServer(nil, ServerConfig{IdleTimeout: 25 * time.Millisecond})
	go func() {
		server.handleConn(context.Background(), serverConn, nil)
		close(done)
	}()

	if err := sendStartupPacket(clientConn, "idle-user", "test"); err != nil {
		t.Fatalf("send startup: %v", err)
	}
	assertStartupMessages(t, clientConn, 8)
	if err := clientConn.SetReadDeadline(time.Now().Add(time.Second)); err != nil {
		t.Fatalf("set client deadline: %v", err)
	}
	if _, _, err := ReadMessage(clientConn); err == nil {
		t.Fatal("quiet session remained open past idle timeout")
	}
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("idle session handler did not exit")
	}
}

func TestStartupParserRejectsAmbiguousOrTruncatedFields(t *testing.T) {
	valid := startupPayloadForTest("alice", "database", "test")
	result, err := parseStartupPayload(valid)
	if err != nil || result.User != "alice" || result.Database != "test" {
		t.Fatalf("valid startup rejected: result=%+v err=%v", result, err)
	}

	withoutTerminator := append([]byte(nil), valid[:len(valid)-1]...)
	if _, err := parseStartupPayload(withoutTerminator); err == nil {
		t.Fatal("startup without terminating NUL was accepted")
	}
	duplicate := startupProtocolForTest()
	duplicate = append(duplicate, "user\x00alice\x00user\x00mallory\x00\x00"...)
	if _, err := parseStartupPayload(duplicate); err == nil {
		t.Fatal("duplicate startup user was accepted")
	}
	missingUser := startupProtocolForTest()
	missingUser = append(missingUser, "database\x00test\x00\x00"...)
	if _, err := parseStartupPayload(missingUser); err == nil {
		t.Fatal("startup without user was accepted")
	}
	invalidUTF8 := startupProtocolForTest()
	invalidUTF8 = append(invalidUTF8, "user\x00"...)
	invalidUTF8 = append(invalidUTF8, 0xff, 0x00, 0x00)
	if _, err := parseStartupPayload(invalidUTF8); err == nil {
		t.Fatal("invalid UTF-8 startup user was accepted")
	}
}

func TestSCRAMNonceAcceptsEquals(t *testing.T) {
	if err := validateSCRAMNonce("client=nonce"); err != nil {
		t.Fatalf("RFC-valid SCRAM nonce was rejected: %v", err)
	}
	if _, err := parseSCRAMClientFirst("n,,n=alice,r=client=nonce"); err != nil {
		t.Fatalf("client-first nonce containing '=' was rejected: %v", err)
	}
}

func TestAuthFailureLimiterUsesPerClientAndGlobalBudgets(t *testing.T) {
	limiter := newAuthFailureLimiter(2, time.Minute)
	now := time.Now()
	if !limiter.allow("198.51.100.10", now) {
		t.Fatal("new client was unexpectedly rate limited")
	}
	limiter.recordFailure("198.51.100.10", now)
	limiter.recordFailure("198.51.100.10", now)
	if limiter.allow("198.51.100.10", now) {
		t.Fatal("client was not blocked after failure threshold")
	}
	if !limiter.allow("198.51.100.11", now) {
		t.Fatal("independent client was incorrectly blocked")
	}
	limiter.recordSuccess("198.51.100.10")
	if !limiter.allow("198.51.100.10", now) {
		t.Fatal("successful authentication did not clear client failure state")
	}

	limiter.globalFailures = globalAuthFailureLimit - 1
	limiter.globalWindowStart = now
	limiter.recordFailure("198.51.100.11", now)
	if limiter.allow("198.51.100.12", now) {
		t.Fatal("global authentication failure budget was not enforced")
	}
}

func TestSCRAMMessageLimitRejectsBeforePayloadRead(t *testing.T) {
	var header [5]byte
	header[0] = msgPassword
	binary.BigEndian.PutUint32(header[1:], uint32(4+maxSCRAMMessageBytes+1))
	wire := append(header[:], []byte("payload-not-read")...)
	reader := bytes.NewReader(wire)

	if _, _, err := readMessageWithMax(reader, maxSCRAMMessageBytes); err == nil {
		t.Fatal("oversized SCRAM message was accepted")
	}
	if got, want := reader.Len(), len("payload-not-read"); got != want {
		t.Fatalf("oversized message reader consumed payload: got %d bytes remaining, want %d", got, want)
	}
}

func TestStartupRejectsOversizedSSLRequest(t *testing.T) {
	payload := binary.BigEndian.AppendUint32(nil, uint32(sslRequestCode))
	payload = append(payload, "junk"...)
	wire := binary.BigEndian.AppendUint32(nil, uint32(4+len(payload)))
	wire = append(wire, payload...)
	rw := &startupTestReadWriter{reader: bytes.NewReader(wire)}

	if _, _, err := handleStartupWithConfig(rw, nil, nil, false, ServerConfig{}); err == nil {
		t.Fatal("oversized SSLRequest was accepted")
	}
	msgType, _, err := ReadMessage(bytes.NewReader(rw.writer.Bytes()))
	if err != nil {
		t.Fatalf("read SSLRequest error response: %v", err)
	}
	if msgType != msgErrorResponse {
		t.Fatalf("expected ErrorResponse for oversized SSLRequest, got %q", msgType)
	}
}

func TestStartupParserBoundsFieldAndParameterCounts(t *testing.T) {
	tooLong := startupProtocolForTest()
	tooLong = append(tooLong, "user\x00"...)
	tooLong = append(tooLong, strings.Repeat("u", maxStartupFieldBytes+1)...)
	tooLong = append(tooLong, 0, 0)
	if _, err := parseStartupPayload(tooLong); err == nil {
		t.Fatal("oversized startup field was accepted")
	}

	tooMany := startupProtocolForTest()
	tooMany = append(tooMany, "user\x00alice\x00"...)
	for i := 0; i < maxStartupParameters; i++ {
		tooMany = append(tooMany, fmt.Sprintf("parameter-%03d\x00x\x00", i)...)
	}
	tooMany = append(tooMany, 0)
	if _, err := parseStartupPayload(tooMany); err == nil {
		t.Fatal("excessive startup parameter count was accepted")
	}
}

func TestSCRAMUnknownCredentialIsStable(t *testing.T) {
	config := ServerConfig{
		RequireTLS:            true,
		RequireAuthentication: true,
		PasswordLookup: func(string) (SCRAMCredential, bool) {
			return SCRAMCredential{}, false
		},
	}
	if err := ensureSCRAMUnknownCredential(&config); err != nil {
		t.Fatalf("prepare unknown-user verifier: %v", err)
	}
	first := cloneSCRAMCredential(*config.scramUnknownCredential)
	if err := ensureSCRAMUnknownCredential(&config); err != nil {
		t.Fatalf("re-prepare unknown-user verifier: %v", err)
	}
	second := config.scramUnknownCredential
	if !bytes.Equal(first.Salt, second.Salt) || !bytes.Equal(first.StoredKey, second.StoredKey) || !bytes.Equal(first.ServerKey, second.ServerKey) {
		t.Fatal("unknown-user verifier changed across preparations")
	}
	exchangeOne, err := unknownSCRAMCredentialForExchange(first)
	if err != nil {
		t.Fatal(err)
	}
	exchangeTwo, err := unknownSCRAMCredentialForExchange(first)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Equal(exchangeOne.Salt, exchangeTwo.Salt) {
		t.Fatal("unknown-user SCRAM salt was stable across exchanges")
	}
	if !bytes.Equal(exchangeOne.StoredKey, exchangeTwo.StoredKey) || !bytes.Equal(exchangeOne.ServerKey, exchangeTwo.ServerKey) {
		t.Fatal("unknown-user verifier keys changed across exchanges")
	}
}

func TestSCRAMCredentialLookupRequiresMatchingUsername(t *testing.T) {
	credential, err := deriveSCRAMCredential("bob", "secret", []byte("fixed-scram-salt"), DefaultSCRAMIterations)
	if err != nil {
		t.Fatal(err)
	}
	config := ServerConfig{PasswordLookup: func(string) (SCRAMCredential, bool) {
		return credential, true
	}}
	if _, found := lookupSCRAMCredential(config, "alice"); found {
		t.Fatal("mismatched credential was accepted for the requested user")
	}
}

type startupTestReadWriter struct {
	reader *bytes.Reader
	writer bytes.Buffer
}

func (rw *startupTestReadWriter) Read(p []byte) (int, error) {
	return rw.reader.Read(p)
}

func (rw *startupTestReadWriter) Write(p []byte) (int, error) {
	return rw.writer.Write(p)
}

func TestTLSConfigRejectsDowngradeAndWeakSuites(t *testing.T) {
	cert := testTLSCertificate(t)
	weakVersion := NewServer(nil, ServerConfig{TLSConfig: &tls.Config{
		Certificates: []tls.Certificate{cert},
		MinVersion:   tls.VersionTLS11,
	}})
	if _, err := weakVersion.buildTLSConfig(); err == nil {
		t.Fatal("TLS 1.1 minimum was accepted")
	}
	weakCipher := NewServer(nil, ServerConfig{TLSConfig: &tls.Config{
		Certificates: []tls.Certificate{cert},
		CipherSuites: []uint16{tls.TLS_RSA_WITH_AES_128_CBC_SHA},
	}})
	if _, err := weakCipher.buildTLSConfig(); err == nil {
		t.Fatal("RSA/CBC cipher suite was accepted")
	}

	dynamic := NewServer(nil, ServerConfig{TLSConfig: &tls.Config{
		Certificates: []tls.Certificate{cert},
		GetConfigForClient: func(*tls.ClientHelloInfo) (*tls.Config, error) {
			return &tls.Config{Certificates: []tls.Certificate{cert}, MinVersion: tls.VersionTLS11}, nil
		},
	}})
	cfg, err := dynamic.buildTLSConfig()
	if err != nil {
		t.Fatalf("dynamic TLS config rejected before callback: %v", err)
	}
	if _, err := cfg.GetConfigForClient(&tls.ClientHelloInfo{}); err == nil {
		t.Fatal("dynamic callback bypassed TLS policy")
	}
}

func TestTLSConfigRejectsExpiredAndWeakCertificates(t *testing.T) {
	weakKey := testTLSCertificateWith(t, 1024, time.Now().Add(-time.Hour), time.Now().Add(time.Hour))
	if _, err := NewServer(nil, ServerConfig{TLSConfig: &tls.Config{Certificates: []tls.Certificate{weakKey}}}).buildTLSConfig(); err == nil {
		t.Fatal("1024-bit RSA certificate was accepted")
	}
	expired := testTLSCertificateWith(t, 2048, time.Now().Add(-2*time.Hour), time.Now().Add(-time.Hour))
	if _, err := NewServer(nil, ServerConfig{TLSConfig: &tls.Config{Certificates: []tls.Certificate{expired}}}).buildTLSConfig(); err == nil {
		t.Fatal("expired certificate was accepted")
	}
}

func TestSCRAMVerifierMutationProperty(t *testing.T) {
	for iteration := 0; iteration < 12; iteration++ {
		salt := make([]byte, 16)
		if _, err := rand.Read(salt); err != nil {
			t.Fatal(err)
		}
		credential, err := deriveSCRAMCredential("property-user", "property-password", salt, DefaultSCRAMIterations)
		if err != nil {
			t.Fatal(err)
		}
		authMessage := make([]byte, 97)
		if _, err := rand.Read(authMessage); err != nil {
			t.Fatal(err)
		}
		salted := testPBKDF2SHA256([]byte("property-password"), salt, credential.Iterations)
		clientKey := testHMACSHA256(salted, []byte("Client Key"))
		stored := sha256.Sum256(clientKey)
		clientSignature := testHMACSHA256(stored[:], authMessage)
		proof := make([]byte, len(clientKey))
		for i := range proof {
			proof[i] = clientKey[i] ^ clientSignature[i]
		}
		if !verifySCRAMProof(credential.StoredKey, authMessage, proof) {
			t.Fatalf("valid proof rejected at iteration %d", iteration)
		}
		for i := range proof {
			mutated := append([]byte(nil), proof...)
			mutated[i] ^= 1
			if verifySCRAMProof(credential.StoredKey, authMessage, mutated) {
				t.Fatalf("mutated proof accepted at iteration %d byte %d", iteration, i)
			}
		}
	}
}

func TestSCRAMCredentialOwnership(t *testing.T) {
	credential, err := deriveSCRAMCredential("alice", "secret", []byte("fixed-scram-salt"), DefaultSCRAMIterations)
	if err != nil {
		t.Fatal(err)
	}
	originalSalt := append([]byte(nil), credential.Salt...)
	originalStored := append([]byte(nil), credential.StoredKey...)
	originalServer := append([]byte(nil), credential.ServerKey...)
	server := NewServer(nil, ServerConfig{Credentials: map[string]SCRAMCredential{"alice": credential}})
	credential.Salt[0] ^= 0xff
	credential.StoredKey[0] ^= 0xff
	credential.ServerKey[0] ^= 0xff
	if got := server.config.Credentials["alice"]; !bytes.Equal(got.Salt, originalSalt) || !bytes.Equal(got.StoredKey, originalStored) || !bytes.Equal(got.ServerKey, originalServer) {
		t.Fatal("server retained caller-owned SCRAM credential memory")
	}
}

func TestSCRAMPasswordPreparationIsStable(t *testing.T) {
	salt := []byte("fixed-scram-salt")
	composed, err := deriveSCRAMCredential("alice", "\u00e5", salt, DefaultSCRAMIterations)
	if err != nil {
		t.Fatal(err)
	}
	decomposed, err := deriveSCRAMCredential("alice", "a\u030a", salt, DefaultSCRAMIterations)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(composed.StoredKey, decomposed.StoredKey) || !bytes.Equal(composed.ServerKey, decomposed.ServerKey) {
		t.Fatal("SASLprep-compatible password normalization was not stable")
	}
}

func TestServerContextCancellationClosesStartupConnections(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	server := NewServer(nil, ServerConfig{Addr: "127.0.0.1:0"})
	errCh := make(chan error, 1)
	go func() { errCh <- server.Serve(ctx) }()
	for i := 0; i < 100 && server.Addr() == ""; i++ {
		time.Sleep(time.Millisecond)
	}
	if server.Addr() == "" {
		t.Fatal("server did not start")
	}
	conn, err := net.DialTimeout("tcp", server.Addr(), time.Second)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	cancel()
	select {
	case err := <-errCh:
		if err != context.Canceled {
			t.Fatalf("Serve returned %v after context cancellation", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Serve did not close a blocked startup connection")
	}
}

func TestServerRequireTLSActualListener(t *testing.T) {
	cert := testTLSCertificate(t)
	server := NewServer(nil, ServerConfig{
		Addr:       "127.0.0.1:0",
		TLSConfig:  &tls.Config{Certificates: []tls.Certificate{cert}},
		RequireTLS: true,
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	errCh := make(chan error, 1)
	go func() { errCh <- server.Serve(ctx) }()
	for i := 0; i < 100 && server.Addr() == ""; i++ {
		time.Sleep(time.Millisecond)
	}
	if server.Addr() == "" {
		t.Fatal("server did not start")
	}
	conn, err := net.DialTimeout("tcp", server.Addr(), time.Second)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	writeSSLRequest(t, conn)
	var response [1]byte
	if _, err := io.ReadFull(conn, response[:]); err != nil || response[0] != 'S' {
		t.Fatalf("TLS negotiation failed: response=%q err=%v", response, err)
	}
	secure := tls.Client(conn, &tls.Config{InsecureSkipVerify: true, MinVersion: tls.VersionTLS12}) // self-signed test certificate
	if err := secure.Handshake(); err != nil {
		t.Fatal(err)
	}
	if err := sendStartupPacket(secure, "tls-user", "test"); err != nil {
		t.Fatal(err)
	}
	assertStartupMessages(t, secure, 8)
	cancel()
	server.Close()
	select {
	case <-errCh:
	case <-time.After(2 * time.Second):
		t.Fatal("TLS server did not stop")
	}
}

type startupTestError struct{ message string }

func (e *startupTestError) Error() string { return e.message }

func writeSSLRequest(t *testing.T, conn net.Conn) {
	t.Helper()
	var packet [8]byte
	binary.BigEndian.PutUint32(packet[:4], 8)
	binary.BigEndian.PutUint32(packet[4:], uint32(sslRequestCode))
	if _, err := conn.Write(packet[:]); err != nil {
		t.Fatalf("write SSLRequest: %v", err)
	}
}

func startupProtocolForTest() []byte {
	return binary.BigEndian.AppendUint32(nil, uint32(protocolVersion))
}

func startupPayloadForTest(user, key, value string) []byte {
	payload := startupProtocolForTest()
	payload = append(payload, "user\x00"...)
	payload = append(payload, user...)
	payload = append(payload, 0)
	payload = append(payload, key...)
	payload = append(payload, 0)
	payload = append(payload, value...)
	payload = append(payload, 0, 0)
	return payload
}

func assertStartupMessages(t *testing.T, reader io.Reader, count int) {
	t.Helper()
	for i := 0; i < count; i++ {
		if _, _, err := ReadMessage(reader); err != nil {
			t.Fatalf("startup message %d: %v", i, err)
		}
	}
}

func testTLSCertificate(t *testing.T) tls.Certificate {
	return testTLSCertificateWith(t, 2048, time.Now().Add(-time.Minute), time.Now().Add(time.Hour))
}

func testTLSCertificateWith(t *testing.T, bits int, notBefore, notAfter time.Time) tls.Certificate {
	t.Helper()
	key, err := rsa.GenerateKey(rand.Reader, bits)
	if err != nil {
		t.Fatal(err)
	}
	template := &x509.Certificate{
		SerialNumber:          big.NewInt(1),
		Subject:               pkix.Name{CommonName: "localhost"},
		NotBefore:             notBefore,
		NotAfter:              notAfter,
		KeyUsage:              x509.KeyUsageDigitalSignature | x509.KeyUsageKeyEncipherment,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		DNSNames:              []string{"localhost"},
		IPAddresses:           nil,
		BasicConstraintsValid: true,
	}
	der, err := x509.CreateCertificate(rand.Reader, template, template, &key.PublicKey, key)
	if err != nil {
		t.Fatal(err)
	}
	cert, err := x509.ParseCertificate(der)
	if err != nil {
		t.Fatal(err)
	}
	return tls.Certificate{Certificate: [][]byte{der}, PrivateKey: key, Leaf: cert}
}

// Keep the client proof calculation independent from the server helpers. A
// SCRAM test that calls scramHi/scramHMAC on both sides could pass while both
// implementations shared the same PBKDF2 or HMAC mistake.
func testPBKDF2SHA256(password, salt []byte, iterations int) []byte {
	input := make([]byte, len(salt)+4)
	copy(input, salt)
	binary.BigEndian.PutUint32(input[len(salt):], 1)
	h := hmac.New(sha256.New, password)
	_, _ = h.Write(input)
	u := h.Sum(nil)
	out := append([]byte(nil), u...)
	for i := 1; i < iterations; i++ {
		h = hmac.New(sha256.New, password)
		_, _ = h.Write(u)
		u = h.Sum(nil)
		for j := range out {
			out[j] ^= u[j]
		}
	}
	return out
}

func testHMACSHA256(key, data []byte) []byte {
	h := hmac.New(sha256.New, key)
	_, _ = h.Write(data)
	return h.Sum(nil)
}
