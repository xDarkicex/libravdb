package pgwire

import (
	"bytes"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"encoding/base64"
	"encoding/binary"
	"io"
	"net"
	"strconv"
	"strings"
	"testing"
)

func TestSCRAMCredentialRejectsInvalidPRECISPassword(t *testing.T) {
	if _, err := NewSCRAMCredential("alice", "bad\x00password"); err == nil {
		t.Fatal("invalid PRECIS password was accepted")
	}
	if _, err := NewSCRAMCredential("alice", string([]byte{0xff, 0xfe})); err == nil {
		t.Fatal("invalid UTF-8 password was accepted")
	}
}

func TestSCRAMSHA256PlusProofAndBinding(t *testing.T) {
	credential, err := deriveSCRAMCredential("alice", "correct horse battery staple", []byte("fixed-scram-salt"), DefaultSCRAMIterations)
	if err != nil {
		t.Fatal(err)
	}
	binding := []byte("test-tls-server-end-point-binding")
	server, client := net.Pipe()
	defer client.Close()
	defer server.Close()
	config := ServerConfig{
		Credentials:           map[string]SCRAMCredential{"alice": credential},
		scramChannelBinding:   binding,
		scramChannelBindingOK: true,
	}
	result := make(chan error, 1)
	go func() { result <- authenticateSCRAM(server, "alice", config) }()

	msgType, payload, err := ReadMessage(client)
	if err != nil || msgType != msgAuth || len(payload) < 4 || binary.BigEndian.Uint32(payload[:4]) != 10 {
		t.Fatalf("expected AuthenticationSASL, type=%q err=%v payload=%x", msgType, err, payload)
	}
	mechanisms := string(payload[4:])
	if !strings.Contains(mechanisms, scramSHA256PlusMechanism+"\x00") || !strings.Contains(mechanisms, scramSHA256Mechanism+"\x00") {
		t.Fatalf("server did not advertise both SCRAM mechanisms: %q", mechanisms)
	}

	nonce := "client-plus-nonce"
	gs2Header := "p=" + scramTLSBindingType + ",,"
	clientFirst := gs2Header + "n=alice,r=" + nonce
	initial := append([]byte(scramSHA256PlusMechanism+"\x00"), 0, 0, 0, 0)
	binary.BigEndian.PutUint32(initial[len(scramSHA256PlusMechanism)+1:], uint32(len(clientFirst)))
	initial = append(initial, clientFirst...)
	if err := WriteMessage(client, msgPassword, initial); err != nil {
		t.Fatalf("send SCRAM-PLUS client-first: %v", err)
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
	if err != nil || iterations < DefaultSCRAMIterations {
		t.Fatalf("parse iterations: %v", err)
	}
	channelBinding := append([]byte(gs2Header), binding...)
	clientFinalWithoutProof := "c=" + base64.StdEncoding.EncodeToString(channelBinding) + ",r=" + serverNonce
	authMessage := clientFirst[len(gs2Header):] + "," + serverFirst + "," + clientFinalWithoutProof
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
		t.Fatalf("send SCRAM-PLUS client-final: %v", err)
	}
	msgType, payload, err = ReadMessage(client)
	if err != nil || msgType != msgAuth || len(payload) < 4 || binary.BigEndian.Uint32(payload[:4]) != 12 {
		t.Fatalf("expected AuthenticationSASLFinal, type=%q err=%v", msgType, err)
	}
	msgType, payload, err = ReadMessage(client)
	if err != nil || msgType != msgAuth || len(payload) < 4 || binary.BigEndian.Uint32(payload[:4]) != 0 {
		t.Fatalf("expected AuthenticationOk, type=%q err=%v payload=%x", msgType, err, payload)
	}
	if err := <-result; err != nil {
		t.Fatalf("SCRAM-PLUS authentication failed: %v", err)
	}
}

func TestSCRAMRequireChannelBindingRejectsClassicMechanism(t *testing.T) {
	credential, err := deriveSCRAMCredential("alice", "secret", []byte("fixed-scram-salt"), DefaultSCRAMIterations)
	if err != nil {
		t.Fatal(err)
	}
	server, client := net.Pipe()
	defer client.Close()
	defer server.Close()
	config := ServerConfig{
		RequireChannelBinding: true,
		Credentials:           map[string]SCRAMCredential{"alice": credential},
		scramChannelBinding:   []byte("binding"),
		scramChannelBindingOK: true,
	}
	result := make(chan error, 1)
	go func() { result <- authenticateSCRAM(server, "alice", config) }()
	if _, _, err := ReadMessage(client); err != nil {
		t.Fatal(err)
	}
	first := "n,,n=alice,r=nonce"
	initial := append([]byte(scramSHA256Mechanism+"\x00"), 0, 0, 0, 0)
	binary.BigEndian.PutUint32(initial[len(scramSHA256Mechanism)+1:], uint32(len(first)))
	initial = append(initial, first...)
	if err := WriteMessage(client, msgPassword, initial); err != nil {
		t.Fatal(err)
	}
	if err := <-result; err == nil || !strings.Contains(err.Error(), "required") {
		t.Fatalf("expected required channel-binding error, got %v", err)
	}
}

func TestTLSServerEndPointBindingUsesCertificateSignatureHash(t *testing.T) {
	cert := testTLSCertificate(t)
	cfg := &tls.Config{Certificates: []tls.Certificate{cert}}
	got, ok, err := tlsServerEndPointBinding(cfg)
	if err != nil || !ok {
		t.Fatalf("expected static TLS channel binding, ok=%v err=%v", ok, err)
	}
	parsed, err := x509.ParseCertificate(cert.Certificate[0])
	if err != nil {
		t.Fatal(err)
	}
	want := sha256.Sum256(cert.Certificate[0])
	if !bytes.Equal(got, want[:]) {
		t.Fatalf("unexpected tls-server-end-point binding for %v certificate", parsed.SignatureAlgorithm)
	}
}

func TestStartupTLSAdvertisesSCRAMPlus(t *testing.T) {
	credential, err := deriveSCRAMCredential("alice", "secret", []byte("fixed-scram-salt"), DefaultSCRAMIterations)
	if err != nil {
		t.Fatal(err)
	}
	cert := testTLSCertificate(t)
	server, client := net.Pipe()
	serverResult := make(chan error, 1)
	go func() {
		_, _, err := handleStartupWithConfig(server, nil, &tls.Config{Certificates: []tls.Certificate{cert}}, true, ServerConfig{
			RequireTLS:  true,
			Credentials: map[string]SCRAMCredential{"alice": credential},
		})
		serverResult <- err
	}()
	defer client.Close()
	defer server.Close()

	writeSSLRequest(t, client)
	var response [1]byte
	if _, err := io.ReadFull(client, response[:]); err != nil {
		t.Fatal(err)
	}
	if response[0] != 'S' {
		t.Fatalf("expected TLS acceptance, got %q", response[0])
	}
	secure := tls.Client(client, &tls.Config{InsecureSkipVerify: true, MinVersion: tls.VersionTLS12})
	if err := secure.Handshake(); err != nil {
		t.Fatal(err)
	}
	if err := sendStartupPacket(secure, "alice", "test"); err != nil {
		t.Fatal(err)
	}
	msgType, payload, err := ReadMessage(secure)
	if err != nil || msgType != msgAuth || len(payload) < 4 || binary.BigEndian.Uint32(payload[:4]) != 10 {
		t.Fatalf("expected AuthenticationSASL, type=%q err=%v", msgType, err)
	}
	if !strings.Contains(string(payload[4:]), scramSHA256PlusMechanism+"\x00") {
		t.Fatalf("TLS startup did not advertise SCRAM-SHA-256-PLUS: %q", payload[4:])
	}
	_ = secure.Close()
	if err := <-serverResult; err == nil {
		t.Fatal("expected incomplete authentication to terminate startup")
	}
}
