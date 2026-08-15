package pgwire

import (
	"bytes"
	"context"
	"crypto/tls"
	"io"
	"net"
	"testing"
	"time"
)

type proxyTestConn struct {
	reader *bytes.Reader
	remote net.Addr
	writer bytes.Buffer
}

func (c *proxyTestConn) Read(p []byte) (int, error)  { return c.reader.Read(p) }
func (c *proxyTestConn) Write(p []byte) (int, error) { return c.writer.Write(p) }
func (c *proxyTestConn) Close() error                { return nil }
func (c *proxyTestConn) LocalAddr() net.Addr {
	return &net.TCPAddr{IP: net.ParseIP("192.0.2.10"), Port: 5432}
}
func (c *proxyTestConn) RemoteAddr() net.Addr             { return c.remote }
func (c *proxyTestConn) SetDeadline(time.Time) error      { return nil }
func (c *proxyTestConn) SetReadDeadline(time.Time) error  { return nil }
func (c *proxyTestConn) SetWriteDeadline(time.Time) error { return nil }

func TestPROXYv1TrustedIdentityAndReadBoundary(t *testing.T) {
	networks, err := parseTrustedProxyCIDRs([]string{"192.0.2.0/24"})
	if err != nil {
		t.Fatal(err)
	}
	wire := []byte("PROXY TCP4 203.0.113.9 192.0.2.10 42310 5432\r\nSSLRequest")
	conn := &proxyTestConn{
		reader: bytes.NewReader(wire),
		remote: &net.TCPAddr{IP: net.ParseIP("192.0.2.44"), Port: 40000},
	}
	addr, err := readPROXYv1(conn, networks)
	if err != nil {
		t.Fatalf("read PROXY header: %v", err)
	}
	got, ok := addr.(*net.TCPAddr)
	if !ok || !got.IP.Equal(net.ParseIP("203.0.113.9")) || got.Port != 42310 {
		t.Fatalf("unexpected forwarded identity: %#v", addr)
	}
	remaining, _ := ioReadAll(conn.reader)
	if string(remaining) != "SSLRequest" {
		t.Fatalf("header parser consumed following startup bytes: %q", remaining)
	}
}

func TestPROXYv1RejectsUntrustedPeer(t *testing.T) {
	networks, err := parseTrustedProxyCIDRs([]string{"192.0.2.0/24"})
	if err != nil {
		t.Fatal(err)
	}
	conn := &proxyTestConn{
		reader: bytes.NewReader([]byte("PROXY UNKNOWN\r\n")),
		remote: &net.TCPAddr{IP: net.ParseIP("198.51.100.10"), Port: 40000},
	}
	if _, err := readPROXYv1(conn, networks); err == nil {
		t.Fatal("untrusted peer was allowed to provide a PROXY identity")
	}
}

func TestPROXYv1UnknownPreservesProxyAddress(t *testing.T) {
	networks, err := parseTrustedProxyCIDRs([]string{"192.0.2.0/24"})
	if err != nil {
		t.Fatal(err)
	}
	remote := &net.TCPAddr{IP: net.ParseIP("192.0.2.44"), Port: 40000}
	conn := &proxyTestConn{reader: bytes.NewReader([]byte("PROXY UNKNOWN\r\n")), remote: remote}
	addr, err := readPROXYv1(conn, networks)
	if err != nil {
		t.Fatal(err)
	}
	if addr != remote {
		t.Fatalf("PROXY UNKNOWN did not preserve proxy identity: got %#v", addr)
	}
}

func TestPROXYConfigRequiresExplicitTrustedCIDRs(t *testing.T) {
	ctx := context.Background()
	if err := NewServer(nil, ServerConfig{ProxyProtocol: true}).Serve(ctx); err == nil {
		t.Fatal("PROXY protocol enabled without trusted CIDRs")
	}
	if err := NewServer(nil, ServerConfig{TrustedProxyCIDRs: []string{"192.0.2.0/24"}}).Serve(ctx); err == nil {
		t.Fatal("trusted CIDRs accepted while PROXY protocol was disabled")
	}
	if err := NewServer(nil, ServerConfig{ProxyProtocol: true, TrustedProxyCIDRs: []string{"not-a-cidr"}}).Serve(ctx); err == nil {
		t.Fatal("invalid trusted proxy CIDR accepted")
	}
}

func TestPROXYv1ActualTLSStartup(t *testing.T) {
	cert := testTLSCertificate(t)
	server := NewServer(nil, ServerConfig{
		Addr:              "127.0.0.1:0",
		TLSConfig:         &tls.Config{Certificates: []tls.Certificate{cert}},
		RequireTLS:        true,
		ProxyProtocol:     true,
		TrustedProxyCIDRs: []string{"127.0.0.0/8"},
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	errCh := make(chan error, 1)
	go func() { errCh <- server.Serve(ctx) }()
	for i := 0; i < 100 && server.Addr() == ""; i++ {
		time.Sleep(time.Millisecond)
	}
	if server.Addr() == "" {
		t.Fatal("PROXY-enabled server did not start")
	}
	conn, err := net.DialTimeout("tcp", server.Addr(), time.Second)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	if _, err := io.WriteString(conn, "PROXY TCP4 203.0.113.9 127.0.0.1 40000 5432\r\n"); err != nil {
		t.Fatal(err)
	}
	writeSSLRequest(t, conn)
	var response [1]byte
	if _, err := io.ReadFull(conn, response[:]); err != nil || response[0] != 'S' {
		t.Fatalf("PROXY/TLS negotiation failed: response=%q err=%v", response, err)
	}
	secure := tls.Client(conn, &tls.Config{InsecureSkipVerify: true, MinVersion: tls.VersionTLS12})
	if err := secure.Handshake(); err != nil {
		t.Fatal(err)
	}
	if err := sendStartupPacket(secure, "proxy-user", "test"); err != nil {
		t.Fatal(err)
	}
	assertStartupMessages(t, secure, 6)
	cancel()
	server.Close()
	select {
	case <-errCh:
	case <-time.After(2 * time.Second):
		t.Fatal("PROXY-enabled server did not stop")
	}
}

func TestAuthMetricsReflectStartupEvents(t *testing.T) {
	credential, err := deriveSCRAMCredential("alice", "secret", []byte("fixed-scram-salt"), DefaultSCRAMIterations)
	if err != nil {
		t.Fatal(err)
	}
	metrics := &authMetrics{}
	config := ServerConfig{
		RequireAuthentication: true,
		Credentials:           map[string]SCRAMCredential{"alice": credential},
		authLimiter:           newAuthFailureLimiter(5, time.Minute),
		authMetrics:           metrics,
		authClientKey:         "198.51.100.30",
	}
	rw := &startupTestReadWriter{reader: bytes.NewReader(malformedAuthStartupWire())}
	if _, _, err := handleStartupWithConfig(rw, nil, nil, false, config); err == nil {
		t.Fatal("malformed authentication unexpectedly succeeded")
	}
	stats := metrics.snapshot()
	if stats.Attempts != 1 || stats.Failures != 1 || stats.Successes != 0 || stats.ActiveSCRAM != 0 {
		t.Fatalf("unexpected auth metrics: %+v", stats)
	}
}

func ioReadAll(reader *bytes.Reader) ([]byte, error) {
	remaining := make([]byte, reader.Len())
	_, err := reader.Read(remaining)
	return remaining, err
}
