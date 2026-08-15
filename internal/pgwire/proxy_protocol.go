package pgwire

import (
	"bytes"
	"fmt"
	"io"
	"net"
	"strconv"
)

const maxPROXYv1HeaderBytes = 108

// proxyConn preserves the original socket while presenting the authenticated
// source address declared by a trusted PROXY-protocol frontend.
type proxyConn struct {
	net.Conn
	remote net.Addr
}

func (c *proxyConn) RemoteAddr() net.Addr { return c.remote }

func parseTrustedProxyCIDRs(values []string) ([]*net.IPNet, error) {
	if len(values) == 0 {
		return nil, fmt.Errorf("PROXY protocol requires at least one trusted proxy CIDR")
	}
	networks := make([]*net.IPNet, 0, len(values))
	for _, value := range values {
		_, network, err := net.ParseCIDR(value)
		if err != nil {
			return nil, fmt.Errorf("invalid trusted proxy CIDR %q: %w", value, err)
		}
		networks = append(networks, network)
	}
	return networks, nil
}

func trustedProxyPeer(addr net.Addr, networks []*net.IPNet) bool {
	if addr == nil {
		return false
	}
	host, _, err := net.SplitHostPort(addr.String())
	if err != nil {
		return false
	}
	ip := net.ParseIP(host)
	if ip == nil {
		return false
	}
	for _, network := range networks {
		if network.Contains(ip) {
			return true
		}
	}
	return false
}

// readPROXYv1 consumes exactly one bounded PROXY v1 line. It must be called
// before SSLRequest/TLS negotiation. Only peers inside trustedNetworks may
// provide a forwarded identity; direct clients cannot spoof it.
func readPROXYv1(conn net.Conn, trustedNetworks []*net.IPNet) (net.Addr, error) {
	if conn == nil {
		return nil, fmt.Errorf("nil connection")
	}
	if !trustedProxyPeer(conn.RemoteAddr(), trustedNetworks) {
		return nil, fmt.Errorf("PROXY protocol peer is not trusted")
	}
	header := make([]byte, 0, maxPROXYv1HeaderBytes)
	var one [1]byte
	for len(header) < maxPROXYv1HeaderBytes {
		n, err := conn.Read(one[:])
		if n > 0 {
			header = append(header, one[0])
			if len(header) >= 2 && header[len(header)-2] == '\r' && header[len(header)-1] == '\n' {
				return parsePROXYv1Line(header[:len(header)-2], conn.RemoteAddr())
			}
		}
		if err != nil {
			if err == io.EOF {
				return nil, fmt.Errorf("truncated PROXY protocol header")
			}
			return nil, fmt.Errorf("read PROXY protocol header: %w", err)
		}
		if n == 0 {
			return nil, fmt.Errorf("empty PROXY protocol read")
		}
	}
	return nil, fmt.Errorf("PROXY protocol header exceeds %d bytes", maxPROXYv1HeaderBytes)
}

func parsePROXYv1Line(line []byte, fallback net.Addr) (net.Addr, error) {
	fields := bytes.Fields(line)
	if len(fields) == 2 && bytes.Equal(fields[0], []byte("PROXY")) && bytes.Equal(fields[1], []byte("UNKNOWN")) {
		return fallback, nil
	}
	if len(fields) != 6 || !bytes.Equal(fields[0], []byte("PROXY")) {
		return nil, fmt.Errorf("invalid PROXY protocol header")
	}
	protocol := string(fields[1])
	sourceIP := net.ParseIP(string(fields[2]))
	destinationIP := net.ParseIP(string(fields[3]))
	if sourceIP == nil || destinationIP == nil {
		return nil, fmt.Errorf("invalid PROXY protocol address")
	}
	if protocol == "TCP4" {
		if sourceIP.To4() == nil || destinationIP.To4() == nil {
			return nil, fmt.Errorf("TCP4 PROXY protocol address is not IPv4")
		}
		sourceIP = sourceIP.To4()
	} else if protocol == "TCP6" {
		if sourceIP.To16() == nil || destinationIP.To16() == nil || sourceIP.To4() != nil || destinationIP.To4() != nil {
			return nil, fmt.Errorf("TCP6 PROXY protocol address is not IPv6")
		}
		sourceIP = sourceIP.To16()
	} else {
		return nil, fmt.Errorf("unsupported PROXY protocol transport %q", protocol)
	}
	sourcePort, err := parseProxyPort(fields[4])
	if err != nil {
		return nil, fmt.Errorf("invalid PROXY source port: %w", err)
	}
	if _, err := parseProxyPort(fields[5]); err != nil {
		return nil, fmt.Errorf("invalid PROXY destination port: %w", err)
	}
	return &net.TCPAddr{IP: sourceIP, Port: sourcePort}, nil
}

func parseProxyPort(value []byte) (int, error) {
	port, err := strconv.ParseUint(string(value), 10, 16)
	if err != nil {
		return 0, err
	}
	return int(port), nil
}
