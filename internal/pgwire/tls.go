package pgwire

import (
	"crypto/ecdsa"
	"crypto/ed25519"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/sha512"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"time"
)

const minimumTLSVersion = tls.VersionTLS12

// tlsServerEndPointBinding returns the RFC 5929 tls-server-end-point channel
// binding for the statically configured leaf certificate. The server side of
// crypto/tls exposes the peer certificate, not its own certificate, so a
// dynamic certificate callback cannot be used safely here without plumbing
// the selected certificate through the handshake. Such configurations simply
// do not advertise SCRAM-SHA-256-PLUS (and cannot satisfy RequireChannelBinding).
func tlsServerEndPointBinding(cfg *tls.Config) ([]byte, bool, error) {
	if cfg == nil || cfg.GetCertificate != nil || cfg.GetConfigForClient != nil || len(cfg.Certificates) == 0 || len(cfg.Certificates[0].Certificate) == 0 {
		return nil, false, nil
	}
	leaf, err := x509.ParseCertificate(cfg.Certificates[0].Certificate[0])
	if err != nil {
		return nil, false, fmt.Errorf("parse TLS server certificate for channel binding: %w", err)
	}
	certificateDER := cfg.Certificates[0].Certificate[0]
	var binding []byte
	switch leaf.SignatureAlgorithm {
	case x509.MD2WithRSA, x509.MD5WithRSA, x509.SHA1WithRSA,
		x509.DSAWithSHA1, x509.ECDSAWithSHA1:
		sum := sha256.Sum256(certificateDER)
		binding = sum[:]
	case x509.SHA256WithRSA, x509.SHA256WithRSAPSS,
		x509.ECDSAWithSHA256, x509.DSAWithSHA256:
		sum := sha256.Sum256(certificateDER)
		binding = sum[:]
	case x509.SHA384WithRSA, x509.SHA384WithRSAPSS, x509.ECDSAWithSHA384:
		sum := sha512.Sum384(certificateDER)
		binding = sum[:]
	case x509.SHA512WithRSA, x509.SHA512WithRSAPSS, x509.ECDSAWithSHA512:
		sum := sha512.Sum512(certificateDER)
		binding = sum[:]
	default:
		return nil, false, fmt.Errorf("unsupported TLS certificate signature algorithm %v for channel binding", leaf.SignatureAlgorithm)
	}
	return append([]byte(nil), binding...), true, nil
}

func (s *Server) buildTLSConfig() (*tls.Config, error) {
	if s.config.TLSConfig == nil && (s.config.TLSCertificateFile != "" || s.config.TLSKeyFile != "") {
		s.config.TLSConfig = &tls.Config{}
	}
	if s.config.TLSConfig == nil {
		if s.config.RequireTLS {
			return nil, fmt.Errorf("pgwire RequireTLS is enabled but no TLS configuration was provided")
		}
		return nil, nil
	}
	if (s.config.TLSCertificateFile == "") != (s.config.TLSKeyFile == "") {
		return nil, fmt.Errorf("TLS certificate and key files must be provided together")
	}

	cfg := s.config.TLSConfig.Clone()
	if cfg.MinVersion == 0 {
		cfg.MinVersion = minimumTLSVersion
	}
	if s.config.TLSCertificateFile != "" {
		cert, err := tls.LoadX509KeyPair(s.config.TLSCertificateFile, s.config.TLSKeyFile)
		if err != nil {
			return nil, fmt.Errorf("load pgwire TLS certificate: %w", err)
		}
		cfg.Certificates = append(cfg.Certificates, cert)
	}
	if len(cfg.Certificates) == 0 && cfg.GetCertificate == nil && cfg.GetConfigForClient == nil {
		return nil, fmt.Errorf("pgwire TLS configuration has no server certificate")
	}
	if err := hardenTLSConfig(cfg); err != nil {
		return nil, err
	}
	return cfg, nil
}

func hardenTLSConfig(cfg *tls.Config) error {
	if cfg == nil {
		return fmt.Errorf("nil TLS configuration")
	}
	if cfg.MinVersion == 0 {
		cfg.MinVersion = minimumTLSVersion
	}
	if cfg.MinVersion < minimumTLSVersion {
		return fmt.Errorf("pgwire TLS minimum version must be TLS 1.2 or newer")
	}
	if cfg.MaxVersion != 0 && cfg.MaxVersion < cfg.MinVersion {
		return fmt.Errorf("pgwire TLS maximum version is below minimum version")
	}
	if cfg.Renegotiation != tls.RenegotiateNever {
		return fmt.Errorf("pgwire TLS renegotiation must be disabled")
	}
	if len(cfg.Certificates) == 0 && cfg.GetCertificate == nil && cfg.GetConfigForClient == nil {
		return fmt.Errorf("pgwire TLS configuration has no server certificate")
	}
	if len(cfg.CipherSuites) == 0 {
		// Go's default TLS 1.2 preference list may include CBC suites. Set an
		// explicit AEAD-only list so the wire server's policy does not depend on
		// toolchain defaults or GODEBUG settings. TLS 1.3 suites are selected
		// independently by crypto/tls and are already AEAD-only.
		cfg.CipherSuites = approvedTLS12CipherSuites()
	}
	for _, suite := range cfg.CipherSuites {
		if !secureTLS12CipherSuite(suite) {
			return fmt.Errorf("pgwire TLS cipher suite %d is not an approved TLS 1.2 suite", suite)
		}
	}
	for i, certificate := range cfg.Certificates {
		if err := validateServerCertificate(certificate); err != nil {
			return fmt.Errorf("invalid pgwire TLS certificate %d: %w", i, err)
		}
	}
	if callback := cfg.GetCertificate; callback != nil {
		cfg.GetCertificate = func(hello *tls.ClientHelloInfo) (*tls.Certificate, error) {
			certificate, err := callback(hello)
			if err != nil {
				return nil, err
			}
			if certificate == nil {
				return nil, fmt.Errorf("TLS certificate callback returned nil")
			}
			if err := validateServerCertificate(*certificate); err != nil {
				return nil, err
			}
			return certificate, nil
		}
	}

	// A dynamic certificate callback is allowed, but its returned config must
	// obey the same policy. Remove the callback from the selected clone to
	// prevent recursive callback chains from bypassing validation.
	if callback := cfg.GetConfigForClient; callback != nil {
		cfg.GetConfigForClient = func(hello *tls.ClientHelloInfo) (*tls.Config, error) {
			selected, err := callback(hello)
			if err != nil || selected == nil {
				return selected, err
			}
			selected = selected.Clone()
			selected.GetConfigForClient = nil
			if err := hardenTLSConfig(selected); err != nil {
				return nil, err
			}
			return selected, nil
		}
	}
	return nil
}

func approvedTLS12CipherSuites() []uint16 {
	return []uint16{
		tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
		tls.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
		tls.TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256,
		tls.TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
		tls.TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
		tls.TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256,
	}
}

func secureTLS12CipherSuite(suite uint16) bool {
	switch suite {
	case tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
		tls.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
		tls.TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256,
		tls.TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
		tls.TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
		tls.TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256:
		return true
	default:
		return false
	}
}

func validateServerCertificate(certificate tls.Certificate) error {
	if len(certificate.Certificate) == 0 {
		return fmt.Errorf("certificate chain is empty")
	}
	if certificate.PrivateKey == nil {
		return fmt.Errorf("certificate private key is missing")
	}
	leaf, err := x509.ParseCertificate(certificate.Certificate[0])
	if err != nil {
		return fmt.Errorf("parse leaf certificate: %w", err)
	}
	now := time.Now()
	if now.Before(leaf.NotBefore) || now.After(leaf.NotAfter) {
		return fmt.Errorf("leaf certificate is outside its validity period")
	}
	switch key := leaf.PublicKey.(type) {
	case *rsa.PublicKey:
		if key.Size() < 256 { // 2048-bit RSA minimum.
			return fmt.Errorf("RSA leaf key is smaller than 2048 bits")
		}
	case *ecdsa.PublicKey:
		if key.Curve == nil || key.Curve.Params().BitSize < 256 {
			return fmt.Errorf("ECDSA leaf key is smaller than 256 bits")
		}
	case ed25519.PublicKey:
		// Ed25519 has a fixed secure key size.
	default:
		return fmt.Errorf("unsupported leaf public-key type %T", leaf.PublicKey)
	}
	return nil
}
