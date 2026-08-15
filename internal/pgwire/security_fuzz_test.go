package pgwire

import (
	"encoding/binary"
	"testing"
)

func FuzzParseStartupPayloadNeverPanics(f *testing.F) {
	valid := startupPayloadForTest("alice", "database", "test")
	f.Add(valid)
	f.Add([]byte{0, 0, 3, 0, 'u', 's', 'e', 'r'})
	f.Add([]byte{0xff, 0xff, 0xff, 0xff})
	f.Fuzz(func(t *testing.T, payload []byte) {
		_, _ = parseStartupPayload(payload)
	})
}

func FuzzSCRAMParsersNeverPanics(f *testing.F) {
	f.Add("n,,n=alice,r=client-nonce")
	f.Add("n,,n=alice,r=client-nonce,m=unsupported")
	f.Add("c=biws,r=server-nonce,p=AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=")
	f.Add("")
	f.Fuzz(func(t *testing.T, message string) {
		_, _ = parseSCRAMClientFirst(message)
		_, _ = parseSCRAMClientFinal(message)
		_, _ = parseSCRAMAttributes(message)
	})
}

func FuzzSASLInitialResponseNeverPanics(f *testing.F) {
	valid := append([]byte("SCRAM-SHA-256\x00"), 0, 0, 0, 0)
	binary.BigEndian.PutUint32(valid[len("SCRAM-SHA-256")+1:], uint32(len("n,,n=alice,r=nonce")))
	valid = append(valid, "n,,n=alice,r=nonce"...)
	f.Add(valid)
	f.Add([]byte{0, 0, 0, 0})
	f.Fuzz(func(t *testing.T, payload []byte) {
		_, _, _ = parseSASLInitialResponse(payload)
	})
}
