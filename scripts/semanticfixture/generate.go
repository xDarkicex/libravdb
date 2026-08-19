//go:build ignore

// Generate a real Nomic 768d HNSW scale fixture from LongMemEval text.
// Run this file from the sibling libravdbd module so its embedding engine and
// local GGUF assets are available; the generated binary remains outside Git.
package main

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"strings"
	"sync"
	"time"

	apexjson "github.com/xDarkicex/apexJSON/v2"
	"github.com/zephyr-systems/libravdbd/embed"
)

const headerBytes = 64

var magic = [8]byte{'L', 'V', 'S', 'E', 'M', '0', '0', '1'}

type message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type item struct {
	Question         string      `json:"question"`
	HaystackSessions [][]message `json:"haystack_sessions"`
}

func main() {
	corpus := flag.String("corpus", "", "LongMemEval JSON corpus")
	backend := flag.String("backend", "onnx-local", "embedding backend: onnx-local or gguf")
	model := flag.String("model", "", "Nomic model directory or GGUF file")
	runtimePath := flag.String("runtime", "", "ONNX runtime library")
	llamaLib := flag.String("llama-lib", "", "llama.cpp shared library")
	output := flag.String("output", "", "output semantic fixture")
	vectorCount := flag.Int("vectors", 50000, "document vector count")
	queryCount := flag.Int("queries", 100, "query vector count")
	maxChars := flag.Int("max-chars", 512, "maximum Unicode characters per document chunk")
	flag.Parse()
	if *corpus == "" || *model == "" || *output == "" || (*backend == "onnx-local" && *runtimePath == "") {
		flag.Usage()
		os.Exit(2)
	}

	documents, queries, err := loadTexts(*corpus, *vectorCount, *queryCount, *maxChars)
	if err != nil {
		panic(err)
	}
	fmt.Printf("loaded %d unique documents and %d queries\n", len(documents), len(queries))

	engine := embed.NewWithConfig(embed.Config{
		Backend:      *backend,
		Profile:      "nomic-embed-text-v1.5",
		RuntimePath:  *runtimePath,
		ModelPath:    *model,
		LlamaLibPath: *llamaLib,
		Dimensions:   768,
		Normalize:    true,
	})
	if !engine.Ready() {
		panic("embedding engine unavailable: " + engine.Reason())
	}

	out, err := os.Create(*output)
	if err != nil {
		panic(err)
	}
	defer out.Close()

	modelFile := *model
	if *backend == "onnx-local" {
		modelFile += "/model.onnx"
	}
	modelTag, err := hashModel(modelFile)
	if err != nil {
		panic(err)
	}
	header := make([]byte, headerBytes)
	copy(header, magic[:])
	binary.LittleEndian.PutUint32(header[8:12], 768)
	binary.LittleEndian.PutUint32(header[12:16], uint32(len(documents)))
	binary.LittleEndian.PutUint32(header[16:20], uint32(len(queries)))
	binary.LittleEndian.PutUint64(header[24:32], modelTag)
	if _, err := out.Write(header); err != nil {
		panic(err)
	}

	ctx := context.Background()
	started := time.Now()
	const batchSize = 8
	for start := 0; start < len(documents); start += batchSize {
		end := min(start+batchSize, len(documents))
		vectors, err := embedDocuments(ctx, engine, documents[start:end], *backend)
		if err != nil {
			panic(fmt.Errorf("embed documents %d:%d: %w", start, end, err))
		}
		for i, vector := range vectors {
			if err := writeVector(out, vector); err != nil {
				panic(fmt.Errorf("write document %d: %w", start+i, err))
			}
		}
		if end%1000 == 0 || end == len(documents) {
			elapsed := time.Since(started)
			fmt.Printf("documents %d/%d (%.1f/s, %s)\n", end, len(documents), float64(end)/elapsed.Seconds(), elapsed.Round(time.Second))
		}
	}

	for i, query := range queries {
		vector, err := engine.EmbedQuery(ctx, query)
		if err != nil {
			panic(fmt.Errorf("embed query %d: %w", i, err))
		}
		if err := writeVector(out, vector); err != nil {
			panic(fmt.Errorf("write query %d: %w", i, err))
		}
	}
	if err := out.Sync(); err != nil {
		panic(err)
	}
	fmt.Printf("wrote %s in %s\n", *output, time.Since(started).Round(time.Second))
}

func embedDocuments(ctx context.Context, engine *embed.Engine, documents []string, backend string) ([][]float32, error) {
	if backend != "gguf" {
		return engine.BatchEmbedDocuments(ctx, documents)
	}

	vectors := make([][]float32, len(documents))
	var wg sync.WaitGroup
	jobs := make(chan int)
	errCh := make(chan error, 1)
	for range 2 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				vector, err := engine.EmbedDocument(ctx, documents[i])
				if err != nil {
					select {
					case errCh <- fmt.Errorf("embed document %d: %w", i, err):
					default:
					}
					continue
				}
				vectors[i] = vector
			}
		}()
	}
	for i := range documents {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	select {
	case err := <-errCh:
		return nil, err
	default:
		return vectors, nil
	}
}

func loadTexts(path string, vectorCount, queryCount, maxChars int) ([]string, []string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}
	decoder, err := apexjson.NewDecoder()
	if err != nil {
		return nil, nil, err
	}
	defer decoder.Free()
	if err := decoder.Parse(data); err != nil {
		return nil, nil, err
	}
	root := decoder.Root()
	if root.Type() != apexjson.TypeArray {
		return nil, nil, fmt.Errorf("corpus root is not an array")
	}

	documents := make([]string, 0, vectorCount)
	queries := make([]string, 0, queryCount)
	seenDocuments := make(map[string]struct{}, vectorCount)
	seenQueries := make(map[string]struct{}, queryCount)
	items := root.ArrayIter()
	for items.Next() {
		record := items.Value()
		query := strings.TrimSpace(objectText(record, "question"))
		if query != "" && len(queries) < queryCount {
			if _, exists := seenQueries[query]; !exists {
				seenQueries[query] = struct{}{}
				queries = append(queries, query)
			}
		}
		if len(documents) >= vectorCount {
			continue
		}
		sessions := objectField(record, "haystack_sessions")
		sessionIter := sessions.ArrayIter()
		for sessionIter.Next() {
			messages := sessionIter.Value()
			messageIter := messages.ArrayIter()
			for messageIter.Next() {
				msg := messageIter.Value()
				text := strings.TrimSpace(objectText(msg, "content"))
				if text == "" {
					continue
				}
				runes := []rune(text)
				if maxChars > 0 && len(runes) > maxChars {
					text = string(runes[:maxChars])
				}
				key := msg.Role + "\x00" + text
				if _, exists := seenDocuments[key]; exists {
					continue
				}
				seenDocuments[key] = struct{}{}
				documents = append(documents, msg.Role+": "+text)
				if len(documents) == vectorCount {
					break
				}
			}
			if len(documents) == vectorCount {
				break
			}
		}
	}
	if len(documents) != vectorCount || len(queries) != queryCount {
		return nil, nil, fmt.Errorf("corpus yielded documents=%d/%d queries=%d/%d", len(documents), vectorCount, len(queries), queryCount)
	}
	return documents, queries, nil
}

func objectField(value apexjson.Value, name string) apexjson.Value {
	it := value.ObjectIter()
	for it.Next() {
		if it.Key() == name {
			return it.Value()
		}
	}
	return apexjson.Value{}
}

func objectText(value apexjson.Value, name string) string {
	field := objectField(value, name)
	if field.Type() != apexjson.TypeString {
		return ""
	}
	return field.Str()
}

func writeVector(w io.Writer, vector []float32) error {
	if len(vector) != 768 {
		return fmt.Errorf("got %d dimensions, want 768", len(vector))
	}
	buf := make([]byte, len(vector)*4)
	var norm float64
	for i, value := range vector {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return fmt.Errorf("non-finite value at dimension %d", i)
		}
		norm += float64(value * value)
		binary.LittleEndian.PutUint32(buf[i*4:i*4+4], math.Float32bits(value))
	}
	if math.Abs(norm-1) > 0.001 {
		return fmt.Errorf("vector norm squared %.8f is not normalized", norm)
	}
	_, err := w.Write(buf)
	return err
}

func hashModel(path string) (uint64, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	hash := sha256.New()
	if _, err := io.Copy(hash, f); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(hash.Sum(nil)[:8]), nil
}
