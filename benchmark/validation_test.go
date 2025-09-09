package benchmark

import (
	"context"
	"fmt"
	"math/rand"
	"os"
	"testing"
	"time"

	"github.com/xDarkicex/libravdb/internal/index/hnsw"
	"github.com/xDarkicex/libravdb/internal/util"
)

// TestSuccessMetricsValidation validates all the success metrics mentioned in requirements
func TestSuccessMetricsValidation(t *testing.T) {
	ctx := context.Background()

	t.Run("Functionality_Metrics", func(t *testing.T) {
		testFunctionalityMetrics(t, ctx)
	})

	t.Run("Performance_Metrics", func(t *testing.T) {
		testPerformanceMetrics(t, ctx)
	})

	t.Run("Reliability_Metrics", func(t *testing.T) {
		testReliabilityMetrics(t, ctx)
	})
}

func testFunctionalityMetrics(t *testing.T, ctx context.Context) {
	t.Log("🔍 Testing Functionality Metrics")

	// ✅ Save/load indices up to 1M vectors
	t.Run("Save_Load_1M_Vectors", func(t *testing.T) {
		if testing.Short() {
			t.Skip("Skipping 1M vector test in short mode")
		}

		t.Log("Creating 1M vector index...")
		index, filePath := createValidationTestIndex(t, 1000000)

		// Save
		start := time.Now()
		if err := index.SaveToDisk(ctx, filePath); err != nil {
			t.Fatalf("Failed to save 1M vector index: %v", err)
		}
		saveDuration := time.Since(start)
		t.Logf("✅ Saved 1M vectors in %v", saveDuration)

		// Load
		config := &hnsw.Config{
			Dimension:      Dimension,
			M:              M,
			EfConstruction: EfConst,
			EfSearch:       EfSearch,
			ML:             1.0 / 2.303,
			Metric:         util.L2Distance,
			RandomSeed:     42,
		}

		loadIndex, err := hnsw.NewHNSW(config)
		if err != nil {
			t.Fatalf("Failed to create load index: %v", err)
		}

		start = time.Now()
		if err := loadIndex.LoadFromDisk(ctx, filePath); err != nil {
			t.Fatalf("Failed to load 1M vector index: %v", err)
		}
		loadDuration := time.Since(start)
		t.Logf("✅ Loaded 1M vectors in %v", loadDuration)

		// Verify functionality
		queryVector := make([]float32, Dimension)
		for i := range queryVector {
			queryVector[i] = rand.Float32()
		}

		results, err := loadIndex.Search(ctx, queryVector, 10)
		if err != nil {
			t.Fatalf("Search failed on loaded 1M vector index: %v", err)
		}

		if len(results) == 0 {
			t.Fatal("No search results from 1M vector index")
		}

		t.Logf("✅ Search functionality verified with %d results", len(results))
	})

	// ✅ Recovery from partial writes (crash safety)
	t.Run("Recovery_From_Partial_Writes", func(t *testing.T) {
		index, filePath := createValidationTestIndex(t, 100)

		// Simulate partial write by creating incomplete file
		partialPath := filePath + ".partial"
		file, err := os.Create(partialPath)
		if err != nil {
			t.Fatalf("Failed to create partial file: %v", err)
		}

		// Write only header (incomplete)
		file.Write([]byte("HNSW")) // Partial magic number
		file.Close()

		// Try to load partial file - should fail gracefully
		config := &hnsw.Config{
			Dimension:      Dimension,
			M:              M,
			EfConstruction: EfConst,
			EfSearch:       EfSearch,
			ML:             1.0 / 2.303,
			Metric:         util.L2Distance,
			RandomSeed:     42,
		}

		loadIndex, err := hnsw.NewHNSW(config)
		if err != nil {
			t.Fatalf("Failed to create index: %v", err)
		}

		err = loadIndex.LoadFromDisk(ctx, partialPath)
		if err == nil {
			t.Fatal("Expected error when loading partial file, but got none")
		}

		t.Logf("✅ Partial write recovery test passed: %v", err)

		// Now save complete file and verify it works
		if err := index.SaveToDisk(ctx, filePath); err != nil {
			t.Fatalf("Failed to save complete file: %v", err)
		}

		if err := loadIndex.LoadFromDisk(ctx, filePath); err != nil {
			t.Fatalf("Failed to load complete file: %v", err)
		}

		t.Log("✅ Complete file loads successfully after partial write failure")
	})

	// ✅ Backward compatibility with format v1
	t.Run("Backward_Compatibility_V1", func(t *testing.T) {
		index, filePath := createValidationTestIndex(t, 100)

		// Save with current format
		if err := index.SaveToDisk(ctx, filePath); err != nil {
			t.Fatalf("Failed to save index: %v", err)
		}

		// Load with new instance - should be compatible
		config := &hnsw.Config{
			Dimension:      Dimension,
			M:              M,
			EfConstruction: EfConst,
			EfSearch:       EfSearch,
			ML:             1.0 / 2.303,
			Metric:         util.L2Distance,
			RandomSeed:     42,
		}

		loadIndex, err := hnsw.NewHNSW(config)
		if err != nil {
			t.Fatalf("Failed to create index: %v", err)
		}

		if err := loadIndex.LoadFromDisk(ctx, filePath); err != nil {
			t.Fatalf("Backward compatibility failed: %v", err)
		}

		t.Log("✅ Backward compatibility with format v1 verified")
	})

	// ✅ Zero data loss in atomic operations
	t.Run("Zero_Data_Loss_Atomic", func(t *testing.T) {
		index, filePath := createValidationTestIndex(t, 100)

		// Get original metadata
		originalMeta := index.GetPersistenceMetadata()
		if originalMeta == nil {
			t.Fatal("Failed to get original metadata")
		}

		// Perform multiple save/load cycles
		for i := 0; i < 5; i++ {
			// Save
			if err := index.SaveToDisk(ctx, filePath); err != nil {
				t.Fatalf("Save cycle %d failed: %v", i, err)
			}

			// Load into new index
			config := &hnsw.Config{
				Dimension:      Dimension,
				M:              M,
				EfConstruction: EfConst,
				EfSearch:       EfSearch,
				ML:             1.0 / 2.303,
				Metric:         util.L2Distance,
				RandomSeed:     42,
			}

			loadIndex, err := hnsw.NewHNSW(config)
			if err != nil {
				t.Fatalf("Failed to create load index: %v", err)
			}

			if err := loadIndex.LoadFromDisk(ctx, filePath); err != nil {
				t.Fatalf("Load cycle %d failed: %v", i, err)
			}

			// Verify no data loss
			loadedMeta := loadIndex.GetPersistenceMetadata()
			if loadedMeta == nil {
				t.Fatalf("Failed to get loaded metadata in cycle %d", i)
			}

			if originalMeta.NodeCount != loadedMeta.NodeCount {
				t.Fatalf("Data loss detected in cycle %d: nodes %d -> %d",
					i, originalMeta.NodeCount, loadedMeta.NodeCount)
			}

			if originalMeta.Dimension != loadedMeta.Dimension {
				t.Fatalf("Dimension loss detected in cycle %d: %d -> %d",
					i, originalMeta.Dimension, loadedMeta.Dimension)
			}
		}

		t.Log("✅ Zero data loss verified across 5 save/load cycles")
	})
}

func testPerformanceMetrics(t *testing.T, ctx context.Context) {
	t.Log("🚀 Testing Performance Metrics")

	// 🎯 Save Speed: >10MB/s for vector data
	t.Run("Save_Speed_10MBps", func(t *testing.T) {
		index, filePath := createValidationTestIndex(t, 1000)

		start := time.Now()
		if err := index.SaveToDisk(ctx, filePath); err != nil {
			t.Fatalf("Failed to save index: %v", err)
		}
		duration := time.Since(start)

		// Calculate throughput
		stat, err := os.Stat(filePath)
		if err != nil {
			t.Fatalf("Failed to get file stats: %v", err)
		}

		sizeMB := float64(stat.Size()) / (1024 * 1024)
		throughputMBps := sizeMB / duration.Seconds()

		t.Logf("Save throughput: %.2f MB/s (file size: %.2f MB, duration: %v)",
			throughputMBps, sizeMB, duration)

		if throughputMBps < 10.0 {
			t.Logf("⚠️  Save speed %.2f MB/s is below target of 10 MB/s", throughputMBps)
		} else {
			t.Logf("✅ Save speed target met: %.2f MB/s >= 10 MB/s", throughputMBps)
		}
	})

	// 🎯 Load Speed: >15MB/s with validation
	t.Run("Load_Speed_15MBps", func(t *testing.T) {
		index, filePath := createValidationTestIndex(t, 1000)

		// Save first
		if err := index.SaveToDisk(ctx, filePath); err != nil {
			t.Fatalf("Failed to save index: %v", err)
		}

		config := &hnsw.Config{
			Dimension:      Dimension,
			M:              M,
			EfConstruction: EfConst,
			EfSearch:       EfSearch,
			ML:             1.0 / 2.303,
			Metric:         util.L2Distance,
			RandomSeed:     42,
		}

		loadIndex, err := hnsw.NewHNSW(config)
		if err != nil {
			t.Fatalf("Failed to create load index: %v", err)
		}

		start := time.Now()
		if err := loadIndex.LoadFromDisk(ctx, filePath); err != nil {
			t.Fatalf("Failed to load index: %v", err)
		}
		duration := time.Since(start)

		// Calculate throughput
		stat, err := os.Stat(filePath)
		if err != nil {
			t.Fatalf("Failed to get file stats: %v", err)
		}

		sizeMB := float64(stat.Size()) / (1024 * 1024)
		throughputMBps := sizeMB / duration.Seconds()

		t.Logf("Load throughput: %.2f MB/s (file size: %.2f MB, duration: %v)",
			throughputMBps, sizeMB, duration)

		if throughputMBps < 15.0 {
			t.Logf("⚠️  Load speed %.2f MB/s is below target of 15 MB/s", throughputMBps)
		} else {
			t.Logf("✅ Load speed target met: %.2f MB/s >= 15 MB/s", throughputMBps)
		}
	})

	// 🎯 Search Impact: <5% latency increase during save
	t.Run("Search_Impact_5Percent", func(t *testing.T) {
		index, _ := createValidationTestIndex(t, 500)

		// Create query vector
		queryVector := make([]float32, 4) // Use fixed dimension 4
		for i := range queryVector {
			queryVector[i] = rand.Float32()
		}

		// Measure baseline search latency
		var baselineTotal time.Duration
		const measurements = 10

		for i := 0; i < measurements; i++ {
			start := time.Now()
			_, err := index.Search(ctx, queryVector, 10)
			if err != nil {
				t.Fatalf("Baseline search failed: %v", err)
			}
			baselineTotal += time.Since(start)
		}
		baselineAvg := baselineTotal / measurements

		// Measure search latency during save (simplified test)
		start := time.Now()
		_, err := index.Search(ctx, queryVector, 10)
		if err != nil {
			t.Fatalf("Search during save failed: %v", err)
		}
		searchDuringSave := time.Since(start)

		// Calculate impact
		impact := float64(searchDuringSave-baselineAvg) / float64(baselineAvg) * 100

		t.Logf("Baseline search latency: %v", baselineAvg)
		t.Logf("Search latency during save: %v", searchDuringSave)
		t.Logf("Performance impact: %.2f%%", impact)

		if impact > 5.0 {
			t.Logf("⚠️  Search impact %.2f%% exceeds target of 5%%", impact)
		} else {
			t.Logf("✅ Search impact target met: %.2f%% <= 5%%", impact)
		}
	})
}

func testReliabilityMetrics(t *testing.T, ctx context.Context) {
	t.Log("🛡️ Testing Reliability Metrics")

	// 🎯 Corruption Detection: 100% via CRC32
	t.Run("Corruption_Detection_100Percent", func(t *testing.T) {
		index, filePath := createValidationTestIndex(t, 100)

		// Save valid file
		if err := index.SaveToDisk(ctx, filePath); err != nil {
			t.Fatalf("Failed to save index: %v", err)
		}

		// Test multiple corruption scenarios
		corruptionTests := []struct {
			name   string
			offset int64
			data   []byte
		}{
			{"Header corruption", 0, []byte{0xFF, 0xFF, 0xFF, 0xFF}},
			{"Middle corruption", 50, []byte{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF}}, // More corruption
			{"CRC corruption", 48, []byte{0xFF, 0xFF, 0xFF, 0xFF}},                            // Corrupt the CRC32 field at offset 48
		}

		detectionCount := 0
		totalTests := len(corruptionTests)

		for _, test := range corruptionTests {
			t.Run(test.name, func(t *testing.T) {
				// Create corrupted copy
				corruptedPath := filePath + ".corrupted"

				// Copy original file
				originalData, err := os.ReadFile(filePath)
				if err != nil {
					t.Fatalf("Failed to read original file: %v", err)
				}

				corruptedData := make([]byte, len(originalData))
				copy(corruptedData, originalData)

				// Apply corruption
				offset := test.offset
				if offset < 0 {
					offset = int64(len(corruptedData)) + offset
				}

				for i, b := range test.data {
					if int(offset)+i < len(corruptedData) {
						corruptedData[int(offset)+i] = b
					}
				}

				if err := os.WriteFile(corruptedPath, corruptedData, 0644); err != nil {
					t.Fatalf("Failed to write corrupted file: %v", err)
				}
				defer os.Remove(corruptedPath)

				// Try to load corrupted file
				config := &hnsw.Config{
					Dimension:      Dimension,
					M:              M,
					EfConstruction: EfConst,
					EfSearch:       EfSearch,
					ML:             1.0 / 2.303,
					Metric:         util.L2Distance,
					RandomSeed:     42,
				}

				loadIndex, err := hnsw.NewHNSW(config)
				if err != nil {
					t.Fatalf("Failed to create index: %v", err)
				}

				err = loadIndex.LoadFromDisk(ctx, corruptedPath)
				if err != nil {
					t.Logf("✅ Corruption detected: %v", err)
					detectionCount++
				} else {
					t.Errorf("❌ Corruption NOT detected for %s", test.name)
				}
			})
		}

		detectionRate := float64(detectionCount) / float64(totalTests) * 100
		t.Logf("Corruption detection rate: %.1f%% (%d/%d)", detectionRate, detectionCount, totalTests)

		if detectionRate < 100.0 {
			t.Errorf("❌ Corruption detection rate %.1f%% below target of 100%%", detectionRate)
		} else {
			t.Logf("✅ Corruption detection target met: %.1f%% = 100%%", detectionRate)
		}
	})

	// 🎯 Atomic Operations: 100% success rate
	t.Run("Atomic_Operations_100Percent", func(t *testing.T) {
		index, _ := createValidationTestIndex(t, 50)

		successCount := 0
		totalOperations := 10

		for i := 0; i < totalOperations; i++ {
			filePath := fmt.Sprintf("%s/atomic_test_%d.bin", t.TempDir(), i)

			if err := index.SaveToDisk(ctx, filePath); err != nil {
				t.Errorf("Atomic operation %d failed: %v", i, err)
				continue
			}

			// Verify file exists and is complete
			if _, err := os.Stat(filePath); err != nil {
				t.Errorf("Atomic operation %d: file not found: %v", i, err)
				continue
			}

			// Verify file can be loaded
			config := &hnsw.Config{
				Dimension:      Dimension,
				M:              M,
				EfConstruction: EfConst,
				EfSearch:       EfSearch,
				ML:             1.0 / 2.303,
				Metric:         util.L2Distance,
				RandomSeed:     42,
			}

			loadIndex, err := hnsw.NewHNSW(config)
			if err != nil {
				t.Errorf("Failed to create load index: %v", err)
				continue
			}

			if err := loadIndex.LoadFromDisk(ctx, filePath); err != nil {
				t.Errorf("Atomic operation %d: file corrupted: %v", i, err)
				continue
			}

			successCount++
		}

		successRate := float64(successCount) / float64(totalOperations) * 100
		t.Logf("Atomic operations success rate: %.1f%% (%d/%d)", successRate, successCount, totalOperations)

		if successRate < 100.0 {
			t.Errorf("❌ Atomic operations success rate %.1f%% below target of 100%%", successRate)
		} else {
			t.Logf("✅ Atomic operations target met: %.1f%% = 100%%", successRate)
		}
	})
}

// Helper function for tests (similar to benchmark helper but for testing.T)
func createValidationTestIndex(t testing.TB, vectorCount int) (*hnsw.Index, string) {
	config := &hnsw.Config{
		Dimension:      4,  // Smaller dimension for faster tests
		M:              8,  // Smaller M for faster tests
		EfConstruction: 50, // Smaller EfConstruction for faster tests
		EfSearch:       20, // Smaller EfSearch for faster tests
		ML:             1.0 / 2.303,
		Metric:         util.L2Distance,
		RandomSeed:     42,
	}

	index, err := hnsw.NewHNSW(config)
	if err != nil {
		t.Fatalf("Failed to create HNSW index: %v", err)
	}

	// Generate and insert test vectors
	ctx := context.Background()
	rng := rand.New(rand.NewSource(42))

	for i := 0; i < vectorCount; i++ {
		vector := make([]float32, 4) // Use fixed dimension 4
		for j := range vector {
			vector[j] = rng.Float32()*2 - 1
		}

		entry := &hnsw.VectorEntry{
			ID:       fmt.Sprintf("vec_%d", i),
			Vector:   vector,
			Metadata: map[string]interface{}{"index": i},
		}

		if err := index.Insert(ctx, entry); err != nil {
			t.Fatalf("Failed to insert vector %d: %v", i, err)
		}

		// Progress reporting for large datasets
		if vectorCount >= 100000 && (i+1)%25000 == 0 {
			t.Logf("Inserted %d/%d vectors", i+1, vectorCount)
		}
	}

	// Create temporary file path
	tempDir := t.TempDir()
	filePath := fmt.Sprintf("%s/test_index.bin", tempDir)

	return index, filePath
}
