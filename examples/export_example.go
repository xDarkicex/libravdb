//go:build ignore

package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	apexjson "github.com/xDarkicex/apexJSON/v2"
	"github.com/xDarkicex/libravdb/libravdb"
)

func main() {
	path := flag.String("db", "./libravdb_data", "path to LibraVDB storage")
	collection := flag.String("collection", "", "export specific collection (default: all)")
	format := flag.String("format", "json", "output format: json, csv, markdown")
	output := flag.String("output", "", "output file (default: stdout)")
	flag.Parse()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	db, err := libravdb.Open(libravdb.WithStoragePath(*path))
	if err != nil {
		log.Fatalf("open database: %v", err)
	}
	defer db.Close()

	var collections []string
	if *collection != "" {
		collections = []string{*collection}
	} else {
		collections = db.ListCollections()
	}

	var out *os.File
	if *output != "" {
		out, err = os.Create(*output)
		if err != nil {
			log.Fatalf("create output file: %v", err)
		}
		defer out.Close()
	} else {
		out = os.Stdout
	}

	for _, name := range collections {
		coll, err := db.GetCollection(name)
		if err != nil {
			log.Printf("get collection %q: %v", name, err)
			continue
		}

		records, err := coll.ListAll(ctx)
		if err != nil {
			log.Printf("list all in %q: %v", name, err)
			continue
		}

		switch *format {
		case "json":
			exportJSON(out, name, records)
		case "csv":
			exportCSV(out, name, records)
		case "markdown":
			exportMarkdown(out, name, records)
		default:
			log.Fatalf("unknown format: %s", *format)
		}
	}
}

type exportRecord struct {
	ID       string                 `json:"id"`
	Vector   []float32              `json:"vector"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
	Version  uint64                 `json:"version"`
}

func exportJSON(out *os.File, collection string, records []libravdb.Record) error {
	for _, r := range records {
		rec := exportRecord{
			ID:       r.ID,
			Vector:   r.Vector,
			Metadata: r.Metadata,
			Version:  r.Version,
		}
		encoded, err := apexjson.Marshal(map[string]any{
			"collection": collection,
			"record":     rec,
		})
		if err != nil {
			return fmt.Errorf("encode record %s: %w", r.ID, err)
		}
		encoded = append(encoded, '\n')
		if _, err := out.Write(encoded); err != nil {
			return fmt.Errorf("write record %s: %w", r.ID, err)
		}
	}
	return nil
}

func exportCSV(out *os.File, collection string, records []libravdb.Record) error {
	fmt.Fprintf(out, "# Collection: %s\n", collection)
	fmt.Fprintf(out, "# Count: %d\n\n", len(records))

	if len(records) == 0 {
		return nil
	}

	fmt.Fprint(out, "id,version")
	if len(records[0].Metadata) > 0 {
		for k := range records[0].Metadata {
			fmt.Fprintf(out, ",meta_%s", k)
		}
	}
	fmt.Fprint(out, ",vector\n")

	for _, r := range records {
		fmt.Fprintf(out, "%s,%d", r.ID, r.Version)
		for _, v := range r.Metadata {
			fmt.Fprintf(out, ",%v", v)
		}
		vec := make([]string, len(r.Vector))
		for i, f := range r.Vector {
			vec[i] = fmt.Sprintf("%.6f", f)
		}
		fmt.Fprintf(out, ",\"[%s]\"\n", strings.Join(vec, ","))
	}
	return nil
}

func exportMarkdown(out *os.File, collection string, records []libravdb.Record) error {
	fmt.Fprintf(out, "# Collection: %s\n\n", collection)
	fmt.Fprintf(out, "**Count:** %d\n\n", len(records))

	if len(records) == 0 {
		fmt.Fprintln(out, "_No records_")
		return nil
	}

	fmt.Fprintln(out, "| ID | Version | Metadata | Vector |")
	fmt.Fprintln(out, "|---|---|---|---|")

	for _, r := range records {
		meta := fmt.Sprintf("%v", r.Metadata)
		if len(meta) > 50 {
			meta = meta[:47] + "..."
		}
		vec := make([]string, len(r.Vector))
		for i, f := range r.Vector {
			vec[i] = fmt.Sprintf("%.4f", f)
		}
		vecStr := strings.Join(vec, ", ")
		if len(vecStr) > 60 {
			vecStr = vecStr[:57] + "..."
		}
		fmt.Fprintf(out, "| %s | %d | %s | [%s] |\n", r.ID, r.Version, meta, vecStr)
	}
	return nil
}
