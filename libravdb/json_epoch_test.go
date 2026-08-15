package libravdb

import (
	"context"
	"reflect"
	"testing"
)

func TestJSONBWriteOwnershipAndEpochSavepoint(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:json_epoch_ownership"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	col, err := db.CreateCollection(ctx, "json_epoch", WithMetadataOnly(), WithMetadataSchema(MetadataSchema{
		"payload": JSONBField,
	}), WithJSONIndexes(JSONIndexDefinition{
		Name: "json_epoch_name_idx", Column: "payload", Path: "{name}", TextResult: true,
	}))
	if err != nil {
		t.Fatal(err)
	}

	input := map[string]interface{}{
		"name":   "Ada",
		"count":  1.0,
		"nested": map[string]interface{}{"active": true},
	}
	if err := col.Insert(ctx, "d1", nil, map[string]interface{}{"payload": input}); err != nil {
		t.Fatalf("base JSONB insert: %v", err)
	}
	input["name"] = "mutated-after-insert"
	input["nested"].(map[string]interface{})["active"] = false

	base, err := col.Get(ctx, "d1")
	if err != nil {
		t.Fatal(err)
	}
	wantBase := map[string]interface{}{
		"name":   "Ada",
		"count":  int64(1),
		"nested": map[string]interface{}{"active": true},
	}
	if !reflect.DeepEqual(base.Metadata["payload"], wantBase) {
		t.Fatalf("stored JSONB was aliased or not canonical: got %#v want %#v", base.Metadata["payload"], wantBase)
	}
	base.Metadata["payload"].(map[string]interface{})["name"] = "mutated-return-value"
	baseAgain, err := col.Get(ctx, "d1")
	if err != nil {
		t.Fatal(err)
	}
	if baseAgain.Metadata["payload"].(map[string]interface{})["name"] != "Ada" {
		t.Fatal("mutating a returned JSONB map changed persisted state")
	}

	epoch, err := db.BeginEpochTx(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = epoch.Rollback(ctx) }()
	if err := epoch.Savepoint("base"); err != nil {
		t.Fatal(err)
	}
	update := map[string]interface{}{"payload": map[string]interface{}{
		"name":   "branch",
		"count":  2.0,
		"nested": map[string]interface{}{"active": false},
	}}
	if err := epoch.Update(ctx, "json_epoch", "d1", nil, update); err != nil {
		t.Fatalf("staged JSONB update: %v", err)
	}
	update["payload"].(map[string]interface{})["name"] = "mutated-after-stage"

	visible, err := epoch.ListRecords(ctx, "json_epoch")
	if err != nil {
		t.Fatal(err)
	}
	if len(visible) != 1 || visible[0].Metadata["payload"].(map[string]interface{})["name"] != "branch" {
		t.Fatalf("staged JSONB value missing or aliased: %#v", visible)
	}
	if err := epoch.Insert(ctx, "json_epoch", "d2", nil, map[string]interface{}{"payload": `{"name":"branch"}`}); err != nil {
		t.Fatalf("staged JSONB insert: %v", err)
	}
	indexedBranch, err := epoch.Query(ctx, `SELECT id FROM json_epoch WHERE payload#>>'{name}' = 'branch'`, nil)
	if err != nil {
		t.Fatalf("epoch JSON index overlay query: %v", err)
	}
	if indexedBranch.Total != 2 || indexedBranch.Results[0].ID != "d1" || indexedBranch.Results[1].ID != "d2" {
		t.Fatalf("epoch JSON index overlay result: %#v", indexedBranch)
	}
	containmentBranch, err := epoch.Query(ctx, `SELECT id FROM json_epoch WHERE payload @> '{"name":"branch"}' ORDER BY id`, nil)
	if err != nil {
		t.Fatalf("epoch JSON containment overlay query: %v", err)
	}
	if containmentBranch.Total != 2 {
		t.Fatalf("epoch JSON containment overlay result: %#v", containmentBranch)
	}
	if err := epoch.Delete(ctx, "json_epoch", "d1"); err != nil {
		t.Fatalf("staged JSONB delete: %v", err)
	}
	remainingBranch, err := epoch.Query(ctx, `SELECT id FROM json_epoch WHERE payload#>>'{name}' = 'branch'`, nil)
	if err != nil {
		t.Fatalf("epoch JSON delete overlay query: %v", err)
	}
	if remainingBranch.Total != 1 || remainingBranch.Results[0].ID != "d2" {
		t.Fatalf("epoch JSON delete overlay result: %#v", remainingBranch)
	}
	if err := epoch.RollbackTo("base"); err != nil {
		t.Fatal(err)
	}
	rolledBack, err := epoch.ListRecords(ctx, "json_epoch")
	if err != nil {
		t.Fatal(err)
	}
	if len(rolledBack) != 1 || !reflect.DeepEqual(rolledBack[0].Metadata["payload"], wantBase) {
		t.Fatalf("savepoint rollback did not restore canonical JSONB: %#v", rolledBack)
	}
	indexedBase, err := epoch.Query(ctx, `SELECT id FROM json_epoch WHERE payload#>>'{name}' = 'Ada'`, nil)
	if err != nil {
		t.Fatalf("rolled-back JSON index overlay query: %v", err)
	}
	if indexedBase.Total != 1 || indexedBase.Results[0].ID != "d1" {
		t.Fatalf("rolled-back JSON index overlay result: %#v", indexedBase)
	}
	containmentBase, err := epoch.Query(ctx, `SELECT id FROM json_epoch WHERE payload @> '{"name":"Ada"}'`, nil)
	if err != nil || containmentBase.Total != 1 || containmentBase.Results[0].ID != "d1" {
		t.Fatalf("rolled-back JSON containment result: %#v err=%v", containmentBase, err)
	}
}
