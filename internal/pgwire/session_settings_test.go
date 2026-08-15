package pgwire

import "testing"

func TestPGWireSessionSettings(t *testing.T) {
	state := newConnState()
	handled, tag, err := applySessionSettingSQL(state, "SET statement_timeout = 2500")
	if err != nil || !handled {
		t.Fatalf("SET handled=%v tag=%s err=%v", handled, tag, err)
	}
	if tag != "SET" {
		t.Fatalf("SET command tag=%q", tag)
	}
	if state.config.StatementTimeout.Milliseconds() != 2500 {
		t.Fatalf("timeout=%v", state.config.StatementTimeout)
	}
	handled, tag, err = applySessionSettingSQL(state, "SELECT 1")
	if err != nil || handled {
		t.Fatalf("ordinary query handled=%v tag=%s err=%v", handled, tag, err)
	}
	handled, tag, err = applySessionSettingSQL(state, "RESET statement_timeout")
	if err != nil || !handled || state.config.StatementTimeout != 0 {
		t.Fatalf("RESET handled=%v tag=%s err=%v timeout=%v", handled, tag, err, state.config.StatementTimeout)
	}
	if tag != "RESET" {
		t.Fatalf("RESET command tag=%q", tag)
	}
	if handled, tag, err = applySessionSettingSQL(state, "SET enable_seqscan = on"); !handled || err == nil {
		t.Fatalf("unsupported planner setting handled=%v tag=%s err=%v", handled, tag, err)
	}
	results, columns, handled, err := handleSetConfigFunction("SELECT set_config('TimeZone', 'America/Los_Angeles', false)", &state.config, nil)
	if err != nil || !handled || len(columns) != 1 || results == nil || len(results.Results) != 1 {
		t.Fatalf("set_config handled=%v columns=%#v results=%#v err=%v", handled, columns, results, err)
	}
	if state.config.TimeZone != "America/Los_Angeles" {
		t.Fatalf("set_config timezone=%q", state.config.TimeZone)
	}
	results, columns, handled, err = handleAsyncpgJITQuery("SELECT current_setting('jit') AS cur, set_config('jit', 'off', false) AS new", &state.config, nil)
	if err != nil || !handled || len(columns) != 2 || results == nil || len(results.Results) != 1 || state.config.JIT != "off" {
		t.Fatalf("asyncpg jit handled=%v columns=%#v results=%#v jit=%q err=%v", handled, columns, results, state.config.JIT, err)
	}
}
