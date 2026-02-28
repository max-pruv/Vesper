"""Shared runtime state — accessible from both the scheduler and dashboard."""

cycle_state = {
    "total_cycles": 0,
    "last_cycle_time": 0,
    "last_cycle_error": "",
    "exchange_name": "",
    "version": "v4-exchange-autoselect",
    "startup_phase": "not_started",
}
