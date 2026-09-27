"""
TOFFEE state management.

Mirrors ONYX's state.json pattern for resume/overwrite across passes.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from toffee.config import AppState


def load_state(output_dir: Path) -> Optional[AppState]:
    """Load state from state.json if it exists."""
    state_path = output_dir / "state.json"
    if not state_path.exists():
        return None
    with open(state_path, "r") as f:
        return AppState(**json.load(f))


def save_state(output_dir: Path, state: AppState) -> None:
    """Atomically save state to state.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp = output_dir / "state.json.tmp"
    with open(tmp, "w") as f:
        json.dump(state.model_dump(), f, indent=2)
    tmp.replace(output_dir / "state.json")


def prompt_resume_or_overwrite(state: Optional[AppState]) -> str:
    """Prompt user to resume or overwrite existing state."""
    if state is None:
        return "new"

    print(f"\nExisting TOFFEE state found:")
    print(f"  current_pass: {state.current_pass}")
    print(f"  last_completed_step: {state.last_completed_step}")
    print(f"  model_name: {state.model_name}")
    print(f"  started_at: {state.started_at}")
    print()
    print("Options:")
    print("  [R]esume from last completed step")
    print("  [O]verwrite and start fresh")
    print("  [Q]uit")
    print()

    while True:
        choice = input("Choice [R/O/Q]: ").strip().upper()
        if choice in ("R", "RESUME"):
            return "resume"
        elif choice in ("O", "OVERWRITE"):
            return "overwrite"
        elif choice in ("Q", "QUIT"):
            import sys
            sys.exit(0)
        else:
            print("Invalid choice. Please enter R, O, or Q.")
