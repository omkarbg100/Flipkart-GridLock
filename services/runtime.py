from __future__ import annotations

import threading

from services.job_manager import JobManager
from services.settings_store import SettingsStore

_settings_store: SettingsStore | None = None
_job_manager: JobManager | None = None
_lock = threading.Lock()


def get_settings_store() -> SettingsStore:
    global _settings_store
    with _lock:
        if _settings_store is None:
            _settings_store = SettingsStore()
        return _settings_store


def get_job_manager() -> JobManager:
    global _job_manager
    with _lock:
        if _job_manager is None:
            _job_manager = JobManager()
        return _job_manager

