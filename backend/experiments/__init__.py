"""Offline experiment surface — never imported by the production app.

Modules under this package run analysis over aligned cached/extracted vectors
and report results; they are not wired into ``backend.app``, run no inference
at import time, and must stay import-safe (heavy dependencies lazy). Production
modules must not import from here — this is a scratch-and-report area the
experiment owner drives directly.
"""
