"""Stable import path: ``from backend import audio_features``."""
import sys
from importlib import import_module

sys.modules[__name__] = import_module("backend.embeddings.audio_features")
