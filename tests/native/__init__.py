"""Integration tests for the Python↔C++ native executor bridge.

These tests exercise the full end-to-end path:
    Python circuit → QSAL JSON → pybind11 → C++ executor → results → Python
"""
