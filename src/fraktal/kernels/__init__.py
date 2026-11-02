"""Heavy numeric kernels.

Contract (ADR-0005): arrays and plain parameters in, arrays out. No file paths, no global
state, no objects from the rest of the system. Every kernel gets a golden-output test so a
native (Rust or C++) replacement can be proven equivalent.
"""
