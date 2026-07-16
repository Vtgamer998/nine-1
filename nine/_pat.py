"""Conjunto de caracteres regex simplificado (compat com fallback)."""
PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z_À-ÿ]+| ?[0-9]+| ?[^\w\s]+|\s+(?!\S)|\s+"""
