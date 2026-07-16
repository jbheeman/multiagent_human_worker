"""Shared backend helpers."""

from __future__ import annotations


def to_fs_label(value: str) -> str:
    return value.replace("/", "__")
