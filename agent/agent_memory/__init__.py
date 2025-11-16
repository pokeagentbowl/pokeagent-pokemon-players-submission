"""
Agent Memory Module

Provides extensible memory management for agents, starting with location-based memory.
Memory types can be easily added and are stored in a modular JSON structure.
"""

from .memory_manager import MemoryManager

__all__ = ['MemoryManager']
