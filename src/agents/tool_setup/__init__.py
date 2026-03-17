"""Tool setup package exports with lazy loading to avoid circular imports."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "register_agronomy_tools": "src.agents.tool_setup.agronomy.register_agronomy_tools",
    "register_commerce_tools": "src.agents.tool_setup.commerce.register_commerce_tools",
    "register_rate_tools": "src.agents.tool_setup.rates.register_rate_tools",
    "register_research_tools": "src.agents.tool_setup.research.register_research_tools",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'src.agents.tool_setup' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name].rsplit(".", 1)
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
