"""The registry: the CRM-backed list of accounts, contacts, products,
competitors and internal people that mentions resolve against.

Public surface: `Registry` (`.load`, `.import_crm`, `.resolve`, `.get`,
`.ancestors`, `.by_email`, `.by_domain`, `.scan`), the module-level
writers `add_alias`/`queue_unknown`, and the models `Entity`,
`Resolution`, `Match`, `ImportReport`, `EntityKind`.

`python -m portolan.registry import --accounts ... --contacts ...
[--meta ...]` runs an import from the command line (`cli.py`).
"""

from __future__ import annotations

from portolan.registry.models import Entity, EntityKind, ImportReport, Match, Resolution
from portolan.registry.registry import Registry, add_alias, queue_unknown

__all__ = [
    "Entity",
    "EntityKind",
    "ImportReport",
    "Match",
    "Registry",
    "Resolution",
    "add_alias",
    "queue_unknown",
]
