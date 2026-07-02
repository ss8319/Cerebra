"""DermArena dx-track agent — a lean per-case QA pipeline.

Design (see Cerebra/PROGRESS.md): LOAD -> ANALYSE (route) -> PROPOSE (base MLLM) ->
RUN (vision tools) -> DEBATE (fuse) -> EMIT. This package reuses Cerebra's BaseTool/
Dataset contract + the modality router, but NOT the legacy dynamic orchestrator.
"""

from cerebra.dermarena.case import Case, load_cases
from cerebra.dermarena.analyse import build_plan

__all__ = ["Case", "load_cases", "build_plan"]
