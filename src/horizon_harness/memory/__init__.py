"""Memory for HorizonHarness.

The memory package stays outside the core runtime. Memory implementations can
inject ordinary messages, recall context, provide ordinary tools, and observe
completed turns or final State.
"""

from .base import (
    Memory,
    MemoryBinding,
    MemoryContext,
    memory_context_message,
)
from .filesystem import (
    DEFAULT_FILESYSTEM_MEMORY_ROOT,
    FilesystemArtifact,
    FilesystemDistillation,
    FilesystemIndexRow,
    FilesystemMemory,
    FilesystemMemoryLimits,
    FilesystemMemoryPayload,
    make_filesystem_distiller,
)
from .indexed import (
    Embedder,
    IndexedMemory,
    MemoryCandidate,
    MemoryEntry,
    SearchHit,
    SentenceTransformerEmbedder,
    tokenize,
)
from .extraction import (
    ExtractedMemory,
    MemoryEvidenceSource,
    MemoryExtractionResult,
    MemoryExtractor,
    ModelMemoryExtractor,
    TaskMemoryEvidence,
)

__all__ = [
    "FilesystemArtifact",
    "DEFAULT_FILESYSTEM_MEMORY_ROOT",
    "FilesystemDistillation",
    "FilesystemIndexRow",
    "FilesystemMemory",
    "FilesystemMemoryLimits",
    "FilesystemMemoryPayload",
    "Memory",
    "MemoryBinding",
    "MemoryContext",
    "Embedder",
    "ExtractedMemory",
    "IndexedMemory",
    "MemoryCandidate",
    "MemoryEvidenceSource",
    "MemoryExtractionResult",
    "MemoryExtractor",
    "MemoryEntry",
    "SearchHit",
    "SentenceTransformerEmbedder",
    "ModelMemoryExtractor",
    "TaskMemoryEvidence",
    "make_filesystem_distiller",
    "memory_context_message",
    "tokenize",
]
