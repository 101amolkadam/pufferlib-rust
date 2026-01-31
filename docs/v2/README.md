# PufferLib v2 Documentation

*Comprehensive developer documentation for PufferLib Rust implementation.*

---

## Quick Links

| Document | Description |
|:---------|:------------|
| [ARCHITECTURE.md](./ARCHITECTURE.md) | System design and module structure |
| [CONTRIBUTING.md](./CONTRIBUTING.md) | Developer setup and contribution guidelines |
| [ROADMAP_V2.md](../../ROADMAP_V2.md) | Future development phases |

---

## Documentation Structure

```
docs/v2/
├── README.md              # This file
├── ARCHITECTURE.md        # Master architecture overview
├── CONTRIBUTING.md        # Developer guide
│
├── algorithms/            # Phase 5: Advanced Algorithms
│   ├── DECISION_TRANSFORMER.md
│   ├── MAPPO.md
│   ├── WORLD_MODELS.md
│   └── GRPO.md
│
├── production/            # Phase 6: Production Hardening
│   ├── CHECKPOINTING.md
│   ├── LOGGING.md
│   └── DISTRIBUTED.md
│
├── ecosystem/             # Phase 7: Ecosystem Expansion
│   ├── BACKENDS.md
│   └── INTEROP.md
│
└── research/              # Phase 8: Research Frontiers
    ├── SAFE_RL.md
    └── LLM_POLICIES.md
```

---

## Phase 5: Advanced Algorithms

Modern RL algorithm implementations for cutting-edge performance.

| Algorithm | Document | Status |
|:----------|:---------|:-------|
| Decision Transformer | [DECISION_TRANSFORMER.md](./algorithms/DECISION_TRANSFORMER.md) | 📋 Planned |
| MAPPO (Multi-Agent) | [MAPPO.md](./algorithms/MAPPO.md) | 📋 Planned |
| World Models | [WORLD_MODELS.md](./algorithms/WORLD_MODELS.md) | 📋 Planned |
| GRPO | [GRPO.md](./algorithms/GRPO.md) | 📋 Planned |

---

## Phase 6: Production Hardening

Reliability features for production deployments.

| Feature | Document | Status |
|:--------|:---------|:-------|
| Checkpointing | [CHECKPOINTING.md](./production/CHECKPOINTING.md) | 📋 Planned |
| Logging (TensorBoard/W&B) | [LOGGING.md](./production/LOGGING.md) | 📋 Planned |
| Distributed Training | [DISTRIBUTED.md](./production/DISTRIBUTED.md) | 📋 Planned |

---

## Phase 7: Ecosystem Expansion

Backend abstractions and Python interoperability.

| Feature | Document | Status |
|:--------|:---------|:-------|
| Backend Abstraction | [BACKENDS.md](./ecosystem/BACKENDS.md) | 📋 Planned |
| Python Interop (PyO3, HF Hub) | [INTEROP.md](./ecosystem/INTEROP.md) | 📋 Planned |

---

## Phase 8: Research Frontiers

Experimental features for research applications.

| Feature | Document | Status |
|:--------|:---------|:-------|
| Safe RL | [SAFE_RL.md](./research/SAFE_RL.md) | 📋 Planned |
| LLM Policies | [LLM_POLICIES.md](./research/LLM_POLICIES.md) | 📋 Planned |

---

## Getting Started

### For Contributors

1. Read [ARCHITECTURE.md](./ARCHITECTURE.md) for system overview
2. Follow [CONTRIBUTING.md](./CONTRIBUTING.md) for setup
3. Pick a feature from the roadmap and implement

### For Users

1. Check [pufferlib-rust/README.md](../../README.md) for installation
2. See [examples/](../../examples/) for usage patterns
3. Run `cargo doc --open` for API documentation

---

## Status Legend

| Icon | Meaning |
|:-----|:--------|
| ✅ | Complete and tested |
| 🚧 | In progress |
| 📋 | Planned (documented) |
| ❌ | Blocked |

---

*Last updated: 2026-01-28*
