# NEXUS-Omega

A brain-inspired AI backend architecture designed to outperform GPT-6, Astra, DeepSeek, and Claude Fable.

## What Makes This Different?

Most AI models today are just scaled-up versions of the same Transformer architecture. They work, but they're wasteful as hell:
- Millions in compute costs per training run
- Models get amnesia when learning new things
- Memory usage explodes with context length
- Everything's frozen after training

NEXUS-Omega takes a completely different approach. We looked at how the human brain actually works and tried to mimic that:
- Sparse activation (only 1-2% of neurons active at any time)
- Local learning rules instead of global backprop
- Dual memory systems (fast learning + slow consolidation)
- O(N) memory scaling instead of O(N²)

## Status: Working Prototype

All the core layers are implemented and tested:
- [x] Sparse Dynamic Router (1024 micro-experts)
- [x] KAN Activation Edges (learnable B-splines)
- [x] Predictive Coding (Hebbian learning)
- [x] Adaptive Recurrent Depth
- [x] Dual-Plasticity Weights
- [x] Synaptic Consolidation
- [x] Engram Memory Cortex
- [x] Sparse Distributed Input

Tests all pass! Now we just need compute resources to pre-train it properly.

## Theoretical Foundation

Built on established neuroscience principles:

1. **Complementary Learning Systems** - Hippocampus (fast) + Neocortex (slow)
2. **Hebbian Learning** - "Neurons that fire together wire together"
3. **Sparse Distributed Memory** - 10^142 pattern capacity
4. **Predictive Coding** - Local error minimization
5. **Elastic Weight Consolidation** - Prevent catastrophic forgetting

## Goals

- Run on consumer hardware (RTX 4090 class)
- ~10x energy efficiency over GPT-6
- Zero catastrophic forgetting
- Continuous learning during inference
- O(N) instead of O(N²) memory scaling

## Documentation

- [ARCHITECTURE.md](./ARCHITECTURE.md) - Technical deep dive
- [DEVLOG.md](./DEVLOG.md) - Development progress
- [PLANS.md](./PLANS.md) - Future roadmap

## Contact

Built with too much caffeine and late-night coding sessions.