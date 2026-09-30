# 🗳️ Voting Ensembles Master Index

Welcome to the Voting Ensembles architectural memory notes. Voting ensembles provide a principled, low-complexity method to combine multiple distinct models.

---

## 📚 Guides in this Folder

1. [**01. Hard vs. Soft Voting Principles**](file:///home/python/03_Custom_addons/ML/memory_notes/07_Ensemble_Methods/01_Voting/01_hard_vs_soft_voting_principles.md)
   - Condorcet's Jury Theorem and error independence
   - Hard voting (majority vote) formulation and limitations
   - Soft voting (weighted probability averaging) formulation
   - Simple and enterprise production implementation patterns

2. [**02. Production Pipeline & Latency Benchmarks**](file:///home/python/03_Custom_addons/ML/memory_notes/07_Ensemble_Methods/01_Voting/02_production_pipeline_and_latency_benchmarks.md)
   - Real-world latency bottlenecks and multi-model overhead
   - Serialization via `joblib` and thread safety
   - Enterprise deployment checklist and latency budgets
