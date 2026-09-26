# Optimization Blueprint: Scaling Business Entity Resolution to 0.95+ Macro F0.5

## Diagnostic Summary
- **Current Score**: 0.846 Macro $F_{0.5}$
- **Target**: 0.95 - 0.98+ Macro $F_{0.5}$ without score regression

### 5-Pillar Architecture (Option A Implementation)
1. **Multi-Index Candidate Retrieval**: Expand Top-K from 25 to 35, indexing acronyms and postal codes for 98% retrieval recall.
2. **35-Feature Discriminative Vector**: Add Postal/Zip match/contradiction, Double Metaphone phonetics, TF-IDF rare token Jaccard, and Acronym match.
3. **Large-Scale Hard Negative Mining**: Mine 250,000 ground-truth entities against full 6.2M training target pool (~2.5M pairs) to train on real distractor distributions.
4. **Tri-Model Ensemble**: Train LightGBM + CatBoost + XGBoost with calibrated posterior probability blending:
   $$P_{\text{ensemble}} = 0.40 P_{\text{LGBM}} + 0.35 P_{\text{CatBoost}} + 0.25 P_{\text{XGBoost}}$$
5. **Dynamic Expected-F0.5 Optimizer**: Exact analytic subset selection maximizing expected $F_{0.5}$ per entity.
