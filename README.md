# Fault Reactivation Signatures in an Enhanced Geothermal Reservoir

### A Multivariate Geomechanical Analysis Applied to Utah FORGE Microseismic Data

This repository contains the computational workflow for the study:

> **Fault Reactivation Signatures in an Enhanced Geothermal Reservoir: A Multivariate Geomechanical Analysis Applied to Utah FORGE Microseismic Data**

The study investigates whether microseismic and geomechanical variables contain independent information associated with operational fault-reactivation events during Enhanced Geothermal System (EGS) stimulation.

The central methodological question is:

> Does a machine-learning model learn an independent physical relationship, or does it simply learn the rules used to construct the target label?

---

## Overview

The analysis uses the April 2022 microseismic catalog from Utah FORGE Well 16A(78)-32, containing 2,591 events across Stages 1–3.

Because direct observation of fault slip is not available for every microseismic event, an operational fault-reactivation label was constructed using:

- Moment magnitude: Mw ≥ −0.5
- Temporal clustering: ≥3 magnitude-significant events within a 1-hour window

This produced:

- 115 operational reactivation candidates
- 2,476 background events

Spatial assignment to 15 mapped discrete fracture network (DFN) planes was tracked separately and was not included in the composite label.

---

## The Leakage Problem

A central focus of this study is feature leakage caused by overlap between the target definition and predictive features.

The operational label is constructed using magnitude and temporal clustering. Therefore, predictors derived from magnitude, event timing, event counts, or event rates can be mathematically related to the target.

A classifier using such predictors may achieve apparently strong performance by learning the construction of the label, rather than an independent physical precursor.

Therefore, the analysis explicitly audits the candidate features before predictive modeling.

## Analysis Workflow
Utah FORGE microseismic catalog
              │
              ↓
        Event labeling
              │
              ↓
      Label construction audit
              │
              ↓
       Feature leakage audit
              │
              ↓
    Leakage-audited feature set
              │
              ↓
       Geometry baseline
              │
              ↓
    Stage-based holdout testing
              │
              ↓
 Bootstrap + permutation testing
              │
              ↓
      Geomechanical analysis
          /           \
         /             \
Ambient CFF       Pressure diffusion
                       │
                       ↓
             Stage-specific bulk
                permeability
                       │
                       ↓
             Final interpretation

Key Results
1. Geometry
After removing circular predictors, geometric variables retained modest discriminatory information.

Stage 1 → Stage 3
AUC = 0.624, p < 0.0001

Stage 3 → Stage 1
AUC = 0.539, p = 0.23

The asymmetric holdout results indicate that the relationship between event geometry and the operational label is modest and not consistently transferable between stimulation stages. This suggests that the relationship between seismicity and the evolving reservoir state may change during stimulation.

2. Ambient Coulomb Stress

Ambient Coulomb failure stress was evaluated for the 15 mapped DFN planes. All 15 planes remained stress-stable under the tested ambient stress model. 

Thus, ambient static stress provided no meaningful additional discriminatory information for the operational label under the tested conditions. This does not imply that stress is unimportant to induced seismicity. It indicates that the tested ambient stress representation did not explain additional variation in this particular classification problem.

3. Stage-Specific Bulk Permeability

Pressure diffusion was reevaluated using stage-specific bulk permeability estimated from injection data following the methodology of Yu et al. (2024).

Stage	Bulk permeability
Stage 1	1.09 × 10⁻¹² m²
Stage 2	6.98 × 10⁻¹³ m²
Stage 3	3.94 × 10⁻¹³ m²

The resulting modeled pore-pressure perturbations were approximately 5–6 orders of magnitude larger than those obtained using the earlier fracture-scale permeability assumption. However, the revised pressure-diffusion feature still did not provide predictive improvement over geometry alone. This indicates that the negative pressure-diffusion result cannot simply be attributed to the earlier permeability assumption.

4. Independent Focal-Mechanism Validation

Independent focal mechanisms were used to evaluate the spatial and orientational consistency of the mapped DFN structures. 717 focal mechanisms evaluated, 440 matched events, 7.9° median strike agreement. The agreement provides independent support for the mapped DFN geometry.

Importantly, this validates where the structures are, not why individual events satisfy the operational reactivation label.

## Main Interpretation

The leakage-audited analysis shows:

| Variable group	| Result |
| Geometry	| Modest discriminatory signal |
| Ambient Coulomb stress	| No additional predictive lift |
| Pressure diffusion	| No additional predictive lift |
| Focal mechanisms	| Independent support for DFN geometry |

The asymmetric performance between stimulation stages suggests that the relationship between seismicity and the reservoir state is not stationary throughout stimulation. The results therefore suggest that static geometric and geomechanical variables alone are insufficient to fully explain which events satisfy the operational fault-reactivation label.

Future predictive models may need to incorporate time-dependent reservoir evolution, including:

* stress redistribution
* fracture-network evolution
* evolving permeability
* pressure evolution
* seismic migration
* hydromechanical interactions
* Scientific Contribution

The primary contribution of this work is methodological rather than maximizing classification performance. The study demonstrates that supervised learning for EGS fault-reactivation analysis should first establish whether the target and predictors are independent. A model can appear highly predictive when its input features overlap with the rules used to construct the target. By explicitly auditing these relationships and removing circular predictors, this work establishes a more defensible baseline for evaluating the independent information content of microseismic and geomechanical variables.

The goal is not simply to obtain a high AUC, but to determine whether the AUC represents an independent physical relationship.

## Repository Structure

fault-reactivation-signatures/
│
├── README.md
│
├── leakageaudit.py
│   └── Feature and target leakage-audit workflow
│
└── pipeline.py
    └── Main analysis and modeling workflow

## Reproducibility

The analysis is implemented in Python.

## Data

The analysis uses the April 2022 Utah FORGE Well 16A(78)-32 microseismic catalog.
The repository does not redistribute third-party data. Users should obtain the original Utah FORGE dataset from its appropriate source and provide the required input data according to the structure expected by the analysis scripts.

Suryaningtyas, Imas V.
Fault Reactivation Signatures in an Enhanced Geothermal Reservoir: A Multivariate Geomechanical Analysis Applied to Utah FORGE Microseismic Data.

## Keywords
Enhanced Geothermal Systems · EGS · Utah FORGE · microseismicity · fault reactivation · geomechanics · machine learning · feature leakage · data leakage · Coulomb stress · pressure diffusion · bulk permeability · DFN · induced seismicity
