import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report

DATA_PATH = "events_labeled_rules_1_3.csv"
RANDOM_STATE = 42

# STEP 1: AUDIT THE LABEL

print("=" * 70)
print("STEP 1: LABEL AUDIT")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
print(f"\nLoaded {len(df)} events.")
print(f"Label distribution:\n{df['fault_reactivation_label'].value_counts()}\n")

# Check what actually predicts the label perfectly (should reveal circularity)
crosstab_temporal = pd.crosstab(df['temporal_label'], df['fault_reactivation_label'])
crosstab_sig = pd.crosstab(df['is_significant'], df['fault_reactivation_label'])

print("Cross-tab: is_significant (Mw >= -0.5) vs label")
print(crosstab_sig)
print("\nCross-tab: temporal_label vs label")
print(crosstab_temporal)

# Quantify: what fraction of label=1 is explained by is_significant AND
# temporal cluster alone?
exact_match = ((df['is_significant']) &
               (df['temporal_label'] == 'REACTIVATION (\u22653 cluster)')).astype(int)
agreement = (exact_match == df['fault_reactivation_label']).mean()
print(f"\n>>> Rule 'is_significant AND temporal_cluster' reproduces the actual "
      f"label {agreement:.1%} of the time.")
print(">>> This means any feature built from MomMag or from event-timing/")
print(">>> counting within windows is NOT an independent predictor -- it is")
print(">>> a restatement of the label. Do not use p1_* (moment/magnitude)")
print(">>> or p3_* (temporal/inter-event) engineered features as MODEL INPUTS")
print(">>> if your goal is to predict this label. They will always give you")
print(">>> near-perfect, meaningless AUC.\n")

# Check whether spatial features are actually independent of the label
print("Spatial confidence tier vs label (independence check):")
print(pd.crosstab(df['spatial_confidence'], df['fault_reactivation_label'], normalize='columns'))
print(">>> Distributions are similar across classes -> spatial_confidence tier")
print(">>> itself carries little signal, BUT raw geometry (X, Y, Depth,")
print(">>> distance_to_plane, assigned_plane) might still correlate with WHERE")
print(">>> significant clustering happens, for real physical reasons (some")
print(">>> planes/zones may genuinely produce bigger, more clustered events).")
print(">>> That's testable and legitimate -- see Step 2.\n")

# STEP 2: HONEST BASELINE (independent features only)

print("=" * 70)
print("STEP 2: HONEST BASELINE MODEL")
print("=" * 70)

INDEPENDENT_FEATURES = ['X', 'Y', 'Depth', 'distance_to_plane', 'assigned_plane']
print(f"\nUsing ONLY features with no mathematical link to the label:")
print(f"  {INDEPENDENT_FEATURES}")
print("(Explicitly excluded: MomMag, magnitude_category, is_significant,")
print(" magnitude_confidence, temporal_label, temporal_confidence,")
print(" spatial_confidence, combined_confidence -- all are label ingredients")
print(" or near-duplicates of them.)\n")

X = df[INDEPENDENT_FEATURES].copy()
y = df['fault_reactivation_label']

rf = RandomForestClassifier(
    n_estimators=300, max_depth=5, random_state=RANDOM_STATE,
    class_weight='balanced'
)

# NOTE ON CV CHOICE: plain shuffled K-fold on spatiotemporal event data can
# still leak, because nearby events in time/space are not independent
# samples (two events 5 minutes apart in the same cluster can land on
# opposite sides of a random split). Stratified K-fold shuffled is used here
# ONLY as a rough sanity check -- treat it as an upper bound, not truth.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scores = cross_val_score(rf, X, y, cv=cv, scoring='roc_auc')
print(f"Shuffled 5-fold CV AUC (upper-bound estimate, has spatial-autocorrelation")
print(f"risk -- see note above): {scores.mean():.3f} (+/- {scores.std():.3f})")

# STEP 3: STAGE-HOLDOUT VALIDATION (the real generalization test)

print("\n" + "=" * 70)
print("STEP 3: STAGE-HOLDOUT VALIDATION (proper generalization test)")
print("=" * 70)

for s in sorted(df['Stage'].unique()):
    sub = df[df['Stage'] == s]
    n_sig = sub['is_significant'].sum()
    n_pos = (sub['fault_reactivation_label'] == 1).sum()
    print(f"  {s}: n={len(sub)}, n_significant(Mw>=-0.5)={n_sig}, "
          f"max Mw={sub['MomMag'].max():.2f}, n_positive_label={n_pos}")

print("\n>>> Stage 2 has 0 or ~0 positive labels. A train-on-{1,3}/test-on-2")
print(">>> split, as previously proposed, is UNDEFINED (no positive class to")
print(">>> score against). This is not a bug in your code -- Stage 2 produced")
print(">>> almost no significant-magnitude events at all (1 of 948, max")
print(">>> Mw=-0.33). VERIFY against real injection logs whether Stage 2 was")
print(">>> actually the largest-volume stage before writing that into the")
print(">>> paper -- that claim was asserted, not measured, in earlier notes.")

train_mask = df['Stage'].isin(['Stage 1', 'Stage 3'])
test_mask = df['Stage'] == 'Stage 2'
train, test = df[train_mask], df[test_mask]

if test['fault_reactivation_label'].nunique() < 2:
    print(f"\nConfirmed: Stage 2 test set has only "
          f"{test['fault_reactivation_label'].nunique()} class present.")
    print("Alternative validation strategies to consider:")
    print("  a) Train on Stage 1, test on Stage 3 (both have positive cases)")
    print("  b) Time-blocked CV within Stage 1+3 (split chronologically, not randomly)")
    print("  c) Get data from another well/site to use as a true external test set")
else:
    rf.fit(train[INDEPENDENT_FEATURES], train['fault_reactivation_label'])
    preds = rf.predict_proba(test[INDEPENDENT_FEATURES])[:, 1]
    auc = roc_auc_score(test['fault_reactivation_label'], preds)
    print(f"\nStage-holdout AUC: {auc:.3f}")

# Try (a): Stage 1 <-> Stage 3, both directions, since both have positives
print("\n--- Alternative: Train Stage 1, Test Stage 3 ---")
tr = df[df['Stage'] == 'Stage 1']
te = df[df['Stage'] == 'Stage 3']
rf2 = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=RANDOM_STATE, class_weight='balanced')
rf2.fit(tr[INDEPENDENT_FEATURES], tr['fault_reactivation_label'])
preds = rf2.predict_proba(te[INDEPENDENT_FEATURES])[:, 1]
print(f"AUC: {roc_auc_score(te['fault_reactivation_label'], preds):.3f}")

print("\n--- Alternative: Train Stage 3, Test Stage 1 ---")
rf3 = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=RANDOM_STATE, class_weight='balanced')
rf3.fit(te[INDEPENDENT_FEATURES], te['fault_reactivation_label'])
preds = rf3.predict_proba(tr[INDEPENDENT_FEATURES])[:, 1]
print(f"AUC: {roc_auc_score(tr['fault_reactivation_label'], preds):.3f}")

print("\n" + "=" * 70)
print("BOTTOM LINE")
print("=" * 70)
print("""

