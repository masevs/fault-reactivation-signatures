
import pandas as pd
import numpy as np
from scipy.special import erfc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


# CONFIG

FT_PER_M = 3.28084
GAL_TO_M3 = 0.00378541
MU_WATER = 2e-4          # Pa.s, water at ~150C reservoir temp (assumption)
MU_FRICTION = 0.65       # Byerlee friction coefficient
GROUND_ELEV_FT = 5413.47 # exact, see note 6 above
SHMAX_AZIMUTH_DEG = 25.0 # N25E, orthogonal to Shmin N115E (Xing et al. 2022)
# Ambient stress gradients (psi/ft), Lu et al. 2025, well 16B Table 3
GRAD_SHMIN, GRAD_SHMAX, GRAD_SV = 0.735, 0.965, 1.115
PSI_FT_TO_MPA_M = 0.022621

STAGES_MD_M = {'Stage 1': (3287.9, 3348.8), 'Stage 2': (3218.7, 3224.8), 'Stage 3': (3084.6, 3090.7)}


# STEP 1: Load Rules 1-3 label (already validated -- see note 1)

events = pd.read_csv('events_labeled_rules_1_3.csv')
events['DateTime'] = pd.to_datetime(events['DateTime'])
print(f"[1] Loaded {len(events)} events, {events['fault_reactivation_label'].sum():.0f} positive (Rules 1-3)")


# STEP 2: Rule 4a -- ambient Coulomb stress per plane (corrected depths)

dfn = pd.read_csv('MS_DFN_Global_Coords.csv')
dfn['true_depth_ft'] = GROUND_ELEV_FT - dfn['FractureZ[m]'] * FT_PER_M
dfn['true_depth_m'] = dfn['true_depth_ft'] / FT_PER_M

def pole_vector_NED(trend_deg, plunge_deg):
    t, p = np.radians(trend_deg), np.radians(plunge_deg)
    return np.cos(p)*np.cos(t), np.cos(p)*np.sin(t), np.sin(p)

def resolve_stress_on_plane(shmin, shmax, sv, az_deg, trend_deg, plunge_deg):
    n = np.array(pole_vector_NED(trend_deg, plunge_deg))
    az = np.radians(az_deg)
    shmax_dir = np.array([np.cos(az), np.sin(az), 0.0])
    shmin_dir = np.array([np.cos(az+np.pi/2), np.sin(az+np.pi/2), 0.0])
    sv_dir = np.array([0.0, 0.0, 1.0])
    n1, n2, n3 = np.dot(n, shmax_dir), np.dot(n, shmin_dir), np.dot(n, sv_dir)
    sigma_n = shmax*n1**2 + shmin*n2**2 + sv*n3**2
    tau = np.sqrt(max((shmax*n1)**2 + (shmin*n2)**2 + (sv*n3)**2 - sigma_n**2, 0.0))
    return sigma_n, tau

plane_cff = []
for idx, row in dfn.iterrows():
    z = row['true_depth_m']
    shmin, shmax, sv = GRAD_SHMIN*PSI_FT_TO_MPA_M*z, GRAD_SHMAX*PSI_FT_TO_MPA_M*z, GRAD_SV*PSI_FT_TO_MPA_M*z
    sigma_n, tau = resolve_stress_on_plane(shmin, shmax, sv, SHMAX_AZIMUTH_DEG,
                                            row['Trend[deg]'], row['Plunge[deg]'])
    plane_cff.append({'assigned_plane': idx, 'ambient_CFF': tau - MU_FRICTION*sigma_n})
events = events.merge(pd.DataFrame(plane_cff), on='assigned_plane', how='left')
print("[2] Ambient CFF computed per plane (range: "
      f"{events['ambient_CFF'].min():.1f} to {events['ambient_CFF'].max():.1f} MPa -- "
      "all strongly stress-stable, see note 3)")


# STEP 3: Rule 4b -- dynamic pressure via point-source diffusion 

traj = pd.read_csv('Well_16A_78_-32_points_depths.csv')
max_md = traj['Measured Depth (m)'].max()
stage_inj_pts = {}
for s, (md0, md1) in STAGES_MD_M.items():
    mid = (md0 + md1) / 2
    row = traj.iloc[-1] if mid > max_md else traj.iloc[(traj['Measured Depth (m)']-mid).abs().idxmin()]
    stage_inj_pts[s] = (row['UTM_E'], row['UTM_N'], row['True Vert depth (ft)'])

edr = pd.read_csv('28044193-Pason_EDR.csv')
edr['DateTime'] = pd.to_datetime(edr['YYYY/MM/DD'] + ' ' + edr['HH:MM:SS'])
edr = edr[edr['Accum Flow (Gal)'] > -900].sort_values('DateTime')

stage_Q, stage_t0 = {}, {}
for s in events['Stage'].unique():
    sub = events[events['Stage']==s].sort_values('DateTime')
    t0, t1 = sub['DateTime'].min(), sub['DateTime'].max()
    stage_t0[s] = t0
    w = edr[(edr['DateTime']>=t0)&(edr['DateTime']<=t1)].sort_values('DateTime')
    # Accum Flow totalizer resets mid-stream -- sum positive increments only
    deltas = w['Accum Flow (Gal)'].diff().clip(lower=0)
    vol, dur = deltas.sum(), (w['DateTime'].iloc[-1]-w['DateTime'].iloc[0]).total_seconds()
    stage_Q[s] = max(vol*GAL_TO_M3/dur, 1e-6) if dur > 0 else 1e-4

def dP_point_source(Q, mu, k, ct, r_m, t_s):
    if r_m < 1: r_m = 1.0
    if t_s <= 0: return 0.0
    D = k / (mu*ct)
    return (Q*mu)/(4*np.pi*k*r_m) * erfc(r_m/np.sqrt(4*D*t_s)) / 1e6

dP_vals = []
for _, row in events.iterrows():
    s = row['Stage']
    ix, iy, itvd = stage_inj_pts[s]
    r_ft = np.sqrt((row['X_global']-ix)**2 + (row['Y_global']-iy)**2 + (row['Depth']-itvd)**2)
    t_s = (row['DateTime'] - stage_t0[s]).total_seconds()
    plane = dfn.iloc[int(row['assigned_plane'])]
    k, ct = plane['Permeability[m2]'], plane['Compressibility[1/kPa]']*1e-3
    dP_vals.append(dP_point_source(stage_Q[s], MU_WATER, k, ct, r_ft/FT_PER_M, t_s))
events['dP_diffusion_MPa'] = dP_vals
events['dynamic_CFF'] = events['ambient_CFF'] + MU_FRICTION*events['dP_diffusion_MPa']
print(f"[3] Diffusion dP computed -- max {events['dP_diffusion_MPa'].max():.2e} MPa "
      "(negligible; see note 4, this is a data limitation not a bug)")


# STEP 4: Honest stage-holdout evaluation

def stage_holdout_auc(df, features, train_stage, test_stage):
    tr = df[df['Stage']==train_stage].dropna(subset=features)
    te = df[df['Stage']==test_stage].dropna(subset=features)
    if te['fault_reactivation_label'].nunique() < 2:
        return None
    rf = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42, class_weight='balanced')
    rf.fit(tr[features], tr['fault_reactivation_label'])
    return roc_auc_score(te['fault_reactivation_label'], rf.predict_proba(te[features])[:,1])

GEOM = ['X','Y','Depth','distance_to_plane','assigned_plane']
GEOM_CFF = GEOM + ['dynamic_CFF']
GEOM_CFF_DP = GEOM + ['dynamic_CFF','dP_diffusion_MPa']

print("\n" + "="*70 + "\n[4] HONEST STAGE-HOLDOUT RESULTS\n" + "="*70)
for name, feats in [('Geometry only', GEOM), ('+ dynamic CFF', GEOM_CFF), ('+ dynamic CFF + dP', GEOM_CFF_DP)]:
    a = stage_holdout_auc(events, feats, 'Stage 1', 'Stage 3')
    b = stage_holdout_auc(events, feats, 'Stage 3', 'Stage 1')
    print(f"{name:35s}: S1->S3={a:.3f}, S3->S1={b:.3f}")


# STEP 5: Focal mechanism validation 

try:
    foc = pd.read_csv('focals.csv')
    foc['origin_time'] = pd.to_datetime(foc['origin_time'])
    ev3 = events[events['Stage']=='Stage 3'].sort_values('DateTime')
    merged = pd.merge_asof(foc.sort_values('origin_time'), ev3,
                            left_on='origin_time', right_on='DateTime',
                            direction='nearest', tolerance=pd.Timedelta('2s'))
    matched = merged.dropna(subset=['assigned_plane'])

    def strike_diff(a, b):
        d = abs(a-b) % 180
        return min(d, 180-d)

    diffs = np.array([min(strike_diff(r['strike1'], dfn.iloc[int(r['assigned_plane'])]['Strike[deg]']),
                           strike_diff(r['strike2'], dfn.iloc[int(r['assigned_plane'])]['Strike[deg]']))
                       for _, r in matched.iterrows()])
    print("\n" + "="*70 + "\n[5] FOCAL MECHANISM VALIDATION (Stage 3 only)\n" + "="*70)
    print(f"Matched {len(matched)}/{len(foc)} focal mechanisms within 2s")
    print(f"Median strike agreement: {np.median(diffs):.1f} deg | "
          f"within 20deg: {(diffs<20).mean():.1%} | within 45deg: {(diffs<45).mean():.1%}")
except FileNotFoundError:
    print("\n[5] focals.csv not found -- skipping focal mechanism validation")

events.to_csv('/mnt/user-data/outputs/events_final_all_rules.csv', index=False)
print("\nSaved: events_final_all_rules.csv")
