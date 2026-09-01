import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

target = 'addicted_label'
cat_cols = ['gender','stress_level','academic_work_impact']
base_num_cols = ['age','daily_screen_time_hours','social_media_hours','gaming_hours',
                'work_study_hours','sleep_hours','notifications_per_day',
                'app_opens_per_day','weekend_screen_time']

def feature_engineer(df):
    df = df.copy()
    # missing indicators (missingness itself may be predictive)
    for c in base_num_cols + cat_cols:
        df[f'{c}_missing'] = df[c].isna().astype(int)
    df['missing_count'] = df[[f'{c}_missing' for c in base_num_cols+cat_cols]].sum(axis=1)

    # ratios / interactions (safe against div by zero -> inf handled by replacing later)
    df['social_to_screen'] = df['social_media_hours'] / df['daily_screen_time_hours']
    df['gaming_to_screen'] = df['gaming_hours'] / df['daily_screen_time_hours']
    df['work_to_screen'] = df['work_study_hours'] / df['daily_screen_time_hours']
    df['weekend_to_screen'] = df['weekend_screen_time'] / df['daily_screen_time_hours']
    df['notif_per_open'] = df['notifications_per_day'] / df['app_opens_per_day']
    df['screen_plus_work'] = df['daily_screen_time_hours'] + df['work_study_hours']
    df['leisure_screen'] = df['social_media_hours'] + df['gaming_hours']
    df['sleep_deficit'] = 24 - df['sleep_hours'] - df['daily_screen_time_hours'] - df['work_study_hours']
    df['screen_per_age'] = df['daily_screen_time_hours'] / df['age']
    df['weekday_weekend_diff'] = df['weekend_screen_time'] - df['daily_screen_time_hours']

    ratio_cols = ['social_to_screen','gaming_to_screen','work_to_screen','weekend_to_screen',
                'notif_per_open','screen_plus_work','leisure_screen','sleep_deficit',
                'screen_per_age','weekday_weekend_diff']
    df[ratio_cols] = df[ratio_cols].replace([np.inf,-np.inf], np.nan)
    return df, ratio_cols

train, ratio_cols = feature_engineer(train)
test, _ = feature_engineer(test)

num_cols = base_num_cols + ratio_cols + [f'{c}_missing' for c in base_num_cols+cat_cols] + ['missing_count']

for c in cat_cols:
    train[c] = train[c].astype('category')
    test[c] = test[c].astype('category')
    cats = train[c].cat.categories
    test[c] = test[c].cat.set_categories(cats)

feat_cols = num_cols + cat_cols
X = train[feat_cols]
y = train[target]
X_test = test[feat_cols]

# For xgboost/catboost we need cats encoded as codes (with -1 for missing / NaN handled natively by catboost)
X_cb = X.copy()
for c in cat_cols:
    X_cb[c] = X_cb[c].astype(object).where(X_cb[c].notna(), 'missing').astype(str)
X_test_cb = X_test.copy()
for c in cat_cols:
    X_test_cb[c] = X_test_cb[c].astype(object).where(X_test_cb[c].notna(), 'missing').astype(str)

X_xgb = X.copy()
for c in cat_cols:
    X_xgb[c] = X_xgb[c].cat.codes.replace(-1, np.nan)
X_test_xgb = X_test.copy()
for c in cat_cols:
    X_test_xgb[c] = X_test_xgb[c].cat.codes.replace(-1, np.nan)

n_splits = 3
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X)); pred_lgb = np.zeros(len(X_test))
oof_xgb = np.zeros(len(X)); pred_xgb = np.zeros(len(X_test))
oof_cb  = np.zeros(len(X)); pred_cb  = np.zeros(len(X_test))

lgb_params = dict(
    objective='binary', metric='auc', learning_rate=0.05, num_leaves=63,
    max_depth=-1, min_child_samples=20, subsample=0.85, colsample_bytree=0.7,
    reg_alpha=0.2, reg_lambda=0.5, n_estimators=700, verbosity=-1, n_jobs=1
)
xgb_params = dict(
    objective='binary:logistic', eval_metric='auc', learning_rate=0.06,
    max_depth=6, min_child_weight=5, subsample=0.85, colsample_bytree=0.7,
    reg_alpha=0.2, reg_lambda=1.0, n_estimators=500, tree_method='hist',
    enable_categorical=False, n_jobs=1
)

for fold,(tr_idx,va_idx) in enumerate(skf.split(X,y)):
    y_tr,y_va = y.iloc[tr_idx], y.iloc[va_idx]

    # LightGBM
    X_tr,X_va = X.iloc[tr_idx], X.iloc[va_idx]
    m_lgb = lgb.LGBMClassifier(**lgb_params)
    m_lgb.fit(X_tr,y_tr, eval_set=[(X_va,y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
            categorical_feature=cat_cols)
    p_va = m_lgb.predict_proba(X_va)[:,1]
    oof_lgb[va_idx] = p_va
    pred_lgb += m_lgb.predict_proba(X_test)[:,1] / n_splits
    auc_lgb = roc_auc_score(y_va, p_va)

    # XGBoost
    X_tr_x, X_va_x = X_xgb.iloc[tr_idx], X_xgb.iloc[va_idx]
    m_xgb = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=50)
    m_xgb.fit(X_tr_x, y_tr, eval_set=[(X_va_x, y_va)], verbose=False)
    p_va_x = m_xgb.predict_proba(X_va_x)[:,1]
    oof_xgb[va_idx] = p_va_x
    pred_xgb += m_xgb.predict_proba(X_test_xgb)[:,1] / n_splits
    auc_xgb = roc_auc_score(y_va, p_va_x)

    # CatBoost
    X_tr_c, X_va_c = X_cb.iloc[tr_idx], X_cb.iloc[va_idx]
    m_cb = CatBoostClassifier(iterations=500, learning_rate=0.08, depth=6,
                                l2_leaf_reg=3, loss_function='Logloss', eval_metric='AUC',
                                cat_features=cat_cols, verbose=False, thread_count=1,
                                early_stopping_rounds=50, random_seed=42)
    m_cb.fit(X_tr_c, y_tr, eval_set=(X_va_c, y_va))
    p_va_c = m_cb.predict_proba(X_va_c)[:,1]
    oof_cb[va_idx] = p_va_c
    pred_cb += m_cb.predict_proba(X_test_cb)[:,1] / n_splits
    auc_cb = roc_auc_score(y_va, p_va_c)

    print(f"Fold {fold}: LGB {auc_lgb:.5f} | XGB {auc_xgb:.5f} | CB {auc_cb:.5f}", flush=True)

print("OOF LGB:", roc_auc_score(y, oof_lgb))
print("OOF XGB:", roc_auc_score(y, oof_xgb))
print("OOF CB :", roc_auc_score(y, oof_cb))

# Weighted blend - search simple weights
best_auc, best_w = 0, None
for w1 in np.arange(0,1.01,0.1):
    for w2 in np.arange(0,1.01-w1,0.1):
        w3 = 1-w1-w2
        blend = w1*oof_lgb + w2*oof_xgb + w3*oof_cb
        auc = roc_auc_score(y, blend)
        if auc > best_auc:
            best_auc, best_w = auc, (w1,w2,w3)

print("Best blend weights (lgb,xgb,cb):", best_w, "OOF AUC:", best_auc)

w1,w2,w3 = best_w
final_pred = w1*pred_lgb + w2*pred_xgb + w3*pred_cb

sub = pd.DataFrame({'id': test['id'], 'addicted_label': final_pred})
sub.to_csv('submission_v2.csv', index=False)
print(sub.head())
print("DONE")
