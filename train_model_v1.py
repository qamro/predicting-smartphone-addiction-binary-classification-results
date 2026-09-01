import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

target = 'addicted_label'
cat_cols = ['gender','stress_level','academic_work_impact']
num_cols = [c for c in train.columns if c not in cat_cols+['id',target]]

for c in cat_cols:
    train[c] = train[c].astype('category')
    test[c] = test[c].astype('category')
    # align categories
    cats = train[c].cat.categories
    test[c] = test[c].cat.set_categories(cats)

X = train[num_cols+cat_cols]
y = train[target]
X_test = test[num_cols+cat_cols]

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
oof = np.zeros(len(X))
preds = np.zeros(len(X_test))
aucs=[]

params = dict(
    objective='binary',
    metric='auc',
    learning_rate=0.03,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    n_estimators=800,
    verbosity=-1
)

for fold,(tr_idx,va_idx) in enumerate(skf.split(X,y)):
    X_tr,X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr,y_va = y.iloc[tr_idx], y.iloc[va_idx]
    model = lgb.LGBMClassifier(**params)
    model.fit(X_tr,y_tr, eval_set=[(X_va,y_va)],
            callbacks=[lgb.early_stopping(100, verbose=False)],
            categorical_feature=cat_cols)
    va_pred = model.predict_proba(X_va)[:,1]
    oof[va_idx] = va_pred
    auc = roc_auc_score(y_va, va_pred)
    aucs.append(auc)
    print(f"Fold {fold} AUC: {auc:.5f}")
    preds += model.predict_proba(X_test)[:,1] / skf.n_splits

print("Mean AUC:", np.mean(aucs))
print("OOF AUC:", roc_auc_score(y, oof))

sub = pd.DataFrame({'id': test['id'], 'addicted_label': preds})
sub.to_csv('submission.csv', index=False)
print(sub.head())
