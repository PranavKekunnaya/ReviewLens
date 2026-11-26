"""Train on UCI Amazon review sentences; keep the final test set out of model selection."""
from pathlib import Path
import hashlib, json, platform, zipfile
import joblib, numpy as np, pandas as pd, sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
ROOT = Path(__file__).resolve().parent

def clean_data(frame):
    frame = frame.copy()
    frame['key'] = frame.text.str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
    conflicts = frame.groupby('key').label.nunique()
    frame = frame[~frame.key.isin(conflicts[conflicts > 1].index)]
    return frame.drop_duplicates('key').reset_index(drop=True)

def main():
    raw = (ROOT/'data/source.zip').read_bytes()
    with zipfile.ZipFile(ROOT/'data/source.zip') as z:
        name = next(n for n in z.namelist() if n.endswith('/amazon_cells_labelled.txt'))
        rows = [line.rsplit('\t', 1) for line in z.read(name).decode('utf-8').splitlines() if line.strip()]
    frame = pd.DataFrame(rows, columns=['text','label']);frame.label = frame.label.astype(int)
    clean = clean_data(frame)
    train, test = train_test_split(clean, test_size=.2, stratify=clean.label, random_state=42)
    assert not set(train.key) & set(test.key)
    pipe = Pipeline([('tfidf',TfidfVectorizer(sublinear_tf=True)),('classifier',LogisticRegression(max_iter=1000,random_state=42))])
    search = GridSearchCV(pipe, {'tfidf__ngram_range':[(1,1),(1,2)],'classifier__C':[.5,1,4]},
                          cv=StratifiedKFold(5,shuffle=True,random_state=42),scoring='f1_macro',n_jobs=1)
    search.fit(train.text,train.label)
    model=search.best_estimator_;pred=model.predict(test.text)
    dummy=DummyClassifier(strategy='most_frequent').fit(np.zeros((len(train),1)),train.label)
    base=dummy.predict(np.zeros((len(test),1)))
    metrics={'dataset':'UCI Sentiment Labelled Sentences / Amazon cells subset',
             'source_sha256':hashlib.sha256(raw).hexdigest(),'seed':42,'raw_rows':len(frame),'clean_rows':len(clean),
             'train_rows':len(train),'test_rows':len(test),'best_parameters':search.best_params_,
             'cv_macro_f1':search.best_score_,'test_accuracy':accuracy_score(test.label,pred),
             'test_macro_f1':f1_score(test.label,pred,average='macro'),
             'baseline_accuracy':accuracy_score(test.label,base),'baseline_macro_f1':f1_score(test.label,base,average='macro'),
             'confusion_matrix':confusion_matrix(test.label,pred,labels=[0,1]).tolist(),
             'classification_report':classification_report(test.label,pred,output_dict=True),
             'python':platform.python_version(),'sklearn':sklearn.__version__}
    for folder in ['models','reports']:(ROOT/folder).mkdir(exist_ok=True)
    joblib.dump(model,ROOT/'models/sentiment.joblib')
    (ROOT/'reports/metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    pd.DataFrame(search.cv_results_).to_csv(ROOT/'reports/cv_results.csv',index=False)
    result=test[['text','label']].copy();result['prediction']=pred;result['positive_score']=model.predict_proba(test.text)[:,1]
    result.to_csv(ROOT/'reports/test_predictions.csv',index=False)
    result[result.label!=result.prediction].to_csv(ROOT/'reports/errors.csv',index=False)
    split=clean[['text','label','key']].copy();split['split']=np.where(split.key.isin(train.key),'train','test');split.to_csv(ROOT/'data/split.csv',index=False)
    print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
