"""CLI inference, batch CSV scoring, and exact linear-model feature contributions."""
import argparse,json
from pathlib import Path
import joblib,pandas as pd
ROOT=Path(__file__).resolve().parent

def predict(model,text):
    if not isinstance(text,str) or not text.strip():raise ValueError('Review must be nonempty text.')
    if len(text)>10000:raise ValueError('Review exceeds the 10,000-character limit.')
    vec=model.named_steps['tfidf'];clf=model.named_steps['classifier']
    x=vec.transform([text])
    if x.nnz==0:return {'label':'unrecognized','positive_score':None,'evidence':[], 'note':'No known vocabulary; no prediction made.'}
    score=float(model.predict_proba([text])[0,1]);terms=vec.get_feature_names_out()
    weights=[{'term':str(terms[i]),'contribution':float(v*clf.coef_[0,i])} for i,v in zip(x.indices,x.data)]
    weights.sort(key=lambda w:abs(w['contribution']),reverse=True)
    return {'label':'positive' if score>=.5 else 'negative','positive_score':round(score,4),
            'evidence':weights[:6],'note':'Model score, not calibrated confidence. Positive contributions favor positive sentiment.'}

def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--text');g.add_argument('--csv',type=Path);p.add_argument('--output',type=Path,default=Path('predictions.csv'))
    args=p.parse_args()
    path=ROOT/'models/sentiment.joblib'
    if not path.exists():p.error('Run python train.py first.')
    model=joblib.load(path) # Load only this trusted local artifact; never an untrusted pickle/joblib file.
    if args.text is not None:
        try:print(json.dumps(predict(model,args.text),indent=2))
        except ValueError as e:p.error(str(e))
    else:
        df=pd.read_csv(args.csv,keep_default_na=False)
        if 'text' not in df:p.error('CSV must contain a text column.')
        outputs=[]
        for t in df.text:
            try:r=predict(model,t)
            except ValueError as e:r={'label':'invalid','positive_score':None,'note':str(e)}
            outputs.append({k:r.get(k) for k in ['label','positive_score','note']})
        pd.concat([df,pd.DataFrame(outputs).add_prefix('model_')],axis=1).to_csv(args.output,index=False)
        print(f'Saved {len(df)} rows to {args.output}')
if __name__=='__main__':main()
