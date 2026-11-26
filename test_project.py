import unittest
import numpy as np,pandas as pd,joblib
from train import clean_data,ROOT
from predict import predict
class Tests(unittest.TestCase):
    def test_duplicates_and_conflicts(self):
        df=pd.DataFrame({'text':['Good',' good ','Bad','bad','Other'],'label':[1,1,0,1,0]})
        out=clean_data(df)
        self.assertEqual(out.text.tolist(),['Good','Other'])
    def test_no_overlap(self):
        df=pd.read_csv(ROOT/'data/split.csv')
        self.assertFalse(set(df[df.split=='train'].key)&set(df[df.split=='test'].key))
    def test_input_validation(self):
        model=joblib.load(ROOT/'models/sentiment.joblib')
        for s in ['', '   ', 'x'*10001]:
            with self.assertRaises(ValueError):predict(model,s)
        self.assertEqual(predict(model,'zxqvzxqv')['label'],'unrecognized')
    def test_explanation_matches_logit(self):
        model=joblib.load(ROOT/'models/sentiment.joblib');text='great sound'
        r=predict(model,text)
        total=sum(x['contribution'] for x in r['evidence'])+model.named_steps['classifier'].intercept_[0]
        self.assertAlmostEqual(total,float(model.decision_function([text])[0]),places=8)
if __name__=='__main__':unittest.main()
