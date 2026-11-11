"""Load Mayne et al. tabular datasets as feature tables + reuse their templates."""
import sys, types, os, pickle
MAYNE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mayne')
sys.path.insert(0, MAYNE)
if 'openml' not in sys.modules:
    sys.modules['openml'] = types.ModuleType('openml')
from src.templates.heart_disease import HeartDisease
from src.templates.pima_diabetes import PimaDiabetes
from src.templates.breast_cancer_recurrence import BreastCancerRecurrence
from src.templates.income import IncomeDataset
from src.templates.attrition import AttritionDataset
from src.templates.bank_marketing import BankMarketing
CLASSES = {c.to_string(): c for c in [HeartDisease, PimaDiabetes, BreastCancerRecurrence, IncomeDataset, AttritionDataset, BankMarketing]}
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_pilot', 'raw_tables.pkl')

def load_all(refresh=False):
    if os.path.exists(CACHE) and not refresh:
        return pickle.load(open(CACHE, 'rb'))
    import io, contextlib
    out = {}
    for name, cls in CLASSES.items():
        with contextlib.redirect_stdout(io.StringIO()):
            df = cls.load_dataset()
        out[name] = df
    pickle.dump(out, open(CACHE, 'wb'))
    return out

def features(df):
    return [c for c in df.columns if c != 'target']

def describe(name, row):
    df_cols = [c for c in row.index if c != 'target']
    return CLASSES[name].description_generator(0, row, df_cols)

def reference_prompt(name, description, answer_last=False):
    return CLASSES[name].create_reference_prompt(description, answer_last=answer_last)
