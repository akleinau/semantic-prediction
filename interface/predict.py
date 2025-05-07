import inspect
import pathlib
import pickle
import sys
import os

import numpy as np
import pandas as pd
from shiny import App, render, ui

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

### Handle local imports


### Handle local imports
#os.chdir('/var/www/html/semantic-prediction')
#sys.path.append('/var/www/html/semantic-prediction')


src_file_path = inspect.getfile(lambda: None)

PACKAGE_PARENT = pathlib.Path(src_file_path).parent
PACKAGE_PARENT2 = PACKAGE_PARENT.parent
#SCRIPT_DIR = PACKAGE_PARENT2 / "rulenn"
#DATAPR_DIR = PACKAGE_PARENT2 / "dataprocessing"
sys.path.append(str(PACKAGE_PARENT2))
#sys.path.append(str(SCRIPT_DIR))
#sys.path.append(str(DATAPR_DIR))

from base import filter_features
from rulenn.rule_nn import RuleNNModel
from rulenn.apply_rules import apply_rules
from dataprocessing.fuzzysets import FUZZY_SETS

###  Server state

checkpoint = 'examples/model_final.json'
path = 'data/hbcp_gen.pkl'
filters = False

model = RuleNNModel.load(checkpoint)
model.model.eval()  # Run in production mode

with open(path, "rb") as fin:
    raw_features, raw_labels = pd.read_pickle(fin)
raw_features[np.isnan(raw_features)] = 0

if filters:
    features = filter_features(raw_features)
else:
    features = raw_features

#Additional filter based on the loaded model. Maybe the only one really needed?
retainedfeatures = [x for x in features.columns if x[1] in model.variables]
features = features[retainedfeatures]

featurenames = [x[1] for x in features.columns]
featuresemantics = pd.read_csv('data/feature-semantics.csv')


# basically just reformats JSON to class format needed to conform to rest of code
class InputClass:

    def __init__(self, values):
        self.values = values


    def meanage(self):
        return self.values["meanage"]

    def proportionfemale(self):
        return self.values["proportionfemale"]

    def meantobacco(self):
        return self.values["meantobacco"]

    def followup(self):
        return self.values["followup"]

    def patientrole(self):
        return self.values["patientrole"]

    def verification(self):
        return self.values["verification"]

    def outcome(self):
        return self.values["outcome"]

    def intervention(self):
        return self.values["intervention"]

    def delivery(self):
        return self.values["delivery"]

    def source(self):
        return self.values["source"]


# predicts for the given input in InputClass format
def predict(input: InputClass):

    test = features.iloc[0].values
    # Baseline
    test[0: len(test)] = 0
    fuzzynames = ['Mean age',
                  'Proportion identifying as female gender',
                  'Proportion identifying as male gender',
                  'Mean number of times tobacco used',
                  'Combined follow up']
    fuzzyvalues = [input.meanage(),
                   input.proportionfemale(),
                   100 -input.proportionfemale(),
                   input.meantobacco(),
                   input.followup()]
    for fname, fvalue in zip(fuzzynames, fuzzyvalues):
        fs = FUZZY_SETS.get(fname)
        for valname, valfs in list(fs.items()):
            colname = f"{fname} ({valname})"
            if colname in featurenames:
                test[featurenames.index(colname)] = valfs(fvalue)
    test[featurenames.index('aggregate patient role')] = input.patientrole()
    test[featurenames.index('Biochemical verification')] = input.verification()
    if input.outcome() is not None:
        test[featurenames.index(input.outcome())] = True

    # Shared attributes have been set, copy this to the control
    control = [i for i in test]  # deep copy
    control[featurenames.index('control')] = 1

    # Set intervention-specific attributes
    for x in input.intervention():
        test[featurenames.index(x)] = True
    for x in input.delivery():
        test[featurenames.index(x)] = True
    for x in input.source():
        test[featurenames.index(x)] = True
    if '11.1 Pharmacological support' in input.intervention():
        if input.pharmacological() is not None:
            test[featurenames.index(input.pharmacological())] = True

    # run prediction
    extendednames = featurenames + ["not " + n for n in featurenames]
    (testrls ,testfit) = apply_rules(model ,test ,extendednames)
    (ctrlrls ,ctrlfit) = apply_rules(model ,control ,extendednames)

    testimpacts = [a[1] for a in testrls]
    ctrlimpacts = [b[1] for b in ctrlrls]
    testonlyimpacts = [a[1] for a in testrls if a not in ctrlrls]
    testnames = [a[0] for a  in testrls]
    ctrlnames = [b[0] for b in ctrlrls]
    testrulestrs = []
    ctrlrulestrs = []

    NO_RULES = 30

    for i ,ruleslst in enumerate(ctrlnames):
        ruleslststr = [x for (x ,w) in ruleslst] # +"(" +str(round(w,1))+")"
        impact = ctrlimpacts[i]
        rulestr = ' & '.join(ruleslststr)
        rulestr = rulestr + ": " + str(round(impact ,1))
        ctrlrulestrs.append(rulestr)

    for i ,ruleslst in enumerate(testnames):
        ruleslststr = [x for (x ,w) in ruleslst] # + "(" +str(round(w,1))+")"
        impact = testimpacts[i]
        rulestr = ' & '.join(ruleslststr)
        rulestr = rulestr + ": " + str(round(impact ,1))
        if rulestr not in ctrlrulestrs:
            testrulestrs.append(rulestr)

    return testrulestrs

example_input = {
    "meanage": 20,
    "proportionfemale": 50,
    "meantobacco": 10,
    "followup": 26,
    "patientrole": 1,
    "verification": 1,
    "outcome": "Abstinence: Continuous ",
    "intervention": [],
    "delivery": [],
    "source": []
}

example_input_instance = InputClass(example_input)

result = predict(example_input_instance)

print(result)