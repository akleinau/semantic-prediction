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

    def pharmacological(self):
        return self.values["pharmacological"]

# if two rules are the same, just the number is different, we want to remove one of them
# eg ["A (<= 12 months), A (<= 15 months)"] -> ["A (<= 12 months)"]
def cleanup_rule(rule: list):

    if len(rule) == 1:
        return rule

    terms_separated = []
    for term_tuple in rule[0]:
        term = term_tuple[0]
        elements = term.split(" (")
        feature = elements[0]
        unit = ""
        if len(elements) > 1:
            elements2 = elements[1].split(" ")
            if len(elements2) > 1:
                comparator = elements2[0]
                value = elements2[1].split(")")[0]

                if len(elements2) > 2:
                    unit = elements2[2].split(")")[0]

            else:
                comparator = ""
                value = elements[1].split(")")[0]

        else:
            value = ""
            comparator = ""

        terms_separated.append({"feature": feature, "comparator": comparator, "value": value, "unit": unit})

    # join terms with same feature
    cleaned_terms = []
    cleaned_features = []
    for term in terms_separated:
        # when this is the first time the feature is seen, add it
        if term["feature"] not in cleaned_features:
            cleaned_terms.append(term)
            cleaned_features.append(term["feature"])
        # if the feature is seen before, join them
        else:
            # find the term in the cleaned_terms
            feature_index = cleaned_features.index(term["feature"])
            prev_term = cleaned_terms[feature_index]

            # if both are <=, take the smaller one
            if term["comparator"] == "<=" and prev_term["comparator"] == "<=":
                if int(term["value"]) < int(prev_term["value"]):
                    cleaned_terms[feature_index]["value"] = term["value"]

            # if both are >=, take the larger one
            elif term["comparator"] == ">=" and prev_term["comparator"] == ">=":
                if int(term["value"]) > int(prev_term["value"]):
                    cleaned_terms[feature_index]["value"] = term["value"]

            # if one is <= and the other is >=, create new comparator -
            elif term["comparator"] == "<=" and prev_term["comparator"] == ">=":
                cleaned_terms[feature_index]["comparator"] = "-"
                cleaned_terms[feature_index]["value_prev"] = prev_term["comparator"]
                cleaned_terms[feature_index]["value"] = term["value"]
            elif term["comparator"] == ">=" and prev_term["comparator"] == "<=":
                cleaned_terms[feature_index]["comparator"] = "-"
                cleaned_terms[feature_index]["value_prev"] = term["comparator"]
                cleaned_terms[feature_index]["value"] = prev_term["value"]

            # if one is <= and the other is -, restrict range
            elif term["comparator"] == "<=" and prev_term["comparator"] == "-":
                if int(term["value"]) < int(prev_term["value"]):
                    cleaned_terms[feature_index]["value"] = term["value"]
                else:
                    cleaned_terms[feature_index]["value"] = prev_term["value"]
            elif term["comparator"] == "-" and prev_term["comparator"] == "<=":
                if int(term["value"]) > int(prev_term["value"]):
                    cleaned_terms[feature_index]["value"] = term["value"]
                else:
                    cleaned_terms[feature_index]["value"] = prev_term["value"]

            # if one is >= and the other is -, restrict range
            elif term["comparator"] == ">=" and prev_term["comparator"] == "-":
                if int(term["value"]) > int(prev_term["value"]):
                    cleaned_terms[feature_index]["value"] = term["value"]
                else:
                    cleaned_terms[feature_index]["value"] = prev_term["value"]
            elif term["comparator"] == "-" and prev_term["comparator"] == ">=":
                if int(term["value"]) < int(prev_term["value"]):
                    cleaned_terms[feature_index]["value"] = term["value"]
                else:
                    cleaned_terms[feature_index]["value"] = prev_term["value"]

    # create the cleaned rules
    cleaned_rule = []
    for term in cleaned_terms:
        if term["comparator"] == "":
            if term["value"] == "":
                cleaned_rule.append([term["feature"] + " " +term["unit"] , 1])
            else:
                cleaned_rule.append([term["feature"] + " (" + term["value"] + " " +term["unit"] + ")", 1])
        elif term["comparator"] == "-":
            cleaned_rule.append([term["feature"] + " (" + term["value_prev"] + term["comparator"] + term["value"] + " " + term["unit"] + ")", 1])
        else:
            cleaned_rule.append([term["feature"] + " (" + term["comparator"] + " " + term["value"] + " " + term["unit"] + ")" , 1])



    return [cleaned_rule, rule[1], rule[2]]  # return the cleaned rule, the impact and the fit


def cleanup_rules(rules):
    return list(map(cleanup_rule, rules))


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
    if input.pharmacological() != "-":
        test[featurenames.index(input.pharmacological())] = True
        test[featurenames.index('11.1 Pharmacological support')] = True

    # run prediction
    extendednames = featurenames + ["not " + n for n in featurenames]
    (testrls ,testfit) = apply_rules(model ,test ,extendednames)
    (ctrlrls ,ctrlfit) = apply_rules(model ,control ,extendednames)

    # clean up rules
    testrls = cleanup_rules(testrls)
    ctrlrls = cleanup_rules(ctrlrls)

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

    return {"testrulestrs": testrulestrs, "ctrlrulestrs": ctrlrulestrs, "testfit": testfit, "ctrlfit": ctrlfit,
            "testonlyimpacts": testonlyimpacts, "testrls": testrls, "ctrlrls": ctrlrls}

example_input = {
    "meanage": 20,
    "proportionfemale": 50,
    "meantobacco": 10,
    "followup": 26,
    "patientrole": 1,
    "verification": 1,
    "outcome": "Abstinence: Continuous ",
    "intervention": [],
    "pharmacological": "-",
    "delivery": [],
    "source": []
}

example_input_instance = InputClass(example_input)

result = predict(example_input_instance)
