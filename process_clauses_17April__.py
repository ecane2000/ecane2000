import re
import os
from typing import Any, Dict, List, Tuple, Set


TEXT_NAMES = ["perlat.txt"]
ADJ_DICT_FILE = "adjectives.txt"
ADV_DICT_FILE = "adverbs.txt"
NOUN_DICT_FILE = "nouns.txt"
NUMERAL_DICT_FILE = "numerals.txt"
PREP_DICT_FILE = "prepositions.txt"


class FeatureSet:
    def __init__(self, feature_tuple):
        # Merge the dictionaries in the tuple into a single dictionary
        self.features = {key: val for feature_dict in feature_tuple for key, val in feature_dict.items()}
        # Sort the dictionary based on keys to ensure consistent order
        self.features = {key: self.features[key] for key in sorted(self.features)}

    def __repr__(self):
        return f"FeatureSet(features={self.features})"

class CompoundSet:
    def __init__(self, constituent_set, pos_set, feature_sets):
        self.constituent_set = constituent_set
        self.pos_set = pos_set
        self.feature_sets = feature_sets

    def __repr__(self):
        return f"CompoundSet(constituent_set={self.constituent_set}, pos_set={self.pos_set}, feature_sets={self.feature_sets})"

def word_tokenize(file_obj) -> List[str]:
    token_list = []
    for line in file_obj:
        # Split the line by one or more whitespace characters
        split_tokens = re.split(r'[\s.,;:!?"“”()\[\]]+', line.strip())

        # Iterate through the tokens in the line
        for token in split_tokens:
            # Clean the token by excluding whitespace characters
            match = re.match('(.*[\’|\'])(.*)', token)
            if match:
                # Append both parts of the matched token to your token list
                token_list.append(match.group(1).strip())
                token_list.append(match.group(2).strip())
            else:
                # Check if clean_token is not empty before writing
                if token:
                    # Write each cleaned token as a separate line in the output file
                    token_list.append(token)
    return token_list

def extract_sentences(text_name):
    """Extracts sentences and their clauses from a given text."""
    # Initialize a list to hold sentences, each of which will be a list of clauses
    sentences = []
    for text_name in TEXT_NAMES:
        with open(text_name, 'r', encoding='utf-8') as file:
            text = file.read()
            # Split text into sentences based on common delimiters
            raw_sentences = re.split(r"[.;:!?]", text)
            
            # Iterate over each sentence
            for raw_sentence in raw_sentences:
                # Trim the sentence and check it's not empty
                trimmed_sentence = raw_sentence.strip()
                if trimmed_sentence:
                    # Split the sentence into clauses based on commas, semicolons, or colons
                    raw_clauses = re.split(r"[,;:]", trimmed_sentence)
                    # Clean up clauses: trimming and removing empty ones
                    clauses = [raw_clause.strip() for raw_clause in raw_clauses if raw_clause.strip() != '']
                    # Append the list of cleaned clauses as a representation of the sentence
                    sentences.append(clauses)
    
    return sentences, clauses


def entry_indexing(entry, entry_list):
    # Find the index of the entry in the entry_list
    try:
        entry_index = entry_list.index(entry)
    except ValueError:
        # Entry not found in the list
        return None, None

    # Get the previous entry if it exists, otherwise None
    prev_entry = entry_list[entry_index - 1] if entry_index > 0 else None
    # Get the next entry if it exists, otherwise None
    post_entry = entry_list[entry_index + 1] if entry_index < len(entry_list) - 1 else None

    if prev_entry is not None:
        prev_entry = prev_entry.lower().strip()
    if post_entry is not None:
        post_entry = post_entry.lower().strip()

    return prev_entry, post_entry

def unify_phrase(head_entry, complement_entry, head_dict, complement_dict, valid_phrase_dict, unified_entry):
    unified_entry_dict = {}

    # Retrieve compounds
    head_compound = head_dict.get(head_entry)
    complement_compound = complement_dict.get(complement_entry)
    
    # Validate POS compatibility
    if not head_compound or not complement_compound or (head_compound.pos_set["POS"], complement_compound.pos_set["POS"]) not in valid_phrase_dict:
        return unified_entry_dict  # Return empty dict if basic validation fails

    # Determine unified POS tag
    unified_pos_tag = valid_phrase_dict[(head_compound.pos_set["POS"], complement_compound.pos_set["POS"])]
    pos_unified_entry = {"POS": unified_pos_tag}
    
    # Combine constituent sets
    constituent_unified_entry = {**head_compound.constituent_set, **complement_compound.constituent_set}
    
    # Handling of feature sets
    unified_feature_sets = []
    if head_compound.feature_sets and complement_compound.feature_sets:
        # Merge feature sets if both are present
        for head_feature_set in head_compound.feature_sets:
            for complement_feature_set in complement_compound.feature_sets:
                if head_feature_set and complement_feature_set and features_are_compatible(head_feature_set.features, complement_feature_set.features):
                    merged_features = merge_features(head_feature_set.features, complement_feature_set.features)
                    # Sort merged features dictionary
                    merged_features_sorted = {key: merged_features[key] for key in sorted(merged_features)}
                    unified_feature_sets.append(FeatureSet((merged_features_sorted,)))
    else:
        # Use non-None feature set or leave as empty if both are None
        unified_feature_sets = head_compound.feature_sets or complement_compound.feature_sets or []

    # Create and return the unified entry based on POS
    unified_entry_compound = CompoundSet(constituent_unified_entry, pos_unified_entry, unified_feature_sets)
    
    if pos_unified_entry["POS"] in {"ADV", "CONJ"}:
        # Allow empty feature sets for NP and PP
        unified_entry_dict[unified_entry] = unified_entry_compound
    else:
        # For other POS types, only add if there are valid features
        if unified_feature_sets:
            unified_entry_dict[unified_entry] = unified_entry_compound
        else:
            print("Rejected non-NP/PP due to no features for entry:", unified_entry)
            return None

    return unified_entry_dict


def features_are_compatible(head_features, complement_features):
    for key in head_features:
        head_val = head_features.get(key, 'ok')  # Default 'ok' for missing keys
        complement_val = complement_features.get(key, 'ok')  # Default 'ok' for missing keys
        if not (head_val == 'ok' or complement_val == 'ok' or head_val == complement_val):
            return False
    return True

def merge_features(head_features, complement_features):
    merged = {}
    for key in set(head_features) | set(complement_features):  # Union of keys
        head_val = head_features.get(key, 'ok')
        complement_val = complement_features.get(key, 'ok')
        # Use 'ok' as a wildcard, allowing any value to match
        merged[key] = head_val if complement_val == 'ok' else complement_val
    return merged


def get_clause_tokens(text_name):
    sentences, clauses = extract_sentences(text_name)
    phrased_clauses = []
    for sentence in sentences:
        for clause in sentence:
            clause_tokens = word_tokenize([clause])  # Removed unnecessary list wrapping
            phrased_clause = []  # Initialize the clause list here
            for token in clause_tokens:
                token = token.lower()
                phrased_clause.append(token)
            phrased_clauses.append(phrased_clause)  # Append the completed clause list once
    return phrased_clauses


def phrase_sentence(entry1, entry2, segment):
    separator = "#"  # Separator for joining tokens
    segment_str = separator.join(segment)
    pattern = re.escape(entry1) + re.escape(separator) + re.escape(entry2)
    unified_entry = entry1 + " " + entry2
    phrased_segment_str = re.sub(pattern, unified_entry, segment_str)
    segment = phrased_segment_str.split(separator)
    return segment

def apply_phrase_sentence(entry1, entry2, segment):
    return phrase_sentence(entry1, entry2, segment)

def get_strings_from_keys(dict_keys):
    all_strings_set = set()

    for key in dict_keys:
        if isinstance(key, tuple):
            all_strings_set.add(key)
        else:
            all_strings_set.add((key,))  # Wrap single strings in a tuple

    return all_strings_set


valid_phrase_dict = {
    ("ref_dem", "ref_marker"): "REF_DEM",
    ("noun_sgm", "noun_marker_sgm"): "NP",
    ("noun_sgf", "noun_marker_sgf"): "NP",
    ("noun_pl", "noun_marker_pl"): "NP",
    ("NP", "ADJ"): "NP",
    ("NP", "PRON"): "NP",
    ("prep", "NP"): "PP",
    ("prep", "REF_DEM"): "PP",
    ("NP", "NP"): "NP",
    ("ADJ", "ADV"): "ADJ",
    ("NP", "quantifier"): "NP",
    ("NP", "REF_DEM"): "NP",
    ("NP", "numeral"): "NP",
    ("prep", "numeral"): "PP",
    ("prep", "quantifier"): "PP",
    ("ADJ", "PP"): "ADJ",
}


def read_numerals_from_dictionary() -> tuple[dict[str, CompoundSet], dict[str, CompoundSet]]:
    """
    Reads the adjectives from the dictionary file and returns 2 dictionaries, 1 with adjectives that have articles and 1
    with adjectives that don't have articles.
    """
    numeral_dict = {}
    with open(NUMERAL_DICT_FILE, mode="r", encoding="utf-8") as dict_file:
        for line in dict_file:
            match = re.match(r'^\s*([^,]+)', line)
            numeral = match.group(1).strip()
            # find alternative lemma if there
            alternative_numeral = None

            pos_numeral = ({"POS": "numeral"})
            constituent_një = ({"një": pos_numeral})
            sg_number_dict = {"NUM": "NUM"}
            sg_gender_dict = {"G": "ok"}
            sg_nd_dict = {"ND": "ND"}
            sg_case_dict = {"CASE": "ok"}
            feature_sg_numeral_combinations = []
            combination = (sg_number_dict, sg_gender_dict, sg_nd_dict, sg_case_dict)
            feature_sg_numeral_combinations.append(combination)
            feature_sg_numeral = [FeatureSet(features) for features in feature_sg_numeral_combinations]
            compound_një = CompoundSet(constituent_një, pos_numeral, feature_sg_numeral)
            numeral_dict["një"] = compound_një

            pos_numeral = ({"POS": "numeral"})
            constituent_tre = ({"tre": pos_numeral})
            pl_number_dict = {"NUM": "pl"}
            pl_gender_dict = {"G": "ok"}
            pl_nd_dict = {"ND": "ok"}
            pl_case_dict = {"CASE": "ok"}
            feature_pl_combinations = []
            combination = (pl_number_dict, pl_gender_dict, pl_nd_dict, pl_case_dict)
            feature_pl_combinations.append(combination)
            feature_pl_numeral = [FeatureSet(features) for features in feature_pl_combinations]
            compound_tre = CompoundSet(constituent_tre, pos_numeral, feature_pl_numeral)
            numeral_dict["tre"] = compound_tre            

            constituent_tri = ({"tri": pos_numeral})
            tri_number_dict = {"NUM": "pl"}
            tri_gender_dict = {"G": "f"}
            tri_nd_dict = {"ND": "ok"}
            tri_case_dict = {"CASE": "ok"}
            feature_tri_combinations = []
            combination = (tri_number_dict, tri_gender_dict, tri_nd_dict, tri_case_dict)
            feature_tri_combinations.append(combination)
            feature_tri = [FeatureSet(features) for features in feature_tri_combinations]
            compound_tri = CompoundSet(constituent_tri, pos_numeral, feature_tri)
            numeral_dict["tri"] = compound_tri            

            if numeral == "njëckë":
                numeral_dict["njëckë"] = compound_një
            else:
                pos_numeral = ({"POS": "numeral"})
                constituent_numeral = ({numeral: pos_numeral})
                feature_pl_combinations = []
                combination = (pl_number_dict, pl_gender_dict, pl_nd_dict, pl_case_dict)
                feature_pl_combinations.append(combination)
                feature_pl_numeral = [FeatureSet(features) for features in feature_pl_combinations]
                compound_numeral = CompoundSet(constituent_numeral, pos_numeral, feature_pl_numeral)
                numeral_dict[numeral] = compound_numeral
    return numeral_dict

quantifier_dict = {}

pos_quantifier = ({"POS": "quantifier"})
constituent_asnjë = ({"asnjë": pos_quantifier})
sg_quantifier_combinations = []
number_dict = {"NUM": "NUM"}
gender_dict = {"G": "ok"}
nd_dict = {"ND": "ND"}
case_dict = {"CASE": "ok"}
sg_quantifier_combination = (number_dict, gender_dict, nd_dict, case_dict)
sg_quantifier_combinations.append(sg_quantifier_combination)
feature_sg_quantifier = [FeatureSet(features) for features in sg_quantifier_combinations]
compound_sg_quantifier = CompoundSet(constituent_asnjë, pos_quantifier, feature_sg_quantifier)
quantifier_dict["asnjë"] = compound_sg_quantifier

constituent_ndonjë = ({"ndonjë": pos_quantifier})
compound_ndonjë = CompoundSet(constituent_ndonjë, pos_quantifier, feature_sg_quantifier)
quantifier_dict["ndonjë"] = compound_ndonjë

constituent_çdo = ({"çdo": pos_quantifier})
çdo_compound = CompoundSet(constituent_çdo, pos_quantifier, feature_sg_quantifier)
quantifier_dict["çdo"] = çdo_compound

pl_quantifier_combinations = []
constituent_disa = ({"disa": pos_quantifier})
number_dict = {"NUM": "pl"}
gender_dict = {"G": "ok"}
nd_dict = {"ND": "ND"}
case_dict = {"CASE": "ok"}
pl_quantifier_combination = (number_dict, gender_dict, nd_dict, case_dict)
pl_quantifier_combinations.append(pl_quantifier_combination)
feature_pl_quantifier = [FeatureSet(features) for features in pl_quantifier_combinations]
disa_compound = CompoundSet(constituent_disa, pos_quantifier, feature_pl_quantifier)
quantifier_dict["disa"] = disa_compound

constituent_ca_quantifier = ({"ca": pos_quantifier})
feature_pl_quantifier = [FeatureSet(features) for features in pl_quantifier_combinations]
ca_compound = CompoundSet(constituent_ca_quantifier, pos_quantifier, feature_pl_quantifier)
quantifier_dict["ca"] = ca_compound

constituent_shumë_quantifier = ({"shumë": pos_quantifier})
shumë_compound = CompoundSet(constituent_shumë_quantifier, pos_quantifier, feature_pl_quantifier)
quantifier_dict["shumë"] = shumë_compound

def read_prepositions_from_dictionary() -> tuple[dict[str, CompoundSet], dict[str, CompoundSet]]:

    prep_dict = {}
    with open(PREP_DICT_FILE, mode="r", encoding="utf-8") as dict_file:
        for line in dict_file:
            prep, cases = line.strip().split(",")
            prep_pos = ({"POS": "prep"})
            prep_constituent = ({prep: prep_pos})
            prep_feature_combinations = []
            prep_number_dict = {"NUM": "ok"}
            prep_gender_dict = {"G": "ok"}
            prep_nd_dict = {"ND": "ok"}
            prep_case_dict = {"CASE": cases}
            combination = (prep_number_dict, prep_gender_dict, prep_nd_dict, prep_case_dict)
            prep_feature_combinations.append(combination)
            prep_feature = [FeatureSet(features) for features in prep_feature_combinations]
           
            prep_compound = CompoundSet(prep_constituent, prep_pos, prep_feature)
            prep_dict[prep] = prep_compound            
        
    return prep_dict

ref_dem_list = ["a", "k"]
ref_dem_dict = {}
k_ref_pos = ({"POS": "ref_dem"})
k_ref_constituent = ({"k": k_ref_pos})
k_feature_combinations = []
k_ref_number_dict = {"NUM": "ok"}
k_ref_gender_dict = {"G": "ok"}
k_ref_nd_dict = {"ND": "ND"}
k_ref_case_dict = {"CASE": "ok"}
k_combination = (k_ref_number_dict, k_ref_gender_dict, k_ref_nd_dict, k_ref_case_dict)
k_feature_combinations.append(k_combination)
k_ref_feature = [FeatureSet(features) for features in k_feature_combinations]
k_ref_compound = CompoundSet(k_ref_constituent, k_ref_pos, k_ref_feature)
ref_dem_dict["k"] = k_ref_compound

a_ref_pos = ({"POS": "ref_dem"})
a_ref_constituent = ({"a": a_ref_pos})
a_feature_combinations = []
a_ref_number_dict = {"NUM": "ok"}
a_ref_gender_dict = {"G": "ok"}
a_ref_nd_dict = {"ND": "ND"}
a_ref_case_dict = {"CASE": "ok"}
a_combination = (k_ref_number_dict, k_ref_gender_dict, k_ref_nd_dict, k_ref_case_dict)
a_feature_combinations.append(k_combination)
a_ref_feature = [FeatureSet(features) for features in a_feature_combinations]
a_ref_compound = CompoundSet(a_ref_constituent, a_ref_pos, a_ref_feature)
ref_dem_dict["a"] = a_ref_compound

ref_marker_list = ["i", "o", "të", "ta", "to", "tij", "saj", "tyre" ]
ref_marker_dict = {}
i_ref_marker_pos = ({"POS": "ref_marker"})
i_ref_marker_constituent = ({"i": i_ref_marker_pos})
i_ref_marker_combinations = []
i_ref_marker_number_dict = {"NUM": "NUM"}
i_ref_marker_gender_dict = {"G": "G"}
i_ref_marker_nd_dict = {"ND": "ND"}
i_ref_marker_case_dict = {"CASE": "CASE"}
i_combination = (i_ref_marker_number_dict, i_ref_marker_gender_dict, i_ref_marker_nd_dict, i_ref_marker_case_dict)
i_ref_marker_combinations.append(i_combination)
# Initializing FeatureSet for each feature set
i_ref_marker_feature = [FeatureSet(features) for features in i_ref_marker_combinations]
i_ref_marker_compound = CompoundSet(i_ref_marker_constituent, i_ref_marker_pos, i_ref_marker_feature)
ref_marker_dict["i"] = i_ref_marker_compound

o_ref_marker_pos = ({"POS": "ref_marker"})
o_ref_marker_constituent = ({"o": o_ref_marker_pos})
o_ref_marker_combinations = []
o_ref_marker_number_dict = {"NUM": "NUM"}
o_ref_marker_gender_dict = {"G": "f"}
o_ref_marker_nd_dict = {"ND": "ND"}
o_ref_marker_case_dict = {"CASE": "CASE"}
# Initializing FeatureSet for each feature set
o_combination = (o_ref_marker_number_dict, o_ref_marker_gender_dict, o_ref_marker_nd_dict, o_ref_marker_case_dict)
o_ref_marker_combinations.append(o_combination)
# Initializing FeatureSet for each feature set
o_ref_marker_feature = [FeatureSet(features) for features in o_ref_marker_combinations]
o_ref_marker_compound = CompoundSet(o_ref_marker_constituent, o_ref_marker_pos, o_ref_marker_feature)
ref_marker_dict["o"] = o_ref_marker_compound

tij_ref_marker_pos = ({"POS": "ref_marker"})
tij_ref_marker_constituent = ({"tij": tij_ref_marker_pos})
tij_ref_marker_combinations = []
tij_ref_marker_number_dict = {"NUM": "NUM"}
tij_ref_marker_gender_dict = {"G": "G"}
tij_ref_marker_nd_dict = {"ND": "ND"}
tij_ref_marker_case_dict = {"CASE": "dat"}

# Initializing FeatureSet for each feature set
tij_combination = (tij_ref_marker_number_dict, tij_ref_marker_gender_dict, tij_ref_marker_nd_dict, tij_ref_marker_case_dict)
tij_ref_marker_combinations.append(tij_combination)
# Initializing FeatureSet for each feature set
tij_ref_marker_feature = [FeatureSet(features) for features in tij_ref_marker_combinations]
tij_ref_marker_compound = CompoundSet(tij_ref_marker_constituent, tij_ref_marker_pos, tij_ref_marker_feature)
ref_marker_dict["tij"] = tij_ref_marker_compound

saj_ref_marker_pos = ({"POS": "ref_marker"})
saj_ref_marker_constituent = ({"saj": saj_ref_marker_pos})
saj_ref_marker_combinations = []
saj_ref_marker_number_dict = {"NUM": "NUM"}
saj_ref_marker_gender_dict = {"G": "f"}
saj_ref_marker_nd_dict = {"ND": "ND"}
saj_ref_marker_case_dict = {"CASE": "dat"}
saj_ref_marker_feature_tuple = (saj_ref_marker_number_dict, saj_ref_marker_gender_dict, saj_ref_marker_nd_dict, saj_ref_marker_case_dict)

# Initializing FeatureSet for each feature set
saj_combination = (saj_ref_marker_number_dict, saj_ref_marker_gender_dict, saj_ref_marker_nd_dict, saj_ref_marker_case_dict)
saj_ref_marker_combinations.append(saj_combination)
# Initializing FeatureSet for each feature set
saj_ref_marker_feature = [FeatureSet(features) for features in saj_ref_marker_combinations]
saj_ref_marker_compound = CompoundSet(saj_ref_marker_constituent, saj_ref_marker_pos, saj_ref_marker_feature)
ref_marker_dict["saj"] = saj_ref_marker_compound

të_ref_marker_pos = ({"POS": "ref_marker"})
të_ref_marker_constituent = ({"të": të_ref_marker_pos})
të_ref_marker_combinations = []
të_ref_marker_number_dict = {"NUM": "NUM"}
të_ref_marker_gender_dict = {}
të_ref_marker_nd_dict = {"ND": "ND"}
të_ref_marker_case_dict = {"CASE": "acc"}
të_combination = (të_ref_marker_number_dict, të_ref_marker_gender_dict, të_ref_marker_nd_dict, të_ref_marker_case_dict)
të_ref_marker_combinations.append(të_combination)
# Initializing FeatureSet for each feature set
të_ref_marker_feature = [FeatureSet(features) for features in të_ref_marker_combinations]
të_ref_marker_compound = CompoundSet(të_ref_marker_constituent, të_ref_marker_pos, të_ref_marker_feature)
ref_marker_dict["të"] = të_ref_marker_compound


tyre_ref_marker_pos = ({"POS": "ref_marker"})
tyre_ref_marker_constituent = ({"tyre": tyre_ref_marker_pos})
tyre_ref_marker_combinations = []
tyre_ref_marker_number_dict = {"NUM": "pl"}
tyre_ref_marker_gender_dict = {}
tyre_ref_marker_nd_dict = {"ND": "ND"}
tyre_ref_marker_case_dict = {"CASE": "dat"}
tyre_combination = (tyre_ref_marker_number_dict, tyre_ref_marker_gender_dict, tyre_ref_marker_nd_dict, tyre_ref_marker_case_dict)
tyre_ref_marker_combinations.append(tyre_combination)
# Initializing FeatureSet for each feature set
tyre_ref_marker_feature = [FeatureSet(features) for features in tyre_ref_marker_combinations]
tyre_ref_marker_compound = CompoundSet(tyre_ref_marker_constituent, tyre_ref_marker_pos, tyre_ref_marker_feature)
ref_marker_dict["tyre"] = tyre_ref_marker_compound

to_ref_marker_pos = ({"POS": "ref_marker"})
to_ref_marker_constituent = ({"to": to_ref_marker_pos})
to_ref_marker_combinations = []
to_ref_marker_number_dict = {"NUM": "pl"}
to_ref_marker_gender_dict = {"G": "f"}
to_ref_marker_nd_dict = {"ND": "ND"}
to_ref_marker_case_dict = {"CASE": "CASE"}
to_combination = (to_ref_marker_number_dict, to_ref_marker_gender_dict, to_ref_marker_nd_dict, to_ref_marker_case_dict)
to_ref_marker_combinations.append(to_combination)
# Initializing FeatureSet for each feature set
to_ref_marker_feature = [FeatureSet(features) for features in to_ref_marker_combinations]
to_ref_marker_compound = CompoundSet(to_ref_marker_constituent, to_ref_marker_pos, to_ref_marker_feature)
ref_marker_dict["to"] = to_ref_marker_compound

ta_ref_marker_pos = ({"POS": "ref_marker"})
ta_ref_marker_constituent = ({"ta": ta_ref_marker_pos})
ta_ref_marker_combinations = []
ta_ref_marker_number_dict = {"NUM": "pl"}
ta_ref_marker_gender_dict = {"G": "G"}
ta_ref_marker_nd_dict = {"ND": "ND"}
ta_ref_marker_case_dict = {"CASE": "CASE"}
ta_combination = (ta_ref_marker_number_dict, ta_ref_marker_gender_dict, ta_ref_marker_nd_dict, ta_ref_marker_case_dict)
ta_ref_marker_combinations.append(ta_combination)
# Initializing FeatureSet for each feature set
ta_ref_marker_feature = [FeatureSet(features) for features in ta_ref_marker_combinations]
ta_ref_marker_compound = CompoundSet(ta_ref_marker_constituent, ta_ref_marker_pos, ta_ref_marker_feature)
ref_marker_dict["ta"] = ta_ref_marker_compound

adjust_merge_dict = {
    ("k", "i"): "ky",
    ("k", "o"): "kjo",
    ("a", "o"): "ajo",
    ("k", "të"): "këtë",
    ("a", "të"): "atë",
    ("k", "tij"): "këtij",
    ("k", "saj"): "kësaj",
    ("k", "ta"): "këta",
    ("k", "to"): "këto",
    ("k", "tyre"): "këtyre"
}

def get_referents(list_referents, list_markers, merger_dictionary, dictionary_referents, dictionary_markers):
    ref_dict = {}
    referent_dict = {}


    for referent in list_referents:
        for marker in list_markers:
            unified_referent = (referent, marker)
            if unified_referent in merger_dictionary:
                unified_referent = merger_dictionary[unified_referent]
                ref_dict = unify_phrase(referent, marker, dictionary_referents, dictionary_markers, valid_phrase_dict, unified_referent)
                referent_dict.update(ref_dict)
            else:
                unified_referent = (referent+marker)
                ref_dict = unify_phrase(referent, marker, dictionary_referents, dictionary_markers, valid_phrase_dict, unified_referent)
                referent_dict.update(ref_dict)

    return referent_dict

def annotate_referents(all_tokens, dictionary_referents):
    annotated_referents = {}
    for token in all_tokens:
        if token in dictionary_referents:
            annotated_referents[token] = dictionary_referents[token]
    return annotated_referents

# opening noun_marker_sgm_dict
noun_marker_sgm_dict = {}

# populating marker_dict
pos_noun_marker = ({"POS": "noun_marker_sgm"})
constituent_noun_marker = ({"": pos_noun_marker})
feature_noun_marker_combinations = []
number_dict = {"NUM": "NUM"}
gender_dict = {"G": "G"}
case_dict = {"CASE": "CASE"}
nd_dict = {"ND": "ND"}
combination = (number_dict, gender_dict, nd_dict, case_dict)
feature_noun_marker_combinations.append(combination)
case_dict = {"CASE": "acc"}
combination = (number_dict, gender_dict, nd_dict, case_dict)
feature_noun_marker_combinations.append(combination)

feature_noun_marker = [FeatureSet(features) for features in feature_noun_marker_combinations]
compound_noun_marker = CompoundSet(constituent_noun_marker, pos_noun_marker, feature_noun_marker)
noun_marker_sgm_dict[""] = compound_noun_marker

pos_u_noun_marker = ({"POS": "noun_marker_sgm"})
constituent_u_noun_marker = ({"": pos_u_noun_marker})
feature_u_noun_marker_combinations = []
number_dict = {"NUM": "NUM"}
gender_dict = {"G": "G"}
case_dict = {"CASE": "CASE"}
nd_dict = {"ND": "def"}
combination = (number_dict, gender_dict, nd_dict, case_dict)
feature_u_noun_marker_combinations.append(combination)
case_dict = {"CASE": "dat"}
nd_dict = {"ND": "ND"}
combination = (number_dict, gender_dict, nd_dict, case_dict)
feature_u_noun_marker_combinations.append(combination)

feature_u_noun_marker = [FeatureSet(features) for features in feature_u_noun_marker_combinations]
compound_u_noun_marker = CompoundSet(constituent_u_noun_marker, pos_u_noun_marker, feature_u_noun_marker)
noun_marker_sgm_dict["u"] = compound_u_noun_marker
    
pos_i_noun_marker = ({"POS": "noun_marker_sgm"})
constituent_i_noun_marker = ({"": pos_i_noun_marker})
feature_i_noun_marker_combinations = []
number_dict = {"NUM": "NUM"}
gender_dict = {"G": "G"}
case_dict = {"CASE": "CASE"}
nd_dict = {"ND": "def"}
combination = (number_dict, gender_dict, nd_dict, case_dict)
feature_i_noun_marker_combinations.append(combination)
case_dict = {"CASE": "dat"}
nd_dict = {"ND": "ND"}
combination = (number_dict, gender_dict, nd_dict, case_dict)
feature_i_noun_marker_combinations.append(combination)

feature_i_noun_marker = [FeatureSet(features) for features in feature_i_noun_marker_combinations]
compound_i_noun_marker = CompoundSet(constituent_i_noun_marker, pos_i_noun_marker, feature_i_noun_marker)
noun_marker_sgm_dict["i"] = compound_i_noun_marker

pos_ut_noun_marker = ({"POS": "noun_marker_sgm"})
constituent_ut_noun_marker = ({"": pos_ut_noun_marker})
feature_ut_noun_marker_combinations = []
number_dict = {"NUM": "NUM"}
gender_dict = {"G": "G"}
case_dict = {"CASE": "dat"}
nd_dict = {"ND": "def"}
combination = (number_dict, gender_dict, nd_dict, case_dict)
feature_ut_noun_marker_combinations.append(combination)

feature_ut_noun_marker = [FeatureSet(features) for features in feature_ut_noun_marker_combinations]
compound_ut_noun_marker = CompoundSet(constituent_ut_noun_marker, pos_ut_noun_marker, feature_ut_noun_marker)
noun_marker_sgm_dict["ut"] = compound_ut_noun_marker

pos_it_noun_marker = ({"POS": "noun_marker_sgm"})
constituent_it_noun_marker = ({"": pos_it_noun_marker})
feature_it_noun_marker_combinations = []
number_dict = {"NUM": "NUM"}
gender_dict = {"G": "G"}
case_dict = {"CASE": "dat"}
nd_dict = {"ND": "def"}
combination = (number_dict, gender_dict, nd_dict, case_dict)
feature_it_noun_marker_combinations.append(combination)

feature_it_noun_marker = [FeatureSet(features) for features in feature_it_noun_marker_combinations]
compound_it_noun_marker = CompoundSet(constituent_it_noun_marker, pos_it_noun_marker, feature_it_noun_marker)
noun_marker_sgm_dict["it"] = compound_it_noun_marker

pos_un_noun_marker = ({"POS": "noun_marker_sgm"})
constituent_un_noun_marker = ({"": pos_un_noun_marker})
feature_un_noun_marker_combinations = []
number_dict = {"NUM": "NUM"}
gender_dict = {"G": "G"}
case_dict = {"CASE": "acc"}
nd_dict = {"ND": "def"}
combination = (number_dict, gender_dict, nd_dict, case_dict)
feature_un_noun_marker_combinations.append(combination)

feature_un_noun_marker = [FeatureSet(features) for features in feature_un_noun_marker_combinations]
compound_un_noun_marker = CompoundSet(constituent_un_noun_marker, pos_un_noun_marker, feature_un_noun_marker)
noun_marker_sgm_dict["un"] = compound_un_noun_marker

pos_in_noun_marker = ({"POS": "noun_marker_sgm"})
constituent_in_noun_marker = ({"": pos_in_noun_marker})
feature_in_noun_marker_combinations = []
number_dict = {"NUM": "NUM"}
gender_dict = {"G": "G"}
case_dict = {"CASE": "acc"}
nd_dict = {"ND": "def"}
combination = (number_dict, gender_dict, nd_dict, case_dict)
feature_in_noun_marker_combinations.append(combination)

feature_in_noun_marker = [FeatureSet(features) for features in feature_in_noun_marker_combinations]
compound_in_noun_marker = CompoundSet(constituent_in_noun_marker, pos_in_noun_marker, feature_in_noun_marker)
noun_marker_sgm_dict["in"] = compound_in_noun_marker

# opening noun_marker_sgf_dict
noun_marker_sgf_dict = {}
pos_noun_marker = ({"POS": "noun_marker_sgf"})
constituent_noun_marker = ({"": pos_noun_marker})
feature_noun_marker_combinations = []
number_dict = {"NUM": "NUM"}
gender_dict = {"G": "f"}
case_dict = {"CASE": "CASE"}
nd_dict = {"ND": "ND"}
combination = (number_dict, gender_dict, nd_dict, case_dict)
feature_noun_marker_combinations.append(combination)

case_dict = {"CASE": "acc"}
combination = (number_dict, gender_dict, nd_dict, case_dict)
feature_noun_marker_combinations.append(combination)

feature_noun_marker = [FeatureSet(features) for features in feature_noun_marker_combinations]
compound_noun_marker = CompoundSet(constituent_noun_marker, pos_noun_marker, feature_noun_marker)
noun_marker_sgf_dict[""] = compound_noun_marker

pos_e_noun_marker = ({"POS": "noun_marker_sgf"})
constituent_e_noun_marker = ({"e": pos_e_noun_marker})
feature_e_noun_marker_combinations = []
number_dict = {"NUM": "NUM"}
gender_dict = {"G": "f"}
case_dict = {"CASE": "dat"}
nd_dict = {"ND": "ND"}
combination = (number_dict, gender_dict, nd_dict, case_dict)
feature_e_noun_marker_combinations.append(combination)

feature_e_noun_marker = [FeatureSet(features) for features in feature_e_noun_marker_combinations]
compound_e_noun_marker = CompoundSet(constituent_e_noun_marker, pos_e_noun_marker, feature_e_noun_marker)
noun_marker_sgf_dict["e"] = compound_e_noun_marker

pos_je_noun_marker = ({"POS": "noun_marker_sgf"})
constituent_je_noun_marker = ({"je": pos_je_noun_marker})
feature_je_noun_marker_combinations = []
number_dict = {"NUM": "NUM"}
gender_dict = {"G": "f"}
case_dict = {"CASE": "dat"}
nd_dict = {"ND": "ND"}
combination = (number_dict, gender_dict, nd_dict, case_dict)
feature_je_noun_marker_combinations.append(combination)

feature_je_noun_marker = [FeatureSet(features) for features in feature_je_noun_marker_combinations]
compound_je_noun_marker = CompoundSet(constituent_je_noun_marker, pos_je_noun_marker, feature_je_noun_marker)
noun_marker_sgf_dict["je"] = compound_je_noun_marker

pos_a_noun_marker = ({"POS": "noun_marker_sgf"})
constituent_a_noun_marker = ({"a": pos_a_noun_marker})
feature_a_noun_marker_combinations = []
number_dict = {"NUM": "NUM"}
gender_dict = {"G": "f"}
case_dict = {"CASE": "CASE"}
nd_dict = {"ND": "def"}
combination = (number_dict, gender_dict, nd_dict, case_dict)
feature_a_noun_marker_combinations.append(combination)

feature_a_noun_marker = [FeatureSet(features) for features in feature_a_noun_marker_combinations]
compound_a_noun_marker = CompoundSet(constituent_a_noun_marker, pos_a_noun_marker, feature_a_noun_marker)
noun_marker_sgf_dict["a"] = compound_a_noun_marker

pos_ja_noun_marker = ({"POS": "noun_marker_sgf"})
constituent_ja_noun_marker = ({"ja": pos_ja_noun_marker})
feature_ja_noun_marker_combinations = []
number_dict = {"NUM": "NUM"}
gender_dict = {"G": "f"}
case_dict = {"CASE": "CASE"}
nd_dict = {"ND": "def"}
combination = (number_dict, gender_dict, nd_dict, case_dict)
feature_ja_noun_marker_combinations.append(combination)

feature_ja_noun_marker = [FeatureSet(features) for features in feature_ja_noun_marker_combinations]
compound_ja_noun_marker = CompoundSet(constituent_ja_noun_marker, pos_ja_noun_marker, feature_ja_noun_marker)
noun_marker_sgf_dict["ja"] = compound_ja_noun_marker

pos_n_noun_marker = ({"POS": "noun_marker_sgf"})
constituent_n_noun_marker = ({"n": pos_n_noun_marker})
feature_n_noun_marker_combinations = []
number_dict = {"NUM": "NUM"}
gender_dict = {"G": "f"}
case_dict = {"CASE": "acc"}
nd_dict = {"ND": "def"}
combination = (number_dict, gender_dict, nd_dict, case_dict)
feature_n_noun_marker_combinations.append(combination)

feature_n_noun_marker = [FeatureSet(features) for features in feature_n_noun_marker_combinations]
compound_n_noun_marker = CompoundSet(constituent_n_noun_marker, pos_n_noun_marker, feature_n_noun_marker)
noun_marker_sgf_dict["n"] = compound_n_noun_marker

pos_në_noun_marker = ({"POS": "noun_marker_sgf"})
constituent_në_noun_marker = ({"në": pos_në_noun_marker})
feature_në_noun_marker_combinations = []
number_dict = {"NUM": "NUM"}
gender_dict = {"G": "f"}
case_dict = {"CASE": "acc"}
nd_dict = {"ND": "def"}
combination = (number_dict, gender_dict, nd_dict, case_dict)
feature_në_noun_marker_combinations.append(combination)

feature_në_noun_marker = [FeatureSet(features) for features in feature_në_noun_marker_combinations]
compound_në_noun_marker = CompoundSet(constituent_në_noun_marker, pos_në_noun_marker, feature_në_noun_marker)
noun_marker_sgf_dict["në"] = compound_në_noun_marker

pos_së_noun_marker = ({"POS": "noun_marker_sgf"})
constituent_së_noun_marker = ({"së": pos_së_noun_marker})
feature_së_noun_marker_combinations = []
number_dict = {"NUM": "NUM"}
gender_dict = {"G": "f"}
case_dict = {"CASE": "dat"}
nd_dict = {"ND": "def"}
combination = (number_dict, gender_dict, nd_dict, case_dict)
feature_së_noun_marker_combinations.append(combination)

feature_së_noun_marker = [FeatureSet(features) for features in feature_së_noun_marker_combinations]
compound_së_noun_marker = CompoundSet(constituent_së_noun_marker, pos_së_noun_marker, feature_së_noun_marker)
noun_marker_sgf_dict["së"] = compound_së_noun_marker

pos_s_noun_marker = ({"POS": "noun_marker_sgf"})
constituent_s_noun_marker = ({"s": pos_s_noun_marker})
feature_s_noun_marker_combinations = []
number_dict = {"NUM": "NUM"}
gender_dict = {"G": "f"}
case_dict = {"CASE": "dat"}
nd_dict = {"ND": "def"}
combination = (number_dict, gender_dict, nd_dict, case_dict)
feature_s_noun_marker_combinations.append(combination)

feature_s_noun_marker = [FeatureSet(features) for features in feature_s_noun_marker_combinations]
compound_s_noun_marker = CompoundSet(constituent_s_noun_marker, pos_s_noun_marker, feature_s_noun_marker)
noun_marker_sgf_dict["s"] = compound_s_noun_marker

# opening noun_marker_pl_dict
noun_marker_pl_dict = {}

pos_noun_marker = ({"POS": "noun_marker_pl"})
constituent_noun_marker = ({"": pos_noun_marker})
feature_noun_marker_combinations = []
number_dict = {"NUM": "pl"}
gender_dict = {"G": "f"}
case_dict = {"CASE": "CASE"}
nd_dict = {"ND": "ND"}
combination = (number_dict, gender_dict, nd_dict, case_dict)
feature_noun_marker_combinations.append(combination)

gender_dict = {"G": "f"}
case_dict = {"CASE": "acc"}
combination = (number_dict, gender_dict, nd_dict, case_dict)
feature_noun_marker_combinations.append(combination)

gender_dict = {"G": "G"}
case_dict = {"CASE": "CASE"}
combination = (number_dict, gender_dict, nd_dict, case_dict)
feature_noun_marker_combinations.append(combination)

gender_dict = {"G": "G"}
case_dict = {"CASE": "acc"}
combination = (number_dict, gender_dict, nd_dict, case_dict)
feature_noun_marker_combinations.append(combination)

feature_noun_marker = [FeatureSet(features) for features in feature_noun_marker_combinations]
compound_noun_marker = CompoundSet(constituent_noun_marker, pos_noun_marker, feature_noun_marker)
noun_marker_sgf_dict[""] = compound_noun_marker

pos_ve_noun_marker = ({"POS": "noun_marker_pl"})
constituent_ve_noun_marker = ({"ve": pos_ve_noun_marker})
feature_ve_noun_marker_combinations = []
number_dict = {"NUM": "pl"}
gender_dict = {"G": "f"}
case_dict = {"CASE": "dat"}
nd_dict = {"ND": "ok"}
combination = (number_dict, gender_dict, nd_dict, case_dict)
feature_ve_noun_marker_combinations.append(combination)

gender_dict = {"G": "G"}
combination = (number_dict, gender_dict, nd_dict, case_dict)
feature_ve_noun_marker_combinations.append(combination)

feature_ve_noun_marker = [FeatureSet(features) for features in feature_ve_noun_marker_combinations]
compound_ve_noun_marker = CompoundSet(constituent_ve_noun_marker, pos_ve_noun_marker, feature_ve_noun_marker)
noun_marker_pl_dict["ve"] = compound_ve_noun_marker

pos_sh_noun_marker = ({"POS": "noun_marker_pl"})
constituent_sh_noun_marker = ({"sh": pos_sh_noun_marker})
feature_sh_noun_marker_combinations = []
number_dict = {"NUM": "pl"}
gender_dict = {"G": "f"}
case_dict = {"CASE": "dat"}
nd_dict = {"ND": "ND"}          
combination = (number_dict, gender_dict, nd_dict, case_dict)
feature_sh_noun_marker_combinations.append(combination)

gender_dict = {"G": "G"}
combination = (number_dict, gender_dict, nd_dict, case_dict)
feature_sh_noun_marker_combinations.append(combination)

feature_sh_noun_marker = [FeatureSet(features) for features in feature_sh_noun_marker_combinations]
compound_sh_noun_marker = CompoundSet(constituent_sh_noun_marker, pos_sh_noun_marker, feature_sh_noun_marker)
noun_marker_pl_dict["sh"] = compound_sh_noun_marker

pos_ish_noun_marker = ({"POS": "noun_marker_pl"})
constituent_ish_noun_marker = ({"ish": pos_ish_noun_marker})
feature_ish_noun_marker_combinations = []
number_dict = {"NUM": "pl"}
gender_dict = {"G": "f"}
case_dict = {"CASE": "dat"}
nd_dict = {"ND": "ND"}
combination = (number_dict, gender_dict, nd_dict, case_dict)
feature_ish_noun_marker_combinations.append(combination)

gender_dict = {"G": "G"}
combination = (number_dict, gender_dict, nd_dict, case_dict)
feature_ish_noun_marker_combinations.append(combination)

feature_ish_noun_marker = [FeatureSet(features) for features in feature_ish_noun_marker_combinations]
compound_ish_noun_marker = CompoundSet(constituent_ish_noun_marker, pos_ish_noun_marker, feature_ish_noun_marker)
noun_marker_pl_dict["ish"] = compound_ish_noun_marker

pos_t_noun_marker = ({"POS": "noun_marker_pl"})
constituent_t_noun_marker = ({"t": pos_t_noun_marker})
feature_t_noun_marker_combinations = []
number_dict = {"NUM": "pl"}
gender_dict = {"G": "f"}
case_dict = {"CASE": "CASE"}
nd_dict = {"ND": "def"}
combination = (number_dict, gender_dict, nd_dict, case_dict)
feature_t_noun_marker_combinations.append(combination)

gender_dict = {"G": "f"}
case_dict = {"CASE": "acc"}
nd_dict = {"ND": "def"}
combination = (number_dict, gender_dict, nd_dict, case_dict)
feature_t_noun_marker_combinations.append(combination)

gender_dict = {"G": "G"}
case_dict = {"CASE": "CASE"}
combination = (number_dict, gender_dict, nd_dict, case_dict)
feature_t_noun_marker_combinations.append(combination)

case_dict = {"CASE": "acc"}
combination = (number_dict, gender_dict, nd_dict, case_dict)
feature_t_noun_marker_combinations.append(combination)
feature_t_noun_marker = [FeatureSet(features) for features in feature_t_noun_marker_combinations]
compound_t_noun_marker = CompoundSet(constituent_t_noun_marker, pos_t_noun_marker, feature_t_noun_marker)
noun_marker_pl_dict["t"] = compound_t_noun_marker

pos_it_noun_marker = ({"POS": "noun_marker_pl"})
constituent_it_noun_marker = ({"it": pos_it_noun_marker})
feature_it_noun_marker_combinations = []
number_dict = {"NUM": "pl"}
gender_dict = {"G": "f"}
case_dict = {"CASE": "CASE"}
nd_dict = {"ND": "def"}
combination = (number_dict, gender_dict, nd_dict, case_dict)
feature_it_noun_marker_combinations.append(combination)

gender_dict = {"G": "f"}
case_dict = {"CASE": "acc"}
nd_dict = {"ND": "def"}
combination = (number_dict, gender_dict, nd_dict, case_dict)
feature_it_noun_marker_combinations.append(combination)

gender_dict = {"G": "G"}
case_dict = {"CASE": "CASE"}
combination = (number_dict, gender_dict, nd_dict, case_dict)
feature_it_noun_marker_combinations.append(combination)

case_dict = {"CASE": "acc"}
combination = (number_dict, gender_dict, nd_dict, case_dict)
feature_it_noun_marker_combinations.append(combination)
feature_it_noun_marker = [FeatureSet(features) for features in feature_it_noun_marker_combinations]
compound_it_noun_marker = CompoundSet(constituent_it_noun_marker, pos_it_noun_marker, feature_it_noun_marker)
noun_marker_pl_dict["it"] = compound_it_noun_marker

pos_të_noun_marker = ({"POS": "noun_marker_pl"})
constituent_të_noun_marker = ({"të": pos_të_noun_marker})
feature_të_noun_marker_combinations = []
number_dict = {"NUM": "pl"}
gender_dict = {"G": "f"}
case_dict = {"CASE": "CASE"}
nd_dict = {"ND": "def"}
combination = (number_dict, gender_dict, nd_dict, case_dict)
feature_të_noun_marker_combinations.append(combination)

gender_dict = {"G": "f"}
case_dict = {"CASE": "acc"}
nd_dict = {"ND": "def"}
combination = (number_dict, gender_dict, nd_dict, case_dict)
feature_të_noun_marker_combinations.append(combination)

gender_dict = {"G": "G"}
case_dict = {"CASE": "CASE"}
combination = (number_dict, gender_dict, nd_dict, case_dict)
feature_të_noun_marker_combinations.append(combination)

case_dict = {"CASE": "acc"}
combination = (number_dict, gender_dict, nd_dict, case_dict)
feature_të_noun_marker_combinations.append(combination)
feature_të_noun_marker = [FeatureSet(features) for features in feature_të_noun_marker_combinations]
compound_të_noun_marker = CompoundSet(constituent_të_noun_marker, pos_të_noun_marker, feature_të_noun_marker)
noun_marker_pl_dict["të"] = compound_të_noun_marker

pos_vet_noun_marker = ({"POS": "noun_marker_pl"})
constituent_vet_noun_marker = ({"vet": pos_vet_noun_marker})
feature_vet_noun_marker_combinations = []
number_dict = {"NUM": "pl"}
gender_dict = {"G": "f"}
case_dict = {"CASE": "dat"}
nd_dict = {"ND": "def"}
combination = (number_dict, gender_dict, nd_dict, case_dict)
feature_vet_noun_marker_combinations.append(combination)

gender_dict = {"G": "G"}
combination = (number_dict, gender_dict, nd_dict, case_dict)
feature_vet_noun_marker_combinations.append(combination)

feature_vet_noun_marker = [FeatureSet(features) for features in feature_vet_noun_marker_combinations]
compound_vet_noun_marker = CompoundSet(constituent_vet_noun_marker, pos_vet_noun_marker, feature_vet_noun_marker)
noun_marker_pl_dict["vet"] = compound_vet_noun_marker


def read_nouns_from_dictionary() -> tuple[dict[str, CompoundSet], dict[str, CompoundSet]]:
    """
    Reads the adjectives from the dictionary file and returns 2 dictionaries, 1 with adjectives that have articles and 1
    with adjectives that don't have articles.
    """
    lemma_noun_sgm_dict = {}
    lemma_noun_sgf_dict = {}
    lemma_noun_pl_dict = {}

    with open(NOUN_DICT_FILE, mode="r", encoding="utf-8") as dict_file:
        for line in dict_file:
            lemma_noun, lemma_noun_pl, gender = line.strip().split(", ")

            # find alternative lemma if there
            alternative_lemma_noun = None
            if len(lemma_noun) > 2:
                if lemma_noun[-1] == "ë":
                    alternative_lemma_noun = lemma_noun[:-1]
                elif lemma_noun[-2] == "ë":
                    alternative_lemma_noun = lemma_noun[:-2] + lemma_noun[-1]  

            if gender == "m":
                lemma_noun_plm = lemma_noun_pl
                lemma_noun_sgm = lemma_noun

                pos_noun_sgm = ({"POS": "noun_sgm"})
                constituent_noun_sgm = ({lemma_noun_sgm: pos_noun_sgm})
                
                feature_sgm_noun_marker_combinations = []
                number_dict = {"NUM": "NUM"}
                gender_dict = {"G": "G"}
                nd_dict = {"ND": "ok"}                
                case_dict = {"CASE": "ok"}

                combination = (number_dict, gender_dict, nd_dict, case_dict)
                feature_sgm_noun_marker_combinations.append(combination)

                feature_noun_sgm = [FeatureSet(features) for features in feature_sgm_noun_marker_combinations]
                compound_noun_sgm = CompoundSet(constituent_noun_sgm, pos_noun_sgm, feature_noun_sgm)

                lemma_noun_sgm_dict[lemma_noun_sgm] = compound_noun_sgm

                if alternative_lemma_noun:
                    lemma_noun_sgm = alternative_lemma_noun
                    lemma_noun_sgm_dict[lemma_noun_sgm] = compound_noun_sgm

                pos_noun_plm = ({"POS": "noun_pl"})
                constituent_noun_plm = ({lemma_noun_plm: pos_noun_plm})

                feature_noun_pl_marker_combinations = []
                number_dict = {"NUM": "pl"}
                gender_dict = {"G": "G"}
                nd_dict = {"ND": "ok"}                
                case_dict = {"CASE": "ok"}

                combination = (number_dict, gender_dict, nd_dict, case_dict)
                feature_noun_pl_marker_combinations.append(combination)

                feature_noun_pl = [FeatureSet(features) for features in feature_noun_pl_marker_combinations]
                compound_noun_pl = CompoundSet(constituent_noun_plm, pos_noun_plm, feature_noun_pl)

                lemma_noun_pl_dict[lemma_noun_plm] = compound_noun_pl

            else:
                lemma_noun_sgf = lemma_noun
                lemma_noun_plf = lemma_noun_pl
           
                pos_noun_sgf = ({"POS": "noun_sgf"})
                constituent_noun_sgf = ({lemma_noun_sgf: pos_noun_sgf})
                feature_noun_sgf_marker_combinations = []
                number_dict = {"NUM": "NUM"}
                gender_dict = {"G": "f"}
                nd_dict = {"ND": "ok"}                
                case_dict = {"CASE": "ok"}
                combination = (number_dict, gender_dict, nd_dict, case_dict)
                feature_noun_sgf_marker_combinations.append(combination)

                feature_noun_sgf = [FeatureSet(features) for features in feature_noun_sgf_marker_combinations]
                compound_noun_sgf = CompoundSet(constituent_noun_sgf, pos_noun_sgf, feature_noun_sgf)

                lemma_noun_sgf_dict[lemma_noun_sgf] = compound_noun_sgf

                if alternative_lemma_noun:
                    lemma_noun_sgf = alternative_lemma_noun
                    lemma_noun_sgf_dict[lemma_noun_sgf] = compound_noun_sgf                                    
                
                pos_noun_plf = ({"POS": "noun_pl"})
                constituent_noun_plf = ({lemma_noun_plf: pos_noun_plf})
                feature_noun_pl_marker_combinations = []
                number_dict = {"NUM": "pl"}
                gender_dict = {"G": "f"}
                nd_dict = {"ND": "ok"}                
                case_dict = {"CASE": "ok"}
                combination = (number_dict, gender_dict, nd_dict, case_dict)
                feature_noun_pl_marker_combinations.append(combination)

                feature_noun_pl = [FeatureSet(features) for features in feature_noun_pl_marker_combinations]
                compound_noun_pl = CompoundSet(constituent_noun_plf, pos_noun_plf, feature_noun_pl)

                lemma_noun_pl_dict[lemma_noun_plf] = compound_noun_pl

    return lemma_noun_sgm_dict, lemma_noun_sgf_dict, lemma_noun_pl_dict


def annotate_nouns(
    all_tokens: List[str],
    dictionary_sgm_nouns: Dict[str, Any],
    dictionary_sgm_noun_markers: Dict[str, Any],
    dictionary_sgf_nouns: Dict[str, Any],
    dictionary_sgf_noun_markers: Dict[str, Any],
    dictionary_pl_nouns: Dict[str, Any],
    dictionary_pl_noun_markers: Dict[str, Any],
) -> Dict[str, Any]:
    annotated_nouns = {}
    noun_categories = {
        'sgm': (dictionary_sgm_nouns, dictionary_sgm_noun_markers),
        'sgf': (dictionary_sgf_nouns, dictionary_sgf_noun_markers),
        'pl': (dictionary_pl_nouns, dictionary_pl_noun_markers),
    }

    for possible_noun_token in all_tokens:
        possible_noun_token = possible_noun_token.lower().strip()
        all_alternatives = []  # To store all valid configurations for each token

        # Process each noun category
        for category, (lemma_dict, marker_dict) in noun_categories.items():
            for marker in marker_dict:
                if possible_noun_token.endswith(marker):
                    lemma = possible_noun_token.removesuffix(marker)
                    if lemma in lemma_dict:
                        # Prepare the data for unification
                        head_entry = lemma
                        complement_entry = marker
                        head_dict = {lemma: lemma_dict[lemma]}
                        complement_dict = {marker: marker_dict[marker]}
                        annotated_noun = unify_phrase(head_entry, complement_entry, head_dict, complement_dict, valid_phrase_dict, possible_noun_token)
                        if annotated_noun:
                            all_alternatives.append(annotated_noun[possible_noun_token])

        # Only add to the main dictionary if there are valid alternatives
        if all_alternatives:
            annotated_nouns[possible_noun_token] = all_alternatives

    return annotated_nouns

def numeral_noun(start_clause, dictionary_numerals, dictionary_nouns, dictionary_noun_phrases, valid_phrase_dict):
    output_noun_phrases = {}
    output_numerals = {}
    output_nouns = {}
    output_clause = start_clause
    processed_unified_numerals = set()

    for entry in output_clause:
        if entry in dictionary_numerals:
            prev_entry, post_entry = entry_indexing(entry, output_clause)
            numeral_entries = dictionary_numerals[entry]

            # Ensure numeral_entries is a list even if it contains only one CompoundSet
            if not isinstance(numeral_entries, list):
                numeral_entries = [numeral_entries]

            if post_entry in dictionary_nouns:
                noun_entries = dictionary_nouns[post_entry]

                # Ensure noun_entries is a list even if it contains only one CompoundSet
                if not isinstance(noun_entries, list):
                    noun_entries = [noun_entries]

                unified_numeral = (entry, post_entry)

                if unified_numeral not in processed_unified_numerals:
                    processed_unified_numerals.add(unified_numeral)
                    output_clause = apply_phrase_sentence(entry, str(post_entry), output_clause)

                    for numeral_compound in numeral_entries:
                        for noun_compound in noun_entries:
                            output_np = unify_phrase(post_entry, entry, {post_entry: noun_compound}, {entry: numeral_compound}, valid_phrase_dict, unified_numeral)
                            if output_np:
                                output_noun_phrases[unified_numeral] = output_np
                                break  # Optionally break after the first successful combination
                        if unified_numeral in output_noun_phrases:
                            break  # Optionally break after the first successful combination

            else:
                output_numerals[entry] = numeral_entries  # Handle single or multiple numeral entries
        else:
            # Handle cases where the entry is not a numeral
            if entry in dictionary_noun_phrases:
                output_noun_phrases[entry] = dictionary_noun_phrases[entry]
            elif entry in dictionary_nouns:
                output_nouns[entry] = dictionary_nouns[entry]

    return output_clause, output_numerals, output_nouns, output_noun_phrases

def quantifier_noun(start_clause, dictionary_quantifiers, dictionary_nouns, dictionary_noun_phrases, valid_phrase_dict):
    output_noun_phrases = {}
    output_quantifiers = {}
    output_nouns = {}
    output_clause = start_clause
    processed_unified_quantifiers = set()

    for entry in output_clause:
        if entry in dictionary_quantifiers:
            prev_entry, post_entry = entry_indexing(entry, output_clause)
            quantifier_entries = dictionary_quantifiers[entry]

            # Ensure quantifier_entries is a list even if it contains only one CompoundSet
            if not isinstance(quantifier_entries, list):
                quantifier_entries = [quantifier_entries]

            if post_entry in dictionary_nouns:
                noun_entries = dictionary_nouns[post_entry]

                # Ensure noun_entries is a list even if it contains only one CompoundSet
                if not isinstance(noun_entries, list):
                    noun_entries = [noun_entries]

                unified_quantifier = (entry, post_entry)

                if unified_quantifier not in processed_unified_quantifiers:
                    processed_unified_quantifiers.add(unified_quantifier)
                    output_clause = apply_phrase_sentence(entry, str(post_entry), output_clause)

                    for quantifier_compound in quantifier_entries:
                        for noun_compound in noun_entries:
                            output_np = unify_phrase(post_entry, entry, {post_entry: noun_compound}, {entry: quantifier_compound}, valid_phrase_dict, unified_quantifier)
                            if output_np:
                                output_noun_phrases[unified_quantifier] = output_np
                                break  # Optionally break after the first successful combination
                        if unified_quantifier in output_noun_phrases:
                            break  # Optionally break after the first successful combination

            else:
                output_quantifiers[entry] = quantifier_entries  # Handle single or multiple quantifier entries
        else:
            # Handle cases where the entry is not a quantifier
            if entry in dictionary_noun_phrases:
                output_noun_phrases[entry] = dictionary_noun_phrases[entry]
            elif entry in dictionary_nouns:
                output_nouns[entry] = dictionary_nouns[entry]

    return output_clause, output_quantifiers, output_nouns, output_noun_phrases

def referent_noun(start_clause, dictionary_referents, dictionary_nominals, dictionary_nouns, dictionary_noun_phrases):
    # this function identifies referent, and, if post_entry in output_nouns, it creates a phrase "3 books"
    # it also output referents separate, and nouns separate if not in unified_np
    output_noun_phrases = {}
    output_referents = {}
    output_nouns = {}
    output_nominals = {}
    output_clause = start_clause
    processed_unified_referents = set()

    for entry in output_clause:
        if entry in dictionary_referents:
            prev_entry, post_entry = entry_indexing(entry, output_clause)
            referent_entries = dictionary_referents[entry]

            # Ensure referent_entries is a list even if it contains only one CompoundSet
            if not isinstance(referent_entries, list):
                referent_entries = [referent_entries]

            if post_entry in dictionary_nouns:
                noun_entries = dictionary_nouns[post_entry]

                # Ensure noun_entries is a list even if it contains only one CompoundSet
                if not isinstance(noun_entries, list):
                    noun_entries = [noun_entries]

                unified_referent = (entry, post_entry)

                if unified_referent not in processed_unified_referents:
                    processed_unified_referents.add(unified_referent)
                    output_clause = apply_phrase_sentence(entry, str(post_entry), output_clause)

                    for referent_compound in referent_entries:
                        for noun_compound in noun_entries:
                            output_np = unify_phrase(post_entry, entry, {post_entry: noun_compound}, {entry: referent_compound}, valid_phrase_dict, unified_referent)
                            if output_np:
                                output_noun_phrases[unified_referent] = output_np
                                break  # Optionally break after the first successful combination
                        if unified_referent in output_noun_phrases:
                            break  # Optionally break after the first successful combination

            else:
                output_referents[entry] = referent_entries  # Handle single or multiple referent entries
        else:
            # Handle cases where the entry is not a referent
            if entry in dictionary_noun_phrases:
                output_noun_phrases[entry] = dictionary_noun_phrases[entry]
            elif entry in dictionary_nouns:
                output_nouns[entry] = dictionary_nouns[entry]


            elif entry in set(dictionary_nominals.keys()):
                output_nominals[entry] = dictionary_nominals[entry]                    
                            
    return output_clause, output_referents, output_nominals, output_nouns, output_noun_phrases

def clean_token(token):
    # Remove any non-alphanumeric characters from the token
    return re.sub(r'\W+', '', token)

def preposition_noun(start_clause, dictionary_prepositions, dictionary_referents, dictionary_nominals, dictionary_nouns, dictionary_noun_phrases, valid_phrase_dict):
    output_noun_phrases = {}
    output_prepositions = {}
    output_nouns = {}
    output_prep_phrases = {}
    output_referents = {}
    output_nominals = {}
    output_clause = start_clause
    processed_unified_prepositions = set()

    for entry in output_clause:
        if entry in dictionary_prepositions:
            prev_entry, post_entry = entry_indexing(entry, output_clause)

            if post_entry in dictionary_nouns:
                process_preposition_noun_pair(entry, post_entry, dictionary_prepositions[entry], dictionary_nouns[post_entry], valid_phrase_dict, processed_unified_prepositions, output_prep_phrases, output_clause)

            elif post_entry and post_entry in dictionary_noun_phrases:
                process_preposition_noun_phrase_pair(entry, post_entry, dictionary_prepositions[entry], dictionary_noun_phrases[post_entry], valid_phrase_dict, processed_unified_prepositions, output_prep_phrases, output_clause)

            else:
                output_prepositions[entry] = dictionary_prepositions[entry]

        else:
            if entry in dictionary_noun_phrases:
                output_noun_phrases[entry] = dictionary_noun_phrases[entry]
            elif entry in dictionary_nouns:
                output_nouns[entry] = dictionary_nouns[entry]
            elif entry in dictionary_referents:
                output_referents[entry] = dictionary_referents[entry]
            elif entry in dictionary_nominals:
                output_nominals[entry] = dictionary_nominals[entry]

    return output_clause, output_prepositions, output_referents, output_nominals, output_nouns, output_noun_phrases, output_prep_phrases

def process_preposition_noun_pair(entry, post_entry, preposition_entries, noun_entries, valid_phrase_dict, processed_set, output_dict, output_clause):
    unified_preposition = (entry, post_entry)
    if unified_preposition not in processed_set:
        processed_set.add(unified_preposition)
        output_clause = apply_phrase_sentence(entry, str(post_entry), output_clause)
        
        # Ensure preposition_entries and noun_entries are lists
        if not isinstance(preposition_entries, list):
            preposition_entries = [preposition_entries]
        if not isinstance(noun_entries, list):
            noun_entries = [noun_entries]

        for preposition_compound in preposition_entries:
            for noun_compound in noun_entries:
                # Create dictionaries expected by unify_phrase
                preposition_dict = {entry: preposition_compound}
                noun_dict = {post_entry: noun_compound}
                output_pp = unify_phrase(entry, post_entry, preposition_dict, noun_dict, valid_phrase_dict, unified_preposition)
                if output_pp:
                    output_dict[unified_preposition] = output_pp
                    break  # Optionally break after the first successful combination
            if unified_preposition in output_dict:
                break

def process_preposition_noun_phrase_pair(entry, post_entry, preposition_entries, noun_phrase_entries, valid_phrase_dict, processed_set, output_dict, output_clause):
    unified_preposition = (entry, post_entry)
    if unified_preposition not in processed_set:
        processed_set.add(unified_preposition)
        output_clause = apply_phrase_sentence(entry, str(post_entry), output_clause)
        for preposition_compound in preposition_entries:
            for noun_phrase_compound in noun_phrase_entries:
                output_pp = unify_phrase(entry, post_entry, preposition_compound, noun_phrase_compound, valid_phrase_dict, unified_preposition)
                if output_pp:
                    output_dict[unified_preposition] = output_pp
                    break  # Optionally break after the first successful combination
            if unified_preposition in output_dict:
                break

def process_sentences(text_name):
    # This functions start "for sentence in sentences" and then "for clause in clauses"
    # so each annotation is done within a clause
    # the needed dictionaries are provided either at the beginning or drawn from functions:
    # before running the function "ref_np", here we already have: numeral_dict, referent_dict, quantifier_dict
    # then, "for clause in clauses", we start first by annotating nouns inside clause, and then the three others
    # final output: "annotated_noun_p". Or, if without noun: annotated_numerals, annotated_quantifiers, annotated_referents
    lemma_noun_sgm_dict, lemma_noun_sgf_dict, lemma_noun_pl_dict = read_nouns_from_dictionary()
    prep_dict = read_prepositions_from_dictionary()
    numeral_dict = read_numerals_from_dictionary()
    referent_dict = get_referents(ref_dem_list, ref_marker_list, adjust_merge_dict, ref_dem_dict, ref_marker_dict)
    
    annotated_sentences = {}

    collected_annotated_numerals = {}
    collected_annotated_quantifiers = {}
    collected_annotated_nominals = {}
    collected_annotated_referents = {}
    collected_annotated_prepositions = {}
    collected_annotated_noun_phrases = {}
    collected_annotated_prep_phrases = {}
    collected_annotated_nouns = {}
    phrased_clauses = get_clause_tokens(text_name)
    collected_final_clause = []
    processed_phrase = set()
    processed_clauses = []
    for phrased_clause in phrased_clauses:
        resulting_phrased_clause = []
        # Initialize dictionaries to collect results

        annotated_nouns = annotate_nouns(
            phrased_clause, lemma_noun_sgm_dict, noun_marker_sgm_dict, lemma_noun_sgf_dict, noun_marker_sgf_dict, lemma_noun_pl_dict, noun_marker_pl_dict
            )
        if annotated_nouns != {}:
            collected_annotated_nouns.update(annotated_nouns)

        # ref_np now acts as a generator, so we iterate over its yields
        updated_clause, annotated_numerals, annotated_nouns, annotated_noun_phrases = numeral_noun(
            phrased_clause, numeral_dict, collected_annotated_nouns, collected_annotated_noun_phrases, valid_phrase_dict
            )

        if annotated_noun_phrases != {}:
            collected_annotated_noun_phrases.update(annotated_noun_phrases)
            

        if annotated_nouns != {}:
            collected_annotated_nouns.update(annotated_nouns)

        if annotated_numerals != {}:
            collected_annotated_nominals.update(annotated_numerals)

        if updated_clause != []:
            processing_clause = updated_clause
        
        updated_clause, annotated_quantifiers, annotated_nouns, annotated_noun_phrases = quantifier_noun(
            processing_clause, quantifier_dict, collected_annotated_nouns, collected_annotated_noun_phrases, valid_phrase_dict
            )

        if annotated_noun_phrases != {}:
            collected_annotated_noun_phrases.update(annotated_noun_phrases)
            

        if annotated_quantifiers != {}:
            collected_annotated_nominals.update(annotated_quantifiers)

        if annotated_nouns != {}:
            collected_annotated_nouns.update(annotated_nouns)

        if updated_clause != []:
            processing_clause = updated_clause                

        updated_clause, annotated_referents, annotated_nominals, annotated_nouns, annotated_noun_phrases = referent_noun(
            processing_clause, referent_dict, collected_annotated_nominals, collected_annotated_nouns, collected_annotated_noun_phrases
        )
        if annotated_referents != {}:
            collected_annotated_referents.update(annotated_referents)

        if annotated_nominals != {}:
            collected_annotated_nominals.update(annotated_nominals)            
            

        if annotated_noun_phrases != {}:
            collected_annotated_noun_phrases.update(annotated_noun_phrases)
            

        if annotated_nouns != {}:
            collected_annotated_nouns.update(annotated_nouns)

        if updated_clause != []:
            processing_clause = updated_clause

        updated_clause, annotated_prepositions, annotated_nominals, annotated_referents, annotated_nouns, annotated_noun_phrases, annotated_prep_phrases = preposition_noun(
                processing_clause, prep_dict, collected_annotated_nominals, collected_annotated_referents, collected_annotated_nouns, collected_annotated_noun_phrases, valid_phrase_dict
            )

        if collected_annotated_prepositions != {}: 
            collected_annotated_prepositions.update(annotated_prepositions)

        if annotated_referents != {}:
            collected_annotated_referents.update(annotated_referents)
            

        if annotated_nominals != {}:
            collected_annotated_nominals.update(annotated_nominals)
        

        if annotated_nouns != {}:
            collected_annotated_nouns.update(annotated_nouns)
    
        if annotated_prep_phrases != {}:
            collected_annotated_prep_phrases.update(annotated_prep_phrases)

        if collected_annotated_noun_phrases != {}: 
            collected_annotated_noun_phrases.update(annotated_noun_phrases)
            
        if updated_clause != []:
            collected_final_clause.append(updated_clause)

        # Remove duplicates from collected_final_clause
        collected_final_clause = [list(item) for item in set(tuple(row) for row in collected_final_clause)]
        resulting_phrased_clause.append(collected_final_clause)                            



    # After collecting,  aggregate or further process the results
    
    annotated_sentences.update(collected_annotated_nominals)
    annotated_sentences.update(collected_annotated_referents)
    annotated_sentences.update(collected_annotated_prep_phrases)
    annotated_sentences.update(collected_annotated_prepositions)
    annotated_sentences.update(collected_annotated_noun_phrases)
    annotated_sentences.update(collected_annotated_nouns)
    resulting_phrased_clause.append(collected_final_clause)

    return annotated_sentences, resulting_phrased_clause
    
if __name__ == "__main__":
    for text_name in TEXT_NAMES:
        with open(text_name, "r", encoding = "utf-8") as file:

            annotated_sentences, resulting_phrased_clause = process_sentences(file)
            output_annotated_sentences = "annotated_sentences.txt"
            with open(output_annotated_sentences, "w", encoding = "utf-8") as file:
                for sentence, annotations in annotated_sentences.items():
                    file.write(f"{sentence}: {annotations}\n") 

            output_sentences = "sentences.txt"
            with open(output_sentences, "w", encoding = "utf-8") as file:
                for sentence  in resulting_phrased_clause:
                    file.write(f"{sentence}\n") 


 
    
