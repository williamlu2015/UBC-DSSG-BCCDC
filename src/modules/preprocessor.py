import json
import re


def remove_symbols(result_full_description):
    return re.sub(r"[^a-zA-Z0-9 |]", "", result_full_description)


def replace_numbers(result_full_description):
    raw_tokens = result_full_description.split()
    tokens = [
        "_NUMBER_" if all(char.isdigit() for char in token) else token
        for token in raw_tokens
    ]
    return " ".join(tokens)


def replace_organisms(result_full_description, candidates_str):
    result = result_full_description.lower()

    candidates_dict = json.loads(candidates_str)
    for _, value in candidates_dict.items():
        for text in value["matched"]:
            result = result.replace(text.lower(), "_ORGANISM_")

    return result


def replace_hepatitis(result_full_description):
    raw_tokens = result_full_description.lower().split()
    tokens = [
        "_ORGANISM_" if token == "hbv" or token == "hbsag" else token
        for token in raw_tokens
    ]
    return " ".join(tokens)
