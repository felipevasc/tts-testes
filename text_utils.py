from num2words import num2words
import re

def fix_periods(str):
    if str == "":                    # Don't change empty strings.
        return str
    if str[-1] in ["?", ".", "!"]:   # Don't change if already okay.
        return str
    if str[-1] == ",":               # Change trailing ',' to '.'.
        return str[:-1] + "."
    return str + "."                 # Otherwise, add '.'.

def replace_words(text,dictionary):
    for key in dictionary.keys():
        text = text.replace(key, dictionary[key])
    return text

conversion_dict = {
    '%' : ' porcento',
    'PSOL': 'pêssól',
    'OAB': 'Oabê',
    'SP': 'São Paulo',
    'RJ': 'Rio de Janeiro',
    '°C': ' graus célsius',
    'ºC': ' graus célsius',
    'voz': 'vóz',
    'Voz': 'vóz'
    
}
def convert_helper(obj):
    num_in_text = int(obj.group(0))
    return num2words(num_in_text,lang='pt_BR').replace(",","")

def convert_numbers(text):
    return re.sub(r'\d+',convert_helper,text)

acronym_translation = {
    'A': 'a',
    'B': 'bê',
    'C': 'cê',
    'D': 'dê',
    'E': 'é',
    'F': 'éfe',
    'G': 'gê',
    'H': 'agá',
    'I': 'I',
    'J': 'jóta',
    'K': 'cá',
    'L': 'éle',
    'M': 'ême',
    'N': 'êne',
    'O': 'ó',
    'P': 'pê',
    'Q': 'quê',
    'R': 'érre',
    'S': 'ésse',
    'T': 'tê',
    'U': 'u',
    'V': 'vê',
    'W': 'dábliu',
    'X': 'xis',
    'Y': 'ípsilon',
    'Z': 'zê'
}
def convert_helper_acronym(obj):
    acronym = obj.group(0)
    final_text = ''
    for letter in acronym:
        final_text += acronym_translation[letter] + ' '
    return final_text[:-1]

def convert_acronyms(text):
    return re.sub(r"\b[A-Z]{2,}\b",convert_helper_acronym,text)
def normalize_text(input_text):
    output_text = replace_words(input_text,conversion_dict)
    output_text = convert_numbers(output_text)
    output_text = convert_acronyms(output_text)
    output_text = fix_periods(output_text)
    return output_text

