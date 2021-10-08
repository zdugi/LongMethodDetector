from keras.models import load_model
from gensim.parsing.preprocessing import preprocess_documents
import pandas as pd
import numpy as np
from functools import reduce
import pickle

model = load_model('model-31.h5')

# preprocessing
def remove_leading(snippet):
    ret = ''
    for line in snippet.split('\n'):
        ret += line.lstrip() + '\n'
    return ret

def force_ascii(string):
    encoded_string = string.encode('ascii', 'ignore')
    return encoded_string.decode()

def extract_clean_body(raw_snippet):
    encoded_string = raw_snippet.encode('ascii', 'ignore')
    cleaned = remove_leading(encoded_string.decode())
    
    # remove double newlines, critical!!
    len_before = len(cleaned)
    tmp = cleaned.replace('\n\n','\n')
    len_after = len(tmp)
    while len_after < len_before:
        len_before = len(tmp)
        tmp = tmp.replace('\n\n', '\n')
        len_after = len(tmp)
        
    cleaned = tmp
    
    # select only body
    start = cleaned.find('{')
    end = cleaned.rfind('}')
    
    return cleaned[start + 1:end].strip()

def predict(code_snippet, threshold = 0.5):
    assert 0. <= threshold <= 1.
    
    if type(code_snippet) == float:
        print(code_snippet)
    
    pattern = [ord(char) for char in code_snippet]
    empty = 100 - len(pattern) % 100
    
    for i in range(empty):
        pattern.append(ord(' '))
    
    pattern = np.array(pattern)
    
    #TODO: pass model
    probs = model.predict(pattern.reshape(-1, 100)).flatten()
    preds = (model.predict(pattern.reshape(-1, 100)) > threshold)
    totalLogBlocks = preds.sum() + 1
    
    newline_vec = []
    
    for i in range(len(pattern)):
        if pattern[i] == ord('\n'):
            newline_vec.append(probs[i])
    
    return totalLogBlocks, preds, pattern, newline_vec

def extract_logical_blocks(snippet, prediction_threshold = .4):
    num, preds, pattern, _ = predict(snippet, prediction_threshold)
    
    preds_flatten = preds.flatten()
    indices = [0] + list(np.argwhere(preds_flatten == True).flatten())
    blocks = [pattern[i:j] for i,j in zip(indices, indices[1:]+[None])]
    
    decoded_blocks = []
    
    for coded_block in blocks:
        decoded_blocks.append(''.join([chr(ci) for ci in coded_block]))
    
    return decoded_blocks

# LSI
stop_words = 'abstract continue for new switch assert default goto package synchronized boolean do if private this break double implements protected throw byte else import public throws case enum instanceof return transient catch extends int short try char final interface static void class finally long strictfp volatile const float native super while'
stop_words_list = stop_words.split()

def encode(doc, dictionary):
    result = []
    #doc = doc.lower().split(' ')
    doc = preprocess_documents([doc])[0]
    for w in dictionary:
        result.append(doc.count(w))
    return np.asarray(result)


if __name__ == '__main__':
    with open('snippet.in.txt', 'rb') as input_snippet_file:
        snippet = input_snippet_file.read().decode("utf-8") 
        cleaned_snippet = extract_clean_body(snippet)
        new_line_probs = predict(cleaned_snippet, .4)[3]

        assert len(new_line_probs) <= 404
        x_padded = new_line_probs + [0] * (404 - len(new_line_probs))

        with open('rf_model.pickle', 'rb') as rf:
            clf = pickle.load(rf)
            print(clf.predict(np.array(x_padded).reshape(1, -1)))

        #print(detect_long_method(logical_blocks))