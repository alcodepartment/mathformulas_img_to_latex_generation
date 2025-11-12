# core/tokenizer.py
#Никита, я тут пока заглукшу накидал везде

SPECIAL = ["<bos>","<eos>","<pad>","<unk>"]
BASIC = ["\\frac","\\sqrt","\\left","\\right","\\sum","\\int","\\log","\\sin","\\cos",
         "{","}","(",")","[","]","^","_","+","-","\\times","\\cdot","=",",",".","\\pi","\\infty","&","\\\\"]
ALNUM = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

VOCAB = SPECIAL + BASIC + ALNUM
itos = {i:t for i,t in enumerate(VOCAB)}
stoi = {t:i for i,t in itos.items()}

def tokenize(s: str):
    out=[]; i=0
    while i<len(s):
        if s[i]=="\\":
            j=i+1
            while j<len(s) and s[j].isalpha(): j+=1
            out.append(s[i:j]); i=j
        elif s[i].isspace(): i+=1
        else: out.append(s[i]); i+=1
    return out

def encode(tokens, max_len: int):
    ids=[stoi["<bos>"]]
    for t in tokens:
        ids.append(stoi.get(t, stoi["<unk>"]))
        if len(ids)>=max_len-1: break
    ids.append(stoi["<eos>"])
    ids += [stoi["<pad>"]]*(max_len-len(ids))
    return ids

def decode(ids):
    return [itos[i] for i in ids if itos[i] not in ("<bos>","<eos>","<pad>")]