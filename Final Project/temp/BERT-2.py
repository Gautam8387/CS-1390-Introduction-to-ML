# import transformers
# tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
# sentence = "The quick brown fox jumped over the lazy dog"
# tokens = tokenizer.tokenize(sentence)
# print(tokens)
# import torch
# model = transformers.BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=2)
# input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
# outputs = model(input_ids)
# predicted_labels = torch.argmax(outputs[0], axis=2)[0].tolist()
# print(predicted_labels)
# nouns = []
# adjectives = []
# for i, token in enumerate(tokens):
#     if predicted_labels[i] == 0: # noun
#         nouns.append(token)
#     elif predicted_labels[i] == 1: # adjective
#         adjectives.append(token)
        
# print("Nouns: ", nouns)
# print("Adjectives: ", adjectives)

# ###### 2 
# import spacy
# nlp = spacy.load("en_core_web_sm")

# sentence = "Paris is the capital of France and it is a beautiful city."

# doc = nlp(sentence)

# nouns = [token.text for token in doc if token.pos_ == "NOUN"]
# adjectives = [token.text for token in doc if token.pos_ == "ADJ"]

# print("Nouns: ", nouns)
# print("Adjectives: ", adjectives)
import spacy
nlp = spacy.load("en_core_web_sm")

# sentence = "The quick brown fox jumped over the lazy dog"#
# sentence = "Paris is the capital of France and it is a beautiful city."
sentence = 'separate criminal investigations they are trying to figure out if trump broke the law and if he did what to do about itto understand these four different cases you need to know a little bit about how criminal investigations in the us work in the first phase investigators gather evidence they might interview witnesses review survailance footage come over financial records or review texts and emails they show that evidence to a randomly selected group of citizens'

doc = nlp(sentence)

nouns = []
adjectives = []
verbs = []

for token in doc:
    if token.ent_type_ == "GPE" or token.ent_type_ == "LOC" or token.pos_ == "NOUN":#if token.dep_ == "nsubj"or token.dep_ == "dobj" or token.pos_ == "NOUN": #  or token.dep_ == "pobj" 
        nouns.append(token.text)
    elif token.pos_ == "ADJ":
        adjectives.append(token.text)
    elif token.pos_ == "VERB":
        verbs.append(token.text)

print("Nouns: ", nouns)
print("Adjectives: ", adjectives)
print("Verbs: ", verbs)
