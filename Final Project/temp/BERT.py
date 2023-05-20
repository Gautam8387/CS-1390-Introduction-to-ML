from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
from transformers import pipeline
import torch

# Load pre-trained BERT model for NER
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
# for sentiment analysis we use BertForSequenceClassification
model_name_2 = 'nlptown/bert-base-multilingual-uncased-sentiment'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
model_2 = AutoModelForSequenceClassification.from_pretrained(model_name_2)

# Create NER pipeline. nER is a task of Named Entity Recognition which performs entity extraction. 
# For example, it can extract the names of people, organizations, locations, etc. from a text.
# To extract adjectives, we can use the task of "sentiment-analysis" instead of "ner"
nlp = pipeline("ner", model=model, tokenizer=tokenizer)#, grouped_entities=True)
nlp_2 = pipeline("sentiment-analysis", model=model_2, tokenizer=tokenizer)

# Example text
text = "Paris is the capital of France and it is a beautiful city."
# text = 'separate criminal investigations they are trying to figure out if trump broke the law and if he did what to do about itto understand these four different cases you need to know a little bit about how criminal investigations in the us work in the first phase investigators gather evidence they might interview witnesses review survailance footage come over financial records or review texts and emails they show that evidence to a randomly selected group of citizens'

# Process the text and extract entities
entities = nlp(text)
entities_2 = nlp_2(text)

print(entities)
print(entities_2)
# Print entities
for entity in entities:
    print(f"{entity['word']}")# ({entity['entity']})")

# Print adjectives
# for entity in entities_2:
#     print(f"{entity['word']} ({entity['score']})")
