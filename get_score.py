#Creating Matcher Object
import spacy
nlp=spacy.load('en_core_web_sm')
from spacy.matcher import Matcher
m_tool=Matcher(nlp.vocab)
#Defining Patterns
