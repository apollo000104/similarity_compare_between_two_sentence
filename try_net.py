import spacy
import numpy as np
nlp = spacy.load('en_core_web_lg')

final_labels=['LISTED ON PD or NOT', 'DEVELOPER NAME', 'PROJECT NAME', 'LISTING ID', 'BEDROOMS', 'Min Price AED', 'Max Price AED',
'Min Size SQF', 'Max Size SQF', 'DLD %', 'Downpayment %', 'Durring Construction %', 'Handover %', 'Handover Date', 'Post Handover %', 'Post Handover Months Number', 'Status']

old_labels=['Unit', 'Type', 'Gross', 'SIZE', 'Price', 'Status']

from scipy import spatial
cosine_similarity = lambda x, y: 1 - spatial.distance.cosine(x, y)

# # Choose the words you wish to compare, and obtain their vectors
# for old in old_labels:
#     similar_labels=[]
#     for final in final_labels:
#         doc1 = nlp(u''+final)
#         doc2 = nlp(u''+old)
#         simty = doc2.similarity(doc1)
#         if simty > 0.4:
#             similar_labels.append(final)
#         else: pass
#     print(old+" :", similar_labels)
    
for words1 in old_labels:
    words1=words1.split(' ')
    # print(words.split(' '))
    vector1=np.zeros(300)
    fixed_thread_value=0.4 # the base level of similarity
    thread_value=0.4
    for word1 in words1:
        vector1=vector1+nlp.vocab[word1].vector
        # print(len(nlp.vocab[word].vector))
    vector1=vector1/len(words1)
    similar_labels=[]
    max_similar_label=[]
    #til here, we get the average vector of short sentence in final labels
    
    #Now, I will get new vectors of old_labels and compare it with "vector"
    for words2 in final_labels:
        label=words2
        words2=words2.split(' ')
        vector2=np.zeros(300)
        for word2 in words2:
            vector2=vector2+nlp.vocab[word2].vector
        vector2=vector2/len(words2)
        similarity=cosine_similarity(vector2, vector1)
        # computed_similarities.append((words2, similarity))
        if similarity > fixed_thread_value:
            similar_labels.append(label)
        if similarity > thread_value:
            thread_value=similarity
            max_similar_label=label
            
    print(word1+':  ', similar_labels, " ", max_similar_label)
