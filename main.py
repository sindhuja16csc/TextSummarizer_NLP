import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from heapq import nlargest
#from  module name import funtionName 
# Load the English language model
nlp = spacy.load("en_core_web_sm")
# Sample document for summarization
document = """Elon Reeve Musk was born on June 28, 1971, in Pretoria, South Africa's administrative capital. He is of British and Pennsylvania Dutch ancestry.[4][5] His mother, Maye (née Haldeman), is a model and dietitian born in Saskatchewan, Canada, and raised in South Africa. His father, Errol Musk, is a South African electromechanical engineer, pilot, sailor, consultant, emerald dealer, and property developer, who partly owned a rental lodge at Timbavati Private Nature Reserve. Elon has a younger brother, Kimbal, a younger sister, Tosca, as well as four paternal half-siblings. Musk was raised in the Anglican Church, in which he was baptized.
The Musk family was wealthy during Elon's youth. Despite both Elon and Errol previously stating that Errol was a part owner of a Zambian emerald mine, in 2023, Errol recounted that the deal he made was to receive "a portion of the emeralds produced at three small mines".  Errol was elected to the Pretoria City Council as a representative of the anti-apartheid Progressive Party and has said that his children shared their father's dislike of apartheid. After his parents divorced in 1980, Elon chose to live primarily with his father. Elon later regretted his decision and became estranged from his father. Elon has recounted trips to a wilderness school that he described as a "paramilitary Lord of the Flies" where "bullying was a virtue" and children were encouraged to fight over rations. In one incident, after an altercation with a fellow pupil, Elon was thrown down concrete steps and beaten severely, leading to him being hospitalized for his injuries. Elon described his father berating him after he was discharged from the hospital. Errol denied berating Elon and claimed, "The boy had just lost his father to suicide and Elon had called him stupid. Elon had a tendency to call people stupid. How could I possibly blame that child?"
Elon was an enthusiastic reader of books, and had attributed his success in part to having read The Lord of the Rings, the Foundation series, and The Hitchhiker's Guide to the Galaxy.  At age ten, he developed an interest in computing and video games, teaching himself how to program from the VIC-20 user manual. At age twelve, Elon sold his BASIC-based game Blastar to PC and Office Technology magazine for approximately $500."""
# Process the document
doc = nlp(document)
# Extract stopwords from spaCy
stopwords = list(STOP_WORDS)
# Compute word frequencies (excluding stopwords and punctuation)
word_frequencies = {}
for word in doc:
    if word.text.lower() not in stopwords and word.is_alpha:
        if word.text.lower() not in word_frequencies:
            word_frequencies[word.text.lower()] = 1
        else:
            word_frequencies[word.text.lower()] += 1

# Normalize word frequencies
maximum_frequency = max(word_frequencies.values())
for word in word_frequencies:
    word_frequencies[word] /= maximum_frequency
# Score sentences
sentence_list = [sentence for sentence in doc.sents] # list comprehernsive 
sentence_score = {}
for sent in sentence_list:
    for word in sent:
        if word.text.lower() in word_frequencies:
            if sent not in sentence_score:
                sentence_score[sent] = word_frequencies[word.text.lower()] 
            else:
                sentence_score[sent] += word_frequencies[word.text.lower()]

# Select the top 3 sentences
summarized_sentences = nlargest(3, sentence_score, key=sentence_score.get)
# Join the sentences to form the summary
final_sentences = [sent.text.replace("\n","") for sent in summarized_sentences] 
summary = "\n ".join(final_sentences)
# Print the summary
print(summary)

