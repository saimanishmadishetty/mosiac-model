from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
class_labels={"0": "alt.atheism", "1": "comp.graphics", "2": "comp.os.ms-windows.misc", "3": "comp.sys.ibm.pc.hardware", "4": "comp.sys.mac.hardware", "5": "comp.windows.x", "6": "misc.forsale", "7": "rec.autos", "8": "rec.motorcycles", "9": "rec.sport.baseball", "10": "rec.sport.hockey", "11": "sci.crypt", "12": "sci.electronics", "13": "sci.med", "14": "sci.space", "15": "soc.religion.christian", "16": "talk.politics.guns", "17": "talk.politics.mideast", "18": "talk.politics.misc", "19": "talk.religion.misc"}
def pre_transform(input_text):
    newsgroups = fetch_20newsgroups(subset='all')
    df = pd.DataFrame({'text': newsgroups.data, 'label': newsgroups.target})
    vectorizer = TfidfVectorizer(max_features=5000)
    vectorizer.fit(df['text'])
    print(input_text)
    print(type(input_text))
    transformed_input = vectorizer.transform([input_text])
    print(transformed_input)
    array_input = transformed_input.toarray().tolist()  # Convert to list of lists
    print("Transformed Input (pre_transform):", array_input)  # Debugging: Print the transformed input
    return array_input[0]
def post_transform(input):
    print("In Custom post_transform method")
    class_name = class_labels[str(input)]
    print("Custom post transformation done")
    print(class_name)
    return class_name
