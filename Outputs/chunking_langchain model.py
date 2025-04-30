import json
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from datetime import date
import nltk

#nltk.download('stopwords') # Run for the first time to download stopwords that will reduce the importance of regukarly used words in english such as and, the, or, etc. in the TF-IDF keyword extractor

def extract_keywords_tfidf(text, num_keywords=8):
    stop_words = stopwords.words('english')
    
    vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range=(1,2))
    
    tfidf_matrix = vectorizer.fit_transform([text])
    
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]
    
    top_indices = scores.argsort()[-num_keywords:][::-1]
    top_keywords = [feature_names[index] for index in top_indices]
    
    return top_keywords


def chunk_data(data):
    chunks = []
    Date = date.today().isoformat()
    
    for heading, sections in data.items():
        if isinstance(sections, dict):
            for section, sub_sections in sections.items():
                if isinstance(sub_sections, dict):
                    for sub_section, content in sub_sections.items():
                        if isinstance(content, list):
                            content_text = f"In the context of {heading}, the section on {section} related to {sub_section} states that {' '.join(content)}."
                        else:
                            content_text = f"In the context of {heading}, the section on {section} related to {sub_section} states that {content}."
                        
                        keywords = extract_keywords_tfidf(content_text)
                        
                        chunk = {
                            "content": content_text,
                            "metadata": {
                                "description": f"{heading} > {section} > {sub_section}",
                                "keywords": keywords,
                                "date" : Date
                            }
                        }
                        chunks.append(chunk)
                else:
                    content = sub_sections if isinstance(sub_sections, str) else ' '.join(sub_sections)
                    content_text = f"In the context of {heading}, the section on {section} states that {content}."
                    keywords = extract_keywords_tfidf(content_text)
                    
                    chunk = {
                        "content": content_text,
                        "metadata": {
                            "description": f"{heading} > {section}",
                            "keywords": keywords,
                            "date" : Date
                        }
                    }
                    chunks.append(chunk)
        else:
            content = sections if isinstance(sections, str) else ' '.join(sections)
            content_text = f"The section on {heading} states that {content}."
            keywords = extract_keywords_tfidf(content_text)
            
            chunk = {
                "content": content_text,
                "metadata": {
                    "description": f"{heading}",
                    "keywords": keywords,
                    "date" : Date
                }
            }
            chunks.append(chunk)
    
    return chunks

with open(r'C:\Users\divya\OneDrive\Documents\Vector Database DAV project\Student_Medical_Rule.json', 'r') as file:
    data = json.load(file)

chunks = chunk_data(data)

file_path = 'chunks.json'

with open(file_path, 'w') as f:
    json.dump(chunks, f, indent=4) 