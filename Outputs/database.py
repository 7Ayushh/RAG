import pymongo
import pickle

if __name__=='__main__':
    client=pymongo.MongoClient('mongodb://localhost:27017/')
    print(client)

    with open('doc.pkl', 'rb') as f:
        doc=pickle.load(f)
    
    db=client['ChatAI']
    collection=db['Collection']

    for heading in doc:
        collection.insert_one(heading)