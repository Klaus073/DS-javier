from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')

# Create a new database
mydb = client['mydatabase']

# Create a new collection in the database
mycollection = mydb['mycollection']

# Insert some data into the collection
data = {'name': 'John', 'age': 30, 'city': 'New York'}
insert_result = mycollection.insert_one(data)
print("Inserted data with ID:", insert_result.inserted_id)

# Fetch and print all documents in the collection
for document in mycollection.find():
    print(document)
