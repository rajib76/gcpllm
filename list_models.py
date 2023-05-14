import cohere

# Get the API key
api_key = "fcFL7uISoIGfbekb9ajJRP30A1nAVdEdbYIZRtxO"

# Create a Cohere client
client = cohere.Client(api_key)

print(client.__getattribute__() )
# # List the available models
# models = client..list_models()
#
# # Print the models
# for model in models:
#     print(model)