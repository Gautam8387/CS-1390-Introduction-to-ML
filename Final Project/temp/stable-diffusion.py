import requests

API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content

# image_bytes = query({
# 	"inputs": "Astronaut riding a horse",
# })

Nouns =  ['Paris', 'capital', 'France', 'city']
Adjectives =  ['beautiful']
Verbs = []

# prompt = "Astronaut riding a horse"
for i in range(max(len(Nouns), len(Adjectives), len(Verbs))):
	# make a prompt using the nouns, adjectives, and verbs
	prompt = ""
	if i < len(Nouns):
		prompt += Nouns[i]
	if i < len(Adjectives):
		prompt += " " + Adjectives[i]
	if i < len(Verbs):
		prompt += " " + Verbs[i]
	# Query the model
	image_bytes = query({"inputs": prompt})
	# Save the image
	with open(f"image_{i}.png", "wb") as f:
		f.write(image_bytes)