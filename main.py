from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
print("Welcome to The Cine GPT")
document='''1. Industry Name & Hub: Known globally as Kollywood, headquartered in the neighborhood of Kodambakkam, Chennai, Tamil Nadu, India.

2. Origin Milestone: The industry's history dates back to 1917 with Keechaka Vadham, the first silent feature film produced in South India by R. Nataraja Mudaliar.

3. First Talkie: The era of sound began in 1931 with the release of Kalidas, a multilingual film directed by H. M. Reddy featuring Tamil dialogue and songs.

4. Market Scale: It stands as one of the largest film industries in India, producing an average of 200 to 250 feature films annually.

5. Economic Impact: The industry contributes significantly to the local economy, with annual box office revenues regularly surpassing ₹1,000 to ₹1,500 crore (INR).

6. Global Distribution: Tamil films enjoy a massive international diaspora market, holding dominant box office territories in Malaysia, Sri Lanka, Singapore, the UK, and North America.

7. Top Grossing Film: 2.0 (2018), starring Rajinikanth and Akshay Kumar, holds the record as the highest-grossing Tamil film of all time, earning over ₹700 crore globally.

8. Technological Pioneer: Kollywood has historically pioneered Indian cinema tech, being home to Asia's first wide-screen formats, advanced VFX houses, and Dolby Atmos adoption.

9. Musical Legacy: The industry is globally recognized for its musical influence, anchored by legendary composers like Ilaiyaraaja and Academy Award-winner A. R. Rahman.

10. Cultural & Political Influence: Tamil cinema shares a unique, deep-rooted connection with regional politics; five former Chief Ministers of Tamil Nadu were high-profile film writers or actors.'''


text='''1. Industry Name & Hub: Known globally as Kollywood, headquartered in the neighborhood of Kodambakkam, Chennai, Tamil Nadu, India.

2. Origin Milestone: The industry's history dates back to 1917 with Keechaka Vadham, the first silent feature film produced in South India by R. Nataraja Mudaliar.

3. First Talkie: The era of sound began in 1931 with the release of Kalidas, a multilingual film directed by H. M. Reddy featuring Tamil dialogue and songs.

4. Market Scale: It stands as one of the largest film industries in India, producing an average of 200 to 250 feature films annually.

5. Economic Impact: The industry contributes significantly to the local economy, with annual box office revenues regularly surpassing ₹1,000 to ₹1,500 crore (INR).

6. Global Distribution: Tamil films enjoy a massive international diaspora market, holding dominant box office territories in Malaysia, Sri Lanka, Singapore, the UK, and North America.

7. Top Grossing Film: 2.0 (2018), starring Rajinikanth and Akshay Kumar, holds the record as the highest-grossing Tamil film of all time, earning over ₹700 crore globally.

8. Technological Pioneer: Kollywood has historically pioneered Indian cinema tech, being home to Asia's first wide-screen formats, advanced VFX houses, and Dolby Atmos adoption.

9. Musical Legacy: The industry is globally recognized for its musical influence, anchored by legendary composers like Ilaiyaraaja and Academy Award-winner A. R. Rahman.

10. Cultural & Political Influence: Tamil cinema shares a unique, deep-rooted connection with regional politics; five former Chief Ministers of Tamil Nadu were high-profile film writers or actors.
'''

chunks=text.split('.')
# splits according to the fullstop kept
print(chunks)

chunks=[chunk.strip() for chunk in chunks if chunk.strip()]
print(chunks)


model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embedding=model.encode(chunks)
print(embedding.shape) #dimesions is called shape

query=input('Enter your query: ')
print(query)

query_embedding=model.encode(query)
print(query_embedding.shape)
print(query_embedding)

similarities = cosine_similarity(
  [query_embedding],
  embedding
)
print(similarities)

best_match_index=similarities.argmax()
print(chunks[best_match_index])

retrieved_chunk=chunks[best_match_index]
prompt=f'''
Context:
{retrieved_chunk}
Question:
{query}
'''
print(prompt)

answer=f'''
{retrieved_chunk}
'''
print(answer) 


  