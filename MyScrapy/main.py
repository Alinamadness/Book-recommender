# Import libraries
import json
import codecs
from cleaning import clean_text_parse, clean_text_request
from vectorizing import tfidf_vectorizing, count_vectorizing
from distance_near import find_near_req_annoy


# Open file with parsed data
with codecs.open('book.json', 'r', "utf-8") as json_file:
    books_json = json.load(json_file)
    json_file.close()
books = {dict_book['title']: dict_book['description'] for dict_book in books_json}
books_list = [(dict_book['title'], dict_book['description']) for dict_book in books_json]


# Cleaning all texts in books into list clean_books
clean_books = []
for sentence in books.values():
    clean_sentence = clean_text_parse(sentence)
    clean_books.append(clean_sentence)


# Create tfidf, count vectorization of each text in books
tfidf, tfidf_matrix = tfidf_vectorizing(clean_books)[0], tfidf_vectorizing(clean_books)[1]
count, count_matrix = count_vectorizing(clean_books)[0], count_vectorizing(clean_books)[1]


# Creating, cleaning and vectorizing the text of the request
request_text = "Detective story about pretty woman"
clean_request = clean_text_request(request_text)
clean_request_vec_tfidf = tfidf.transform([clean_request]).toarray().flatten()
clean_request_vec_count = count.transform([clean_request]).toarray().flatten()

# Finding the nearest topn books to vectorized request
topn = 3

# Print the result
print(f'Text of request: {request_text} \n')

print(f'Tf-idf vectorization top {topn} nearest books: \n')
for index in find_near_req_annoy(tfidf_matrix, clean_request_vec_tfidf, topn=topn, metric='dot'):
    print(f'Title: {books_list[index][0]}')
    # print(books_list[index][1])

print('======================')

print(f'Count vectorization top {topn} nearest books: \n')
for index in find_near_req_annoy(count_matrix, clean_request_vec_count, topn=topn, metric='dot'):
    print(f'Title: {books_list[index][0]}')
    # print(books_list[index][1])


# tfidf показал 1 довольно подходящий запрос из трех предложенных и справился лучше, чем count.
# Думаю, стоит применить еще более усложненные методы векторизации. А также попробовать
# манхэттонское расстояние. Евклидово расстояние не подходит для данной задачи.