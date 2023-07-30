from sklearn.metrics.pairwise import cosine_similarity
from annoy import AnnoyIndex

def find_near_req_annoy(vectorized_matrix, clean_request, num_trees=10, metric='angular', topn=3, search_k=-1, include_distances=False):
    vectorized_matrix_size = vectorized_matrix.shape[1]

    ai = AnnoyIndex(vectorized_matrix_size, metric=metric)
    for ind, item in enumerate(vectorized_matrix):
        ai.add_item(ind, item)

    num_trees = num_trees
    ai.build(num_trees)
    ai.save('books.ann')

    ai_load = AnnoyIndex(vectorized_matrix_size, metric=metric)
    ai_load.load('books.ann')

    # r = vectorized_matrix.transform([clean_request]).toarray().flatten()
    r = clean_request
    nearest_topn = ai_load.get_nns_by_vector(r, topn, search_k=search_k, include_distances=include_distances)
    return nearest_topn

def find_nearest_req(dict_vec, request_vec,topn=5):
    dict_dist_request = {}
    for name, vector in dict_vec.items():
        cos_sim = cosine_similarity(request_vec, [vector])
        dict_dist_request[name] = round(float(cos_sim), 4)

    sorted_dict_dist_request = dict(sorted(dict_dist_request.items(), key=lambda x: x[1], reverse=True)[:topn])
    return sorted_dict_dist_request