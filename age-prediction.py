import csv
import os
import timeit
import numpy as np
import statistics
from scipy.sparse import coo_matrix, csr_matrix

# Configuration
data_path = "data/"
graph_path = os.path.join(data_path, "graph")
demography_path = os.path.join(data_path, "trainDemography")
test_users_path = os.path.join(data_path, "users")
results_path = "prediction/"
birth_dates_path = os.path.join(results_path, "birth_dates")
test_graph_path = os.path.join(results_path, "test_graph")

global_start = timeit.default_timer()


def print_resource_usage(start_time):
    # print(f"Time elapsed: {timeit.default_timer() - start_time} sec.")
    # print(f"Memory usage: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)} MB")
    # uncomment if use linux
    print("resources")


def calculate_max_user_id():
    local_start = timeit.default_timer()
    max_left = 0
    max_right = 0
    link_count = 0
    for filename in [x for x in os.listdir(graph_path) if x.startswith("part")]:
        file = open(os.path.join(graph_path, filename))
        for line in csv.reader(file, delimiter='\t'):
            user = int(line[0])
            if user not in test_users:
                continue
            max_left = max(max_left, user)
            for friendship in line[1][2:len(line[1]) - 2].split("),("):
                link_count += 1
                friendship_parts = friendship.split(",")
                friend = int(friendship_parts[0])
                max_right = max(max_right, friend)
        file.close()
    print_resource_usage(local_start)
    return max(max_left, max_right), link_count


def save_csr(matrix, path):
    np.savez(path, data=matrix.data, indices=matrix.indices, indptr=matrix.indptr, shape=matrix.shape)


def load_csr(path):
    loaded = np.load(f"{path}.npz")
    return csr_matrix((loaded['data'], loaded['indices'], loaded['indptr']), shape=loaded['shape'])


def load_test_users():
    test_users = set()
    for line in csv.reader(open(test_users_path)):
        test_users.add(int(line[0]))
    return test_users


def load_birth_dates():
    local_start = timeit.default_timer()
    birth_dates = np.zeros(max_user_id, dtype=np.int32)
    for filename in [x for x in os.listdir(demography_path) if x.startswith("part")]:
        file = open(os.path.join(demography_path, filename))
        for line in csv.reader(file, delimiter='\t'):
            user = int(line[0])
            birth_dates[user - 1] = int(line[2]) if line[2] != '' else 0
        file.close()
    print("Loaded birth dates.")
    print_resource_usage(local_start)
    return birth_dates


def load_test_graph():
    local_start = timeit.default_timer()
    user_from = np.zeros(links_count, dtype=np.int32)
    user_to = np.zeros(links_count, dtype=np.int32)
    mask = np.ones(links_count, dtype=np.int32)
    cur = 0  # Index of the current link
    for filename in [x for x in os.listdir(graph_path) if x.startswith("part")]:
        file = open(os.path.join(graph_path, filename))
        for line in csv.reader(file, delimiter='\t'):
            user = int(line[0])
            if user not in test_users:
                continue
            for friendship in line[1][2:len(line[1]) - 2].split("),("):
                friendship_parts = friendship.split(",")
                user_from[cur] = user - 1
                user_to[cur] = int(friendship_parts[0]) - 1
                mask[cur] = int(friendship_parts[1]) | 1
                cur += 1
        file.close()
    print("Loaded graph as COO.")
    print_resource_usage(local_start)

    local_start = timeit.default_timer()
    graph = coo_matrix((mask, (user_from, user_to)), shape=(max_user_id, max_user_id)).tocsr()
    del user_from
    del user_to
    del mask
    print("Converted graph to CSR.")
    print_resource_usage(local_start)
    return graph


def extract_and_save_data():
    birth_dates = load_birth_dates()
    np.save(birth_dates_path, birth_dates)
    print("Saved birth dates.")
    del birth_dates
    test_graph = load_test_graph()
    save_csr(test_graph, test_graph_path)
    print("Saved the graph as CSR.")
    del test_graph


def MedianPredict():
    start = timeit.default_timer()
    with open(os.path.join(results_path, "prediction_direct_strange.csv"), 'w') as output:
        writer = csv.writer(output, delimiter=',')
        for user in test_users:
            ptr = test_graph_csr.indptr[user - 1]
            ptr_next = test_graph_csr.indptr[user]
            friends_id_iter = map(lambda x: birth_dates[x], test_graph_csr.indices[ptr:ptr_next])

            median_date = statistics.median(np.fromiter(friends_id_iter, dtype=np.int))
            # average_date = sum(friends_id_iter) / (ptr_next - ptr)
            # date = (median_date + average_date) / 2

            writer.writerow([user, median_date])
        print_resource_usage(start)


def predict():
    start = timeit.default_timer()
    with open(os.path.join(results_path, "prediction_filtered2.csv"), 'w') as output:
        writer = csv.writer(output, delimiter=',')
        for user in test_users:
            ptr = test_graph_csr.indptr[user - 1]
            ptr_next = test_graph_csr.indptr[user]

            friends_dates = np.fromiter(map(lambda x: birth_dates[x], test_graph_csr.indices[ptr:ptr_next]),
                                        dtype=np.int)
            median_date = statistics.median(friends_dates)
            mean_date = statistics.mean(friends_dates)

            filtered_age_list = []
            schoolmates_age_list = []
            college_age_list = []
            army_age_list = []

            for i in range(ptr, ptr_next):
                friend_id = test_graph_csr.indices[i]
                mask = test_graph_csr.data[i]
                age = birth_dates[friend_id]

                # magic number TODO pre-filter data
                if ~(mask & (-1099528)):
                    filtered_age_list.append(age)

                    if mask & (1 << 10):
                            schoolmates_age_list.append(age)
                    if mask & (1 << 14):
                            college_age_list.append(age)
                    if mask & (1 << 15):
                            army_age_list.append(age)
            dates = []

            # TODO understand why median doesn't work
            if filtered_age_list.__len__() > 0:
                dates.append(float(statistics.mean(filtered_age_list)))

            if college_age_list.__len__() > 0:
                college_age = statistics.mean(college_age_list)
                dates.append(float(college_age))

            if schoolmates_age_list.__len__() > 0:
                schoolmates_date = statistics.mean(schoolmates_age_list)
                dates.append(float(schoolmates_date))

            dates.append((median_date + mean_date) / 2)
            date = statistics.mean(dates)

            writer.writerow([user, date])
        print_resource_usage(start)


# Pre-calculated values
max_user_id = 47289241
links_count = 27261623

test_users = load_test_users()

test_graph_csr = load_csr("prediction/test_graph")
birth_dates = np.load(f"{birth_dates_path}.npy")
predict()
print("All done.")
print_resource_usage(global_start)
