import pandas as pd
import numpy as np


def print_top_5_by_smallest_indices_np(str_arr, num_arr):
    # Get indices of the 5 smallest numbers
    smallest_indices = np.argsort(num_arr)[:5]

    # Print corresponding strings and numbers
    for idx in smallest_indices:
        print(f"Index {idx}: {str_arr[idx]} (Number: {num_arr[idx]})")


def count_unique_directed_triangles(adj):
    n = adj.shape[0]
    counted_triangles = set()
    count = 0

    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            for k in range(j+1, n):
                if k == i or k == j:
                    continue
                # 判断是否有三角形 i->j, j->k, k->i
                if adj[i][j] and adj[j][k] and adj[k][i]:
                    # 规范化三角形的节点顺序：按节点编号排序
                    triangle_nodes = tuple(sorted([i, j, k]))
                    if triangle_nodes not in counted_triangles:
                        counted_triangles.add(triangle_nodes)
                        count += 1
    return count


min_value = 100000
conns = np.array([])
files = np.array([], dtype=object)

for i in range(1, 89):
    if i < 10:
        i_str = f"0{str(i)}"
    else:
        i_str = str(i)
    path = f"S0{i_str}.csv"

    df = pd.read_csv(path, header=None)
    coup2 = df.to_numpy()
    conn2 = np.copy(coup2)
    # conn2[conn2 < 0.05] = 0
    conn2[conn2 != 0] = 1

    a = np.sum(conn2) / 90 / 89
    conns = np.append(conns, a)
    files = np.append(files, path)

    if min_value > a:
        min_value = a
        print(f"The lowest connectivity is in {path}:", min_value)
        count_tri = count_unique_directed_triangles(conn2)
        print("The # of triangle is: ", count_tri)

print_top_5_by_smallest_indices_np(files, conns)



