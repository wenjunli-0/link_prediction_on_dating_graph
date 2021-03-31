import numpy as np
import pandas as pd


# distinct id who send msg to the other party; and id who receive msg from the other party
'''
msg_m = pd.read_csv('matching_msg_m.csv').values
msg_f = pd.read_csv('matching_msg_f.csv').values

rows_msg_m = msg_m.shape[0]
rows_msg_f = msg_f.shape[0]
print('there are {} lines in msg_m, and {} lines in msg_f'.format(rows_msg_m, rows_msg_f))


count_m_send = 0
count_f_send = 0

distinct_m_id_send = []
distinct_f_id_send = []
for i in range(rows_msg_m-1):
    id_curt = msg_m[i][0]
    id_next = msg_m[i+1][0]
    if id_curt != id_next:
        distinct_m_id_send.append(id_curt)
        count_m_send += 1

for i in range(rows_msg_f-1):
    id_curt = msg_f[i][0]
    id_next = msg_f[i+1][0]
    if id_curt != id_next:
        distinct_f_id_send.append(id_curt)
        count_f_send += 1

print('there are {} distinct males that have sent msg the other party'.format(count_m_send))
print('there are {} distinct females that have sent msg the other party'.format(count_f_send))

np.savetxt("./march_26/distinct_m_id_send.csv", distinct_m_id_send, delimiter=',', fmt='%d')
np.savetxt("./march_26/distinct_f_id_send.csv", distinct_f_id_send, delimiter=',', fmt='%d')


# distinct id who receive msg
count_f_receive = 0
count_m_receive = 0

distinct_f_id_receive = []
distinct_m_id_receive = []
for i in range(rows_msg_m-1):
    id_curt = msg_m[i][1]
    id_next = msg_m[i+1][1]
    if id_curt != id_next:
        distinct_f_id_receive.append(id_curt)
        count_f_receive += 1

for i in range(rows_msg_f-1):
    id_curt = msg_f[i][1]
    id_next = msg_f[i+1][1]
    if id_curt != id_next:
        distinct_m_id_receive.append(id_curt)
        count_m_receive += 1

print('there are {} distinct males that have received msg the other party'.format(count_m_receive))
print('there are {} distinct females that have received msg the other party'.format(count_f_receive))

np.savetxt("./march_26/distinct_f_id_receive.csv", distinct_f_id_receive, delimiter=',', fmt='%d')
np.savetxt("./march_26/distinct_m_id_receive.csv", distinct_m_id_receive, delimiter=',', fmt='%d')
'''


# count the # of repeated edges in msg_m, and store those edges w/o repeat
'''
count_repeated_edge_msgm = 0
repeat_edge_msgm = []
for i in range(rows_msg_m-1):
    m_id_curt = msg_m[i][0]
    f_id_curt = msg_m[i][1]
    m_id_next = msg_m[i+1][0]
    f_id_next = msg_m[i+1][1]
    if m_id_curt == m_id_next and f_id_curt == f_id_next:
        repeat_edge_msgm.append(m_id_curt)
        count_repeated_edge_msgm += 1

print(repeat_edge_msgm)
print('there are {} repeated edges from a same male to a same female'.format(count_repeated_edge_msgm))


count = 0
msg_m_wo_repeated_edge = np.zeros((rows_msg_m - count_repeated_edge_msgm, 2))
for i in range(rows_msg_m-1):
    m_id_curt = msg_m[i][0]
    f_id_curt = msg_m[i][1]
    m_id_next = msg_m[i+1][0]
    f_id_next = msg_m[i+1][1]
    if m_id_curt == m_id_next and f_id_curt == f_id_next:
        i += 1
    else:
        msg_m_wo_repeated_edge[count][0] = msg_m[i][0]
        msg_m_wo_repeated_edge[count][1] = msg_m[i][1]
        count += 1
print('there are {} distinct edges in msg_m'.format(count))
np.savetxt("./march_26/msg_m_wo_repeated_edge.csv", msg_m_wo_repeated_edge, delimiter=',', fmt='%d')
'''


# count the # of repeated edges in msg_f, and store those edges w/o repeat
'''
count_repeated_edge_msgf = 0
repeat_edge_msgf = []
for i in range(rows_msg_f-1):
    m_id_curt = msg_f[i][0]
    f_id_curt = msg_f[i][1]
    m_id_next = msg_f[i+1][0]
    f_id_next = msg_f[i+1][1]
    if m_id_curt == m_id_next and f_id_curt == f_id_next:
        repeat_edge_msgf.append(m_id_curt)
        count_repeated_edge_msgf += 1

print(repeat_edge_msgf)
print('there are {} repeated edges from a same female to a same male'.format(count_repeated_edge_msgf))


count = 0
msg_f_wo_repeated_edge = np.zeros((rows_msg_f - count_repeated_edge_msgf, 2))
for i in range(rows_msg_f-1):
    f_id_curt = msg_f[i][0]
    m_id_curt = msg_f[i][1]
    f_id_next = msg_f[i+1][0]
    m_id_next = msg_f[i+1][1]
    if f_id_curt == f_id_next and m_id_curt == m_id_next:
        i += 1
    else:
        msg_f_wo_repeated_edge[count][0] = msg_f[i][0]
        msg_f_wo_repeated_edge[count][1] = msg_f[i][1]
        count += 1
print('there are {} distinct edges in msg_f'.format(count))
np.savetxt("./march_26/msg_f_wo_repeated_edge.csv", msg_f_wo_repeated_edge, delimiter=',', fmt='%d')
'''


# find id_f who has been rec to id_m in distinct_m_id_send
profile_m = pd.read_csv('profile_m.csv').values
profile_f = pd.read_csv('profile_f.csv').values

profile_m[:, 1] = 0     # convert 'm' to 0
profile_f[:, 1] = 1     # convert 'f' to 1


# collect attributes of id_m in distinct_m_id_send, and attributes of id_f in distinct_f_id_receive
'''
distinct_m_id_send = pd.read_csv('./march_26/distinct_m_id_send.csv').values
rows_distinct_m_id_send = distinct_m_id_send.shape[0]
attribute_of_distinct_m_id_send = np.zeros((rows_distinct_m_id_send, 35))
count = 0
for i in range(rows_distinct_m_id_send):
    m_id = distinct_m_id_send[i][0]
    index = np.where(profile_m[:, 0] == m_id)       # locate the m_id in profile_m
    attribute_of_distinct_m_id_send[count] = profile_m[index]
    count += 1
np.savetxt("./march_26/attribute_of_distinct_m_id_send.csv", attribute_of_distinct_m_id_send, delimiter=',', fmt='%d')


distinct_f_id_receive = pd.read_csv('./march_26/distinct_f_id_receive.csv').values
rows_distinct_f_id_receive = distinct_f_id_receive.shape[0]
attribute_of_distinct_f_id_receive = np.zeros((rows_distinct_f_id_receive, 35))
count = 0
for i in range(rows_distinct_f_id_receive):
    f_id = distinct_f_id_receive[i][0]
    index = np.where(profile_f[:, 0] == f_id)       # locate the f_id in profile_f
    attribute_of_distinct_f_id_receive[count] = profile_f[index]
    count += 1
np.savetxt("./march_26/attribute_of_distinct_f_id_receive.csv", attribute_of_distinct_f_id_receive, delimiter=',', fmt='%d')
'''


# collect attributes of id_f in distinct_f_id_send, and attributes of id_m in distinct_m_id_receive
'''
distinct_f_id_send = pd.read_csv('./march_26/distinct_f_id_send.csv').values
rows_distinct_f_id_send = distinct_f_id_send.shape[0]
attribute_of_distinct_f_id_send = np.zeros((rows_distinct_f_id_send, 35))
count = 0
for i in range(rows_distinct_f_id_send):
    f_id = distinct_f_id_send[i][0]
    index = np.where(profile_f[:, 0] == f_id)       # locate the m_id in profile_m
    attribute_of_distinct_f_id_send[count] = profile_f[index]
    count += 1
np.savetxt("./march_26/attribute_of_distinct_f_id_send.csv", attribute_of_distinct_f_id_send, delimiter=',', fmt='%d')


distinct_m_id_receive = pd.read_csv('./march_26/distinct_m_id_receive.csv').values
rows_distinct_m_id_receive = distinct_m_id_receive.shape[0]
attribute_of_distinct_m_id_receive = np.zeros((rows_distinct_m_id_receive, 35))
count = 0
for i in range(rows_distinct_m_id_receive):
    m_id = distinct_m_id_receive[i][0]
    index = np.where(profile_m[:, 0] == m_id)       # locate the f_id in profile_f
    attribute_of_distinct_m_id_receive[count] = profile_m[index]
    count += 1
np.savetxt("./march_26/attribute_of_distinct_m_id_receive.csv", attribute_of_distinct_m_id_receive, delimiter=',', fmt='%d')
'''


# distinct id_f who only receive click from a id_m, but do not receive msg from it
'''
matching_msg = pd.read_csv('matching_msg.csv').values
matching_click = pd.read_csv('matching_click.csv').values
print(matching_msg.shape)
print(matching_click.shape)


# f_id_list
f_id_list = profile_f[:, 0]

# f_id_in_msg_m
matching_msg_m = pd.read_csv('matching_msg_m.csv').values
f_id_in_msg_m = matching_msg_m[:, 1]

# f_id_in_click_m
f_id_in_click_m = matching_click[:, 1]

print('the length of f_id_list: ', f_id_list.shape[0])
print('the length of f_id_in_msg_m: ', f_id_in_msg_m.shape[0])
print('the length of f_id_in_click_m: ', f_id_in_click_m.shape[0])


distinct_f_id_be_clicked_by_m_but_not_msg = []
for i in range(f_id_list.shape[0]):
    f_id = f_id_list[i]

    # ensure f_id is not in msg_m
    if not (f_id_in_msg_m == f_id).any():
        if (f_id_in_click_m == f_id).any():
            distinct_f_id_be_clicked_by_m_but_not_msg.append(f_id)

np.savetxt("./march_26/distinct_f_id_be_clicked_by_m_but_not_msg.csv", distinct_f_id_be_clicked_by_m_but_not_msg, delimiter=',', fmt='%d')


# distinct id_m who only receive click from a id_f, but do not receive msg from it
# m_id_list
m_id_list = profile_m[:, 0]

# m_id_in_msg_f
matching_msg_f = pd.read_csv('matching_msg_f.csv').values
m_id_in_msg_f = matching_msg_f[:, 1]

# m_id_in_click_f
m_id_in_click_f = matching_click[:, 1]

print('the length of m_id_list: ', m_id_list.shape[0])
print('the length of m_id_in_msg_f: ', m_id_in_msg_f.shape[0])
print('the length of m_id_in_click_f: ', m_id_in_click_f.shape[0])


distinct_m_id_be_clicked_by_f_but_not_msg = []
for i in range(m_id_list.shape[0]):
    m_id = m_id_list[i]

    # ensure f_id is not in msg_m
    if not (m_id_in_msg_f == m_id).any():
        if (m_id_in_click_f == m_id).any():
            distinct_m_id_be_clicked_by_f_but_not_msg.append(m_id)

np.savetxt("./march_26/distinct_m_id_be_clicked_by_f_but_not_msg.csv", distinct_m_id_be_clicked_by_f_but_not_msg, delimiter=',', fmt='%d')
'''


# find edges between distinct_m_id_send_msg and distinct_f_id_be_clicked_but_not_msg
'''
matching_click = pd.read_csv('matching_click.csv').values

m_id_list = profile_m[:, 0]         # 344552 m users
f_id_list = profile_f[:, 0]         # 203843 f users

count_m = 0
count_f = 0
matching_click_from_m_to_f = np.zeros((124872, 4))
matching_click_from_f_to_m = np.zeros((59419, 4))
for i in range(matching_click.shape[0]):
    id = matching_click[i][0]

    if (f_id_list == id).any():
        matching_click_from_f_to_m[count_f] = matching_click[i]
        count_f += 1
    else:
        matching_click_from_m_to_f[count_m] = matching_click[i]
        count_m += 1

    if (i+1) % 10000 == 0:
        print('{} / {} is done'.format(i+1, matching_click.shape[0]))

np.savetxt("matching_click_from_m_to_f.csv", matching_click_from_m_to_f, delimiter=',', fmt='%d')
np.savetxt("matching_click_from_f_to_m.csv", matching_click_from_f_to_m, delimiter=',', fmt='%d')
print('in matching_click, there are {} edges by m, {} edges by female'.format(count_m, count_f))
print('count matching click edges is finished...')
'''


# unique edges in matching_click and matching_msg
'''
matching_click_from_m_to_f = pd.read_csv('./march_26/matching_click_from_m_to_f.csv').values        # 124872, click edge from m -> f
matching_msg_from_m_to_f = pd.read_csv('./march_26/matching_msg_from_m_to_f.csv').values            # 36024 / 38624,  msg edges from m -> f
print(matching_click_from_m_to_f.shape)
print(matching_msg_from_m_to_f.shape)

# matching_msg_m_unique_edges = np.array(list(set([tuple(t) for t in matching_click_from_m_to_f])))       # find unique rows; same using np.unique()
# matching_click_f_unique_edges = np.array(list(set([tuple(t) for t in matching_msg_from_m_to_f])))       # unique_rows = np.unique(original_array, axis=0)

matching_click_m_unique_edges = np.unique(matching_click_from_m_to_f, axis=0)
matching_msg_m_unique_edges = np.unique(matching_msg_from_m_to_f, axis=0)
print('click m unique: ', matching_click_m_unique_edges.shape)
print('msg m unique: ', matching_msg_m_unique_edges.shape)
np.savetxt("./march_30/matching_click_m_unique_edges.csv", matching_click_m_unique_edges, delimiter=',', fmt='%d')
np.savetxt("./march_30/matching_msg_m_unique_edges.csv", matching_msg_m_unique_edges, delimiter=',', fmt='%d')


matching_click_from_f_to_m = pd.read_csv('./march_26/matching_click_from_f_to_m.csv').values        # 59419, click edge from m -> f
matching_msg_from_f_to_m = pd.read_csv('./march_26/matching_msg_from_f_to_m.csv').values            # 9396 / 10039,  msg edges from m -> f
print(matching_click_from_f_to_m.shape)
print(matching_msg_from_f_to_m.shape)

matching_click_f_unique_edges = np.unique(matching_click_from_f_to_m, axis=0)
matching_msg_f_unique_edges = np.unique(matching_msg_from_f_to_m, axis=0)
print('click f unique: ', matching_click_f_unique_edges.shape)
print('msg f unique: ', matching_msg_f_unique_edges.shape)
np.savetxt("./march_30/matching_click_f_unique_edges.csv", matching_click_f_unique_edges, delimiter=',', fmt='%d')
np.savetxt("./march_30/matching_msg_f_unique_edges.csv", matching_msg_f_unique_edges, delimiter=',', fmt='%d')
'''


'''
matching_click_m_unique_edges = pd.read_csv('./march_30/matching_click_m_unique_edges.csv').values
matching_msg_m_unique_edges = pd.read_csv('./march_30/matching_msg_m_unique_edges.csv').values
matching_click_f_unique_edges = pd.read_csv('./march_30/matching_click_f_unique_edges.csv').values
matching_msg_f_unique_edges = pd.read_csv('./march_30/matching_msg_f_unique_edges.csv').values


def setdiff2d(a, b):
    # check that casting to void will create equal size elements
    assert a.shape[1:] == b.shape[1:]
    assert a.dtype == b.dtype

    # compute dtypes
    void_dt = np.dtype((np.void, a.dtype.itemsize * np.prod(a.shape[1:])))
    orig_dt = np.dtype((a.dtype, a.shape[1:]))

    # convert to 1d void arrays
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    a_void = a.reshape(a.shape[0], -1).view(void_dt)
    b_void = b.reshape(b.shape[0], -1).view(void_dt)

    # Get indices in a that are also in b
    return np.setdiff1d(a_void, b_void).view(orig_dt)


click_but_not_msg_m = setdiff2d(matching_click_m_unique_edges, matching_msg_m_unique_edges)
click_but_not_msg_f = setdiff2d(matching_click_f_unique_edges, matching_msg_f_unique_edges)
print(click_but_not_msg_m.shape)
print(click_but_not_msg_f.shape)
np.savetxt("./march_30/click_but_not_msg_m.csv", click_but_not_msg_m, delimiter=',', fmt='%d')
np.savetxt("./march_30/click_but_not_msg_f.csv", click_but_not_msg_f, delimiter=',', fmt='%d')
'''


msg_m_wo_repeated_edge = pd.read_csv('./march_26/msg_m_wo_repeated_edge.csv').values        # 124872, click edge from m -> f
msg_f_wo_repeated_edge = pd.read_csv('./march_26/msg_f_wo_repeated_edge.csv').values            # 36024 / 38624,  msg edges from m -> f
print(msg_m_wo_repeated_edge.shape)
print(msg_f_wo_repeated_edge.shape)


# only keep those lines that m_id is in distinct_m_id_send_msg
click_but_not_msg_m = pd.read_csv('./march_30/click_but_not_msg_m.csv').values
distinct_m_id_send_msg = pd.read_csv('./march_30/distinct_m_id_send_msg.csv').values

edges_mSendMsg_fClickedByTheseMButNotMsg = np.zeros((63561, 2))
count = 0
for i in range(click_but_not_msg_m.shape[0]):
    m_id = click_but_not_msg_m[i][0]

    if (distinct_m_id_send_msg == m_id).any():
        edges_mSendMsg_fClickedByTheseMButNotMsg[count][0] = m_id
        edges_mSendMsg_fClickedByTheseMButNotMsg[count][1] = click_but_not_msg_m[i][1]
        count += 1

print(count)
np.savetxt("./march_30/edges_mSendMsg_fClickedByTheseMButNotMsg.csv", edges_mSendMsg_fClickedByTheseMButNotMsg, delimiter=',', fmt='%d')


click_but_not_msg_f = pd.read_csv('./march_30/click_but_not_msg_f.csv').values
distinct_f_id_send_msg = pd.read_csv('./march_30/distinct_f_id_send_msg.csv').values

edges_fSendMsg_mClickedByTheseFButNotMsg = np.zeros((30057, 2))
count = 0
for i in range(click_but_not_msg_f.shape[0]):
    f_id = click_but_not_msg_f[i][0]

    if (distinct_f_id_send_msg == f_id).any():
        edges_fSendMsg_mClickedByTheseFButNotMsg[count][0] = f_id
        edges_fSendMsg_mClickedByTheseFButNotMsg[count][1] = click_but_not_msg_f[i][1]
        count += 1

print(count)
np.savetxt("./march_30/edges_fSendMsg_mClickedByTheseFButNotMsg.csv", edges_fSendMsg_mClickedByTheseFButNotMsg, delimiter=',', fmt='%d')




