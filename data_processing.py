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
matching_click = pd.read_csv('matching_click.csv').values
distinct_f_id_receive_msg = pd.read_csv('./march_26/distinct_f_id_receive_msg.csv').values

distinct_m_id_send_msg = pd.read_csv('./march_26/distinct_m_id_send_msg.csv').values
distinct_f_id_be_clicked_by_m_but_not_msg = pd.read_csv('./march_26/distinct_f_id_be_clicked_by_m_but_not_msg.csv').values


f_id_list = profile_f[:, 0]

click_m_wo_repeated_edge = np.zeros((3698, 2))

count = 0
print('there are total of {} lines'.format(distinct_m_id_send_msg.shape[0]))
print('there are total of {} lines'.format(distinct_f_id_receive_msg.shape[0]))


click_m_id_list = []
click_f_id_list = []

for j in range(distinct_m_id_send_msg.shape[0]):
    m_id = distinct_m_id_send_msg[j]

    for k in range(distinct_f_id_receive_msg.shape[0]):
        f_id = distinct_f_id_receive_msg[k]

        if (matching_click[:, 0] == m_id).any():
            if not (matching_click[:, 1] == f_id).any():
                click_m_id_list.append(m_id)
                click_f_id_list.append(f_id)
                count += 1

    if (j+1) % 100 == 0:
        print('{} / {} lines are done'.format(j+1, distinct_m_id_send_msg.shape[0]))

print('there are {} edges between m_id () and f_id ()'.format(count))
np.savetxt("./march_26/click_m_id_list.csv", click_m_id_list, delimiter=',', fmt='%d')
np.savetxt("./march_26/click_f_id_list.csv", click_f_id_list, delimiter=',', fmt='%d')



print('finished')



