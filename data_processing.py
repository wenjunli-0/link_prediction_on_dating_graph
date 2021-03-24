import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import random
from sklearn.metrics import classification_report


# convert matching_data.txt to csv
'''
matching_data = np.zeros((8599012, 4))
f = open('matching_data.txt', 'r')
line = f.readline()
i = 0
while line:
    splitedlist = line[:-1].split(' ')
    if splitedlist[3] == 'rec':
        splitedlist[3] = '0'
    elif splitedlist[3] == 'click':
        splitedlist[3] = '1'
    elif splitedlist[3] == 'msg':
        splitedlist[3] = '2'
    splitedlist = list(map(int, splitedlist))
    matching_data[i] = splitedlist
    i += 1
    line = f.readline()

f.close()
np.savetxt("matching_data.csv", matching_data, delimiter=',')
'''


# count the edge # of rec, click, msg
'''
rec, click, msg = 0, 0, 0
for i in range(rows):
    if matching_data[i][3] == 0:
        rec += 1
    elif matching_data[i][3] == 1:
        click += 1
    elif matching_data[i][3] == 2:
        msg += 1
print('there are {} rec, {} click, {} msg. Total action # is {}'.format(rec, click, msg, rec+click+msg))
'''


# split the matching data into: matching_rec, matching_click, matching_msg
'''
matching_rec = np.zeros((rec, cols))
matching_click = np.zeros((click, cols))
matching_msg = np.zeros((msg, cols))
rec, click, msg = 0, 0, 0

for i in range(rows):
    if matching_data[i][3] == 0:
        matching_rec[rec] = matching_data[i]
        rec += 1
    elif matching_data[i][3] == 1:
        matching_click[click] = matching_data[i]
        click += 1
    elif matching_data[i][3] == 2:
        matching_msg[msg] = matching_data[i]
        msg += 1

np.savetxt("matching_rec.csv", matching_rec, delimiter=',')
np.savetxt("matching_click.csv", matching_click, delimiter=',')
np.savetxt("matching_msg.csv", matching_msg, delimiter=',')
'''


# count msg edges by male, female
'''
profile_m = pd.read_csv('profile_m.csv')
profile_f = pd.read_csv('profile_f.csv')
matching_msg = pd.read_csv('matching_msg.csv')

profile_m = profile_m.values
profile_f = profile_f.values
matching_msg = matching_msg.values

rows_msg = matching_msg.shape[0]
cols_msg = matching_msg.shape[1]
print('there are {} rows, and {} cols in matching_msg'.format(rows_msg, cols_msg))

rows_m = profile_m.shape[0]
cols_m = profile_m.shape[1]
rows_f = profile_f.shape[0]
cols_f = profile_f.shape[1]
print('there are {} male users, and {} female users'.format(rows_m, rows_f))


id_m = profile_m[:, 0].flatten()
id_f = profile_f[:, 0].flatten()

edge_m = 0
edge_f = 0
matching_msg_m = np.zeros((38624, 4))
matching_msg_f = np.zeros((10039, 4))

for i in range(rows_msg):
    id = matching_msg[i][0]
    if (id_m == id).any():
        matching_msg_m[edge_m] = matching_msg[i]
        edge_m += 1
    elif (id_f == id).any():
        matching_msg_f[edge_f] = matching_msg[i]
        edge_f += 1

print('there are {} male edges, and {} female edges'.format(edge_m, edge_f))
np.savetxt("matching_msg_m.csv", matching_msg_m, delimiter=',')
np.savetxt("matching_msg_f.csv", matching_msg_f, delimiter=',')
'''


# find m & f attributes of msg_m edges, and f & m attributes of msg_f edges
'''
profile_m = pd.read_csv('profile_m.csv')
profile_f = pd.read_csv('profile_f.csv')
matching_msg_m = pd.read_csv('matching_msg_m.csv')
matching_msg_f = pd.read_csv('matching_msg_f.csv')

profile_m = profile_m.values
profile_f = profile_f.values
matching_msg_m = matching_msg_m.values
matching_msg_f = matching_msg_f.values

# convert 'm' to 0, 'f' to 1
profile_m[:, 1] = 0
profile_f[:, 1] = 1


rows_msg_m = matching_msg_m.shape[0]
cols_msg_m = matching_msg_m.shape[1]
rows_msg_f = matching_msg_f.shape[0]
cols_msg_f = matching_msg_f.shape[1]
print('there are {} rows, and {} cols in matching_msg_m'.format(rows_msg_m, cols_msg_m))
print('there are {} rows, and {} cols in matching_msg_f'.format(rows_msg_f, cols_msg_f))


# attributes of m/f, for edges created by m
att_edgem_m = np.zeros((rows_msg_m, 35))
att_edgem_f = np.zeros((rows_msg_m, 35))
count_m_m = 0
count_m_f = 0
for i in range(rows_msg_m):
    id_m = matching_msg_m[i][0]
    id_f = matching_msg_m[i][1]

    # locate the id_m in profile_m
    index_m = np.where(profile_m[:, 0] == id_m)
    att_edgem_m[count_m_m] = profile_m[index_m]
    count_m_m += 1

    # locate the id_f in profile_f
    index_f = np.where(profile_f[:, 0] == id_f)
    att_edgem_f[count_m_f] = profile_f[index_f]
    count_m_f += 1

np.savetxt("att_edgem_m.csv", att_edgem_m, delimiter=',')
np.savetxt("att_edgem_f.csv", att_edgem_f, delimiter=',')


# attributes of m/f, for edges created by f
att_edgef_f = np.zeros((rows_msg_f, 35))
att_edgef_m = np.zeros((rows_msg_f, 35))
count_f_f = 0
count_f_m = 0
for i in range(rows_msg_f):
    id_f = matching_msg_f[i][0]
    id_m = matching_msg_f[i][1]

    # locate the id_f in profile_f
    index_f = np.where(profile_f[:, 0] == id_f)
    att_edgef_f[count_f_f] = profile_f[index_f]
    count_f_f += 1

    # locate the id_m in profile_m
    index_m = np.where(profile_m[:, 0] == id_m)
    att_edgef_m[count_f_m] = profile_m[index_m]
    count_f_m += 1

np.savetxt("att_edgef_f.csv", att_edgef_f, delimiter=',')
np.savetxt("att_edgef_m.csv", att_edgef_m, delimiter=',')
'''


# find f that does not receive a msg from the male in att_edgem_m
# load data
'''
profile_m = pd.read_csv('profile_m.csv')
profile_f = pd.read_csv('profile_f.csv')
profile_m = profile_m.values
profile_f = profile_f.values

# convert 'm' to 0, 'f' to 1
profile_m[:, 1] = 0
profile_f[:, 1] = 1

att_edgem_m = pd.read_csv('att_edgem_m.csv')    # shape = (38624, 35)
att_edgem_f = pd.read_csv('att_edgem_f.csv')    # shape = (38624, 35)
att_edgef_f = pd.read_csv('att_edgef_f.csv')    # shape = (10039, 35)
att_edgef_m = pd.read_csv('att_edgef_m.csv')    # shape = (10039, 35)
att_edgem_m = att_edgem_m.values
att_edgem_f = att_edgem_f.values
att_edgef_f = att_edgef_f.values
att_edgef_m = att_edgef_m.values

matching_rec = pd.read_csv('matching_rec.csv')
matching_rec = matching_rec.values

matching_msg_m = pd.read_csv('matching_msg_m.csv')
matching_msg_m = matching_msg_m.values

rows_rec = matching_rec.shape[0]
cols_rec = matching_rec.shape[1]

id_m_list = profile_m[:, 0].flatten()
id_f_list = profile_f[:, 0].flatten()

recEdge_m = 0
recEdge_f = 0
'''
'''
att_recm_f = np.zeros((att_edgem_f.shape[0], att_edgem_f.shape[1]))
count = 0
list_id_m = [0]
for i in range(att_edgem_m.shape[0]):
    id_m = att_edgem_m[i][0]        # id_m who do send msg to a f
    list_id_m.append(id_m)
    this_id_m = list_id_m[-1]
    last_id_m = list_id_m[-2]
    # print('id_m who do send msg to a f: ', id_m)

    index_f_all = np.where((matching_rec[:, 0] == id_m))   # index of all id_f who do not receive a msg from id_m
    # print('index_f_all who do not receive a msg from id_m', index_f_all)
    # print('shape of index_f_all: ', index_f_all[0].shape)

    # choose the j-th as index_f, check if it's a f_id
    size = index_f_all[0].shape
    for j in range(size[0]):
        if this_id_m == last_id_m:
            j += 1      # if this id_m is the same as the last one, take the next j-th

        index_f = index_f_all[0][j]         # index of the f, who does not receive msg
        id_f = matching_rec[index_f][1]
        # print('index_f: {}, id_f: {}, {}-th'.format(index_f, id_f, j))

        # only record the attri. if it's a f
        if (id_f_list == id_f).any():
            index_f_checked = np.where(profile_f[:, 0] == id_f)        # the index of the id_f
            att_recm_f[count] = profile_f[index_f_checked]             #
            # print('the attri. of this f: {}'.format(profile_f[index_f_checked]))
            count += 1
            break
        else:
            j += 1      # if this id_f is not a f, then try next one

    if count == att_edgem_m.shape[0]:
        break

np.savetxt("att_recm_f.csv", att_recm_f, delimiter=',')
'''

# find m that does not receive a msg from the female in att_edgef_f
# load data
'''
profile_m = pd.read_csv('profile_m.csv')
profile_f = pd.read_csv('profile_f.csv')
profile_m = profile_m.values
profile_f = profile_f.values

# convert 'm' to 0, 'f' to 1
profile_m[:, 1] = 0
profile_f[:, 1] = 1

att_edgem_m = pd.read_csv('att_edgem_m.csv')    # shape = (38624, 35)
att_edgem_f = pd.read_csv('att_edgem_f.csv')    # shape = (38624, 35)
att_edgef_f = pd.read_csv('att_edgef_f.csv')    # shape = (10039, 35)
att_edgef_m = pd.read_csv('att_edgef_m.csv')    # shape = (10039, 35)
att_edgem_m = att_edgem_m.values
att_edgem_f = att_edgem_f.values
att_edgef_f = att_edgef_f.values
att_edgef_m = att_edgef_m.values

matching_rec = pd.read_csv('matching_rec.csv')
matching_rec = matching_rec.values

matching_msg_m = pd.read_csv('matching_msg_m.csv')
matching_msg_m = matching_msg_m.values

rows_rec = matching_rec.shape[0]
cols_rec = matching_rec.shape[1]

id_m_list = profile_m[:, 0].flatten()
id_f_list = profile_f[:, 0].flatten()

recEdge_m = 0
recEdge_f = 0
'''
'''
att_recf_m = np.zeros((att_edgef_m.shape[0], att_edgef_m.shape[1]))
count = 0
list_id_f = [0]
for i in range(att_edgef_f.shape[0]):
    id_f = att_edgef_f[i][0]        # id_f who do send msg to a m
    list_id_f.append(id_f)
    this_id_f = list_id_f[-1]
    last_id_f = list_id_f[-2]
    # print('id_f who do send msg to a m: ', id_f)

    index_m_all = np.where((matching_rec[:, 0] == id_f))   # index of all id_m who do not receive a msg from id_f
    # print('index_m_all who do not receive a msg from id_f', index_m_all)
    # print('shape of index_m_all: ', index_m_all[0].shape)

    # choose the j-th as index_m, check if it's a m_id
    size = index_m_all[0].shape
    for j in range(size[0]):
        if this_id_f == last_id_f:
            j += 1      # if this id_m is the same as the last one, take the next j-th

        index_m = index_m_all[0][j]         # index of the m, who does not receive msg
        id_m = matching_rec[index_m][1]
        # print('index_m: {}, id_m: {}, {}-th'.format(index_m, id_m, j))

        # only record the attri. if it's a m
        if (id_m_list == id_m).any():
            index_m_checked = np.where(profile_m[:, 0] == id_m)        # the index of the id_m
            att_recf_m[count] = profile_m[index_m_checked]             #
            # print('the attri. of this m: {}'.format(profile_m[index_m_checked]))
            count += 1
            break
        else:
            j += 1      # if this id_m is not a m, then try next one

    if count == att_edgef_f.shape[0]:
        break

np.savetxt("att_recf_m.csv", att_recf_m, delimiter=',')
'''


# find msgm_m, msgm_f; find msgf_f, msgf_m
ev_centrality_msg = pd.read_csv('/Users/wenjun/SMU/IS709Network/project/project_codes/centrality/new/ev_centrality_single_edge.csv')       # shape = (548395, 2)
od_centrality_msg = pd.read_csv('/Users/wenjun/SMU/IS709Network/project/project_codes/centrality/new/od_centrality_single_edge.csv')       # shape = (548395, 2)

ev_centrality_msg = ev_centrality_msg.values
od_centrality_msg = od_centrality_msg.values


att_edgem_m = pd.read_csv('att_edgem_m.csv')    # shape = (38624, 35)   # m who send a msg
att_edgem_f = pd.read_csv('att_edgem_f.csv')    # shape = (38624, 35)   # f who receive a msg
att_edgem_m = att_edgem_m.values
att_edgem_f = att_edgem_f.values

msgm_m_ev_centrality = np.zeros((38624, 2))
msgm_f_ev_centrality = np.zeros((38624, 2))
msgm_m_ev_centrality[:, 0] = att_edgem_m[:, 0]      # copy id_m in att_edgem_m to msgm_m_ev_centrality
msgm_f_ev_centrality[:, 0] = att_edgem_f[:, 0]      # copy id_f in att_edgem_f to msgm_f_ev_centrality


rows_msgm = att_edgem_m.shape[0]
for i in range(rows_msgm):
    # find centrality for each id_m in att_edgem_m
    id_m = att_edgem_m[i][0]
    if (id_m == od_centrality_msg[:, 0]).any():
        index_array = np.where(od_centrality_msg[:, 0] == id_m)
        index_int = index_array[0][0]
        msgm_m_ev_centrality[i][1] = od_centrality_msg[index_int][1]
    else:
        msgm_m_ev_centrality[i][1] = 0

    # find centrality for each id_f in att_edgem_f
    id_f = att_edgem_f[i][0]
    if (id_f == od_centrality_msg[:, 0]).any():
        index_array = np.where(od_centrality_msg[:, 0] == id_f)
        index_int = index_array[0][0]
        msgm_f_ev_centrality[i][1] = od_centrality_msg[index_int][1]
    else:
        msgm_f_ev_centrality[i][1] = 0

np.savetxt("./centrality/new/msgm_m_od_centrality.csv", msgm_m_ev_centrality, delimiter=',')
np.savetxt("./centrality/new/msgm_f_od_centrality.csv", msgm_f_ev_centrality, delimiter=',')


att_edgef_f = pd.read_csv('att_edgef_f.csv')    # shape = (10039, 35)   # f who send a msg
att_edgef_m = pd.read_csv('att_edgef_m.csv')    # shape = (10039, 35)   # m who receive a msg
att_edgef_f = att_edgef_f.values
att_edgef_m = att_edgef_m.values

msgf_f_ev_centrality = np.zeros((10039, 2))
msgf_m_ev_centrality = np.zeros((10039, 2))
msgf_f_ev_centrality[:, 0] = att_edgef_f[:, 0]      # copy id_m in att_edgem_m to msgm_m_ev_centrality
msgf_m_ev_centrality[:, 0] = att_edgef_m[:, 0]      # copy id_f in att_edgem_f to msgm_f_ev_centrality

rows_msgf = att_edgef_f.shape[0]
for i in range(rows_msgf):
    # find centrality for each id_f in att_edgef_f
    id_f = att_edgef_f[i][0]
    if (id_f == od_centrality_msg[:, 0]).any():
        index_array = np.where(od_centrality_msg[:, 0] == id_f)
        index_int = index_array[0][0]
        msgf_f_ev_centrality[i][1] = od_centrality_msg[index_int][1]
    else:
        msgf_f_ev_centrality[i][1] = 0

    # find centrality for each id_m in att_edgef_m
    id_m = att_edgef_m[i][0]
    if (id_m == od_centrality_msg[:, 0]).any():
        index_array = np.where(od_centrality_msg[:, 0] == id_m)
        index_int = index_array[0][0]
        msgf_m_ev_centrality[i][1] = od_centrality_msg[index_int][1]
    else:
        msgf_m_ev_centrality[i][1] = 0

np.savetxt("./centrality/new/msgf_f_od_centrality.csv", msgf_f_ev_centrality, delimiter=',')
np.savetxt("./centrality/new/msgf_m_od_centrality.csv", msgf_m_ev_centrality, delimiter=',')


# find recm_f
'''
att_recm_f = pd.read_csv('att_recm_f.csv')    # shape = (38624, 35)   # m who send a msg
att_recm_f = att_recm_f.values

recm_f_ev_centrality = np.zeros((38624, 2))
recm_f_ev_centrality[:, 0] = att_recm_f[:, 0]      # copy id_m in att_edgem_m to msgm_m_ev_centrality

rows_recm = att_recm_f.shape[0]
for i in range(rows_recm):
    # find centrality for each id_f in att_recm_f
    id_f = att_recm_f[i][0]
    if (id_f == od_centrality_msg[:, 0]).any():
        index_array = np.where(od_centrality_msg[:, 0] == id_f)
        index_int = index_array[0][0]
        recm_f_ev_centrality[i][1] = od_centrality_msg[index_int][1]
    else:
        recm_f_ev_centrality[i][1] = 0

np.savetxt("./centrality/new/recm_f_ev_centrality.csv", recm_f_ev_centrality, delimiter=',')
'''
'''
att_recf_m = pd.read_csv('att_recf_m.csv')    # shape = (10039, 35)   # f who receive a msg
att_recf_m = att_recf_m.values
recf_m_ev_centrality = np.zeros((10039, 2))
recf_m_ev_centrality[:, 0] = att_recf_m[:, 0]      # copy id_f in att_edgem_f to msgm_f_ev_centrality

rows_recf = att_recf_m.shape[0]
for i in range(rows_recf):
    # find centrality for each id_m in att_edgem_m
    id_m = att_recf_m[i][0]
    if (id_m == ev_centrality_msg[:, 0]).any():
        index_array = np.where(ev_centrality_msg[:, 0] == id_m)
        index_int = index_array[0][0]
        recf_m_ev_centrality[i][1] = ev_centrality_msg[index_int][1]
    else:
        recf_m_ev_centrality[i][1] = 0

np.savetxt("./centrality/new/recf_m_ev_centrality.csv", recf_m_ev_centrality, delimiter=',')
'''



