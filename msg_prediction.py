import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import random
from matplotlib import pyplot
from sklearn.metrics import classification_report


# fix random seed
seed = 20210101
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.


profile_m = pd.read_csv('profile_m.csv').values
profile_f = pd.read_csv('profile_f.csv').values
profile_m[:, 1] = 0     # convert 'm' to 0
profile_f[:, 1] = 1     # convert 'f' to 1
'''
# att_diff_m_msg_f
edges_mSendMsg_fReceivedMsg = pd.read_csv('./march_30/edges_mSendMsg_fReceivedMsg.csv').values
print(edges_mSendMsg_fReceivedMsg.shape)
attribute_difference_m_msg_f = np.zeros((36026, 35))
for i in range(edges_mSendMsg_fReceivedMsg.shape[0]):
    m_id, f_id = edges_mSendMsg_fReceivedMsg[i][0], edges_mSendMsg_fReceivedMsg[i][1]
    index_m, index_f = np.where(profile_m[:, 0] == m_id), np.where(profile_f[:, 0] == f_id)
    attribute_difference_m_msg_f[i] = profile_m[index_m] - profile_f[index_f]
    if (i+1) % 10000 == 0:
        print('{} / {} are done...'.format(i+1, edges_mSendMsg_fReceivedMsg.shape[0]))
np.savetxt("./march_30/attribute_difference_m_msg_f.csv", attribute_difference_m_msg_f, delimiter=',', fmt='%d')


# att_diff_m_click_f
edges_mSendMsg_fClickedByTheseMButNotMsg = pd.read_csv('./march_30/edges_mSendMsg_fClickedByTheseMButNotMsg.csv').values
print(edges_mSendMsg_fClickedByTheseMButNotMsg.shape)

count = 0
attribute_difference_m_click_f = np.zeros((63540, 35))
for i in range(edges_mSendMsg_fClickedByTheseMButNotMsg.shape[0]):
    m_id, f_id = edges_mSendMsg_fClickedByTheseMButNotMsg[i][0], edges_mSendMsg_fClickedByTheseMButNotMsg[i][1]
    index_m, index_f = np.where(profile_m[:, 0] == m_id), np.where(profile_f[:, 0] == f_id)
    if len(index_m[0]) != 0 and len(index_f[0]) != 0:
        attribute_difference_m_click_f[count] = profile_m[index_m] - profile_f[index_f]
        count += 1
    if (i+1) % 10000 == 0:
        print('{} / {} are done...'.format(i+1, edges_mSendMsg_fClickedByTheseMButNotMsg.shape[0]))
print(count)
np.savetxt("./march_30/attribute_difference_m_click_f.csv", attribute_difference_m_click_f, delimiter=',', fmt='%d')


# att_diff_f_msg_m
edges_fSendMsg_mReceivedMsg = pd.read_csv('./march_30/edges_fSendMsg_mReceivedMsg.csv').values
print(edges_fSendMsg_mReceivedMsg.shape)
count = 0
attribute_difference_f_msg_m = np.zeros((9397, 35))
for i in range(edges_fSendMsg_mReceivedMsg.shape[0]):
    f_id, m_id = edges_fSendMsg_mReceivedMsg[i][0], edges_fSendMsg_mReceivedMsg[i][1]
    index_f, index_m = np.where(profile_f[:, 0] == f_id), np.where(profile_m[:, 0] == m_id)

    if len(index_f[0]) != 0 and len(index_m[0]) != 0:
        attribute_difference_f_msg_m[count] = profile_m[index_m] - profile_f[index_f]
        count += 1
    if (i+1) % 2000 == 0:
        print('{} / {} are done...'.format(i+1, edges_fSendMsg_mReceivedMsg.shape[0]))
print(count)
np.savetxt("./march_30/attribute_difference_f_msg_m.csv", attribute_difference_f_msg_m, delimiter=',', fmt='%d')


# att_diff_m_click_f
edges_fSendMsg_mClickedByTheseFButNotMsg = pd.read_csv('./march_30/edges_fSendMsg_mClickedByTheseFButNotMsg.csv').values
print(edges_fSendMsg_mClickedByTheseFButNotMsg.shape)
count = 0
attribute_difference_f_click_m = np.zeros((30044, 35))
for i in range(edges_fSendMsg_mClickedByTheseFButNotMsg.shape[0]):
    f_id, m_id = edges_fSendMsg_mClickedByTheseFButNotMsg[i][0], edges_fSendMsg_mClickedByTheseFButNotMsg[i][1]
    index_f, index_m = np.where(profile_f[:, 0] == f_id), np.where(profile_m[:, 0] == m_id)

    if len(index_f[0]) != 0 and len(index_m[0]) != 0:
        attribute_difference_f_click_m[count] = profile_m[index_m] - profile_f[index_f]
        count += 1
    if (i+1) % 5000 == 0:
        print('{} / {} are done...'.format(i+1, edges_fSendMsg_mClickedByTheseFButNotMsg.shape[0]))
print(count)
np.savetxt("./march_30/attribute_difference_f_click_m.csv", attribute_difference_f_click_m, delimiter=',', fmt='%d')
'''


# ===================== #
#   all 31 attributes   #
# ===================== #
# '''
attribute_difference_m_click_f = pd.read_csv('./march_30/attribute_difference_m_click_f.csv').values     # shape = (63540, 35)   # m -> msg -> f
attribute_difference_m_msg_f = pd.read_csv('./march_30/attribute_difference_m_msg_f.csv').values         # shape = (36026, 35)   # m -> click -> f, but not msg

sig_att_click_m = attribute_difference_m_click_f[:, 4:36]           # shape = (63540, 35)
sig_att_msg_m = attribute_difference_m_msg_f[:, 4:36]             # shape = (36026, 35)

y_0 = np.zeros((63540, ))              # y - msg
y_1 = np.ones((36026, ))             # y - click

x_data = np.concatenate((sig_att_click_m, sig_att_msg_m), axis=0)
y_data = np.concatenate((y_0, y_1), axis=0)

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)

# normalize
std = StandardScaler()
X_train = std.fit_transform(X_train)
X_test = std.fit_transform(X_test)

# LR
LR = LogisticRegression(C=1.0)
LR.fit(X_train, y_train)
print(LR.coef_)
print(LR.coef_.shape)
y_predict_LR = LR.predict(X_test)
print('LR Acc: ', accuracy_score(y_predict_LR, y_test))

# RF
RF = RandomForestClassifier()
RF.fit(X_train, y_train)
y_predict_RF = RF.predict(X_test)
print('RF Acc: ', accuracy_score(y_predict_RF, y_test))
# '''


# ===================== #
#    load centrality    #
# ===================== #
# find centrality for id in m_click_f, m_msg_f, f_click_m, f_msg_m
'''
centrality_ev = pd.read_csv('./march_30/centrality_ev.csv').values
centrality_od = pd.read_csv('./march_30/centrality_od.csv').values


edges_mSendMsg_fReceivedMsg = pd.read_csv('./march_30/edges_mSendMsg_fReceivedMsg.csv').values
centrality_ev_sum_m_msg_f = np.zeros((36026,))
centrality_od_sum_m_msg_f = np.zeros((36026,))
count_ev = 0
count_od = 0
for i in range(edges_mSendMsg_fReceivedMsg.shape[0]):
    m_id, f_id = edges_mSendMsg_fReceivedMsg[i][0], edges_mSendMsg_fReceivedMsg[i][1]
    # ev centrality
    index_m, index_f = np.where(centrality_ev[:, 0] == m_id), np.where(centrality_ev[:, 0] == f_id)
    if len(index_m[0]) != 0 and len(index_f[0]) != 0:
        centrality_ev_sum_m_msg_f[count_ev] = centrality_ev[index_m][0][1] + centrality_ev[index_f][0][1]     # sum = centrality_m + centrality_f
        count_ev += 1
    # od centrality
    index_m, index_f = np.where(centrality_od[:, 0] == m_id), np.where(centrality_od[:, 0] == f_id)
    if len(index_m[0]) != 0 and len(index_f[0]) != 0:
        centrality_od_sum_m_msg_f[count_od] = centrality_od[index_m][0][1] + centrality_od[index_f][0][1]     # sum = centrality_m + centrality_f
        count_od += 1
    if (i+1) % 5000 == 0:
        print('{} / {} are done...'.format(i+1, edges_mSendMsg_fReceivedMsg.shape[0]))
print('count_ev in m_msg_f: ', count_ev)
print('count_od in m_msg_f: ', count_od)
np.savetxt("./march_30/centrality_ev_sum_m_msg_f.csv", centrality_ev_sum_m_msg_f, delimiter=',', fmt='%.8f')
np.savetxt("./march_30/centrality_od_sum_m_msg_f.csv", centrality_od_sum_m_msg_f, delimiter=',', fmt='%.8f')


edges_mSendMsg_fClickedByTheseMButNotMsg = pd.read_csv('./march_30/edges_mSendMsg_fClickedByTheseMButNotMsg.csv').values
centrality_ev_sum_m_click_f = np.zeros((63540,))
centrality_od_sum_m_click_f = np.zeros((63540,))
count_ev = 0
count_od = 0
for i in range(edges_mSendMsg_fClickedByTheseMButNotMsg.shape[0]):
    m_id, f_id = edges_mSendMsg_fClickedByTheseMButNotMsg[i][0], edges_mSendMsg_fClickedByTheseMButNotMsg[i][1]
    # ev centrality
    index_m_profile, index_f_profile = np.where(profile_m[:, 0] == m_id), np.where(profile_f[:, 0] == f_id)
    index_m, index_f = np.where(centrality_ev[:, 0] == m_id), np.where(centrality_ev[:, 0] == f_id)
    if len(index_m_profile[0]) != 0 and len(index_f_profile[0]) != 0 and len(index_m[0]) != 0 and len(index_f[0]) != 0:
        centrality_ev_sum_m_click_f[count_ev] = centrality_ev[index_m][0][1] + centrality_ev[index_f][0][1]       # sum = centrality_m + centrality_f
        count_ev += 1
    # od centrality
    index_m_profile, index_f_profile = np.where(profile_m[:, 0] == m_id), np.where(profile_f[:, 0] == f_id)
    index_m, index_f = np.where(centrality_od[:, 0] == m_id), np.where(centrality_od[:, 0] == f_id)
    if len(index_m_profile[0]) != 0 and len(index_f_profile[0]) != 0 and len(index_m[0]) != 0 and len(index_f[0]) != 0:
        centrality_od_sum_m_click_f[count_od] = centrality_od[index_m][0][1] + centrality_od[index_f][0][1]       # sum = centrality_m + centrality_f
        count_od += 1
    if (i+1) % 5000 == 0:
        print('{} / {} are done...'.format(i+1, edges_mSendMsg_fClickedByTheseMButNotMsg.shape[0]))
print('count_ev in m_click_f: ', count_ev)
print('count_od in m_click_f: ', count_od)
np.savetxt("./march_30/centrality_ev_sum_m_click_f.csv", centrality_ev_sum_m_click_f, delimiter=',', fmt='%.8f')
np.savetxt("./march_30/centrality_od_sum_m_click_f.csv", centrality_od_sum_m_click_f, delimiter=',', fmt='%.8f')


edges_fSendMsg_mReceivedMsg = pd.read_csv('./march_30/edges_fSendMsg_mReceivedMsg.csv').values
centrality_ev_sum_f_msg_m = np.zeros((9397,))
centrality_od_sum_f_msg_m = np.zeros((9397,))
count_ev = 0
count_od = 0
for i in range(edges_fSendMsg_mReceivedMsg.shape[0]):
    f_id, m_id = edges_fSendMsg_mReceivedMsg[i][0], edges_fSendMsg_mReceivedMsg[i][1]
    # ev centrality
    index_f_profile, index_m_profile = np.where(profile_f[:, 0] == f_id), np.where(profile_m[:, 0] == m_id)
    index_f, index_m = np.where(centrality_ev[:, 0] == f_id), np.where(centrality_ev[:, 0] == m_id)
    if len(index_f_profile[0]) != 0 and len(index_m_profile[0]) != 0 and len(index_f[0]) != 0 and len(index_m[0]) != 0:
        centrality_ev_sum_f_msg_m[count_ev] = centrality_ev[index_m][0][1] + centrality_ev[index_f][0][1]     # sum = centrality_m + centrality_f
        count_ev += 1
    # od centrality
    index_f_profile, index_m_profile = np.where(profile_f[:, 0] == f_id), np.where(profile_m[:, 0] == m_id)
    index_f, index_m = np.where(centrality_od[:, 0] == f_id), np.where(centrality_od[:, 0] == m_id)     # sum = centrality_m + centrality_f
    if len(index_f_profile[0]) != 0 and len(index_m_profile[0]) != 0 and len(index_f[0]) != 0 and len(index_m[0]) != 0:
        centrality_od_sum_f_msg_m[count_od] = centrality_od[index_m][0][1] + centrality_od[index_f][0][1]
        count_od += 1
    if (i+1) % 2000 == 0:
        print('{} / {} are done...'.format(i+1, edges_fSendMsg_mReceivedMsg.shape[0]))
print('count_ev in f_msg_m: ', count_ev)
print('count_od in f_msg_m: ', count_od)
np.savetxt("./march_30/centrality_ev_sum_f_msg_m.csv", centrality_ev_sum_f_msg_m, delimiter=',', fmt='%.8f')
np.savetxt("./march_30/centrality_od_sum_f_msg_m.csv", centrality_od_sum_f_msg_m, delimiter=',', fmt='%.8f')


edges_fSendMsg_mClickedByTheseFButNotMsg = pd.read_csv('./march_30/edges_fSendMsg_mClickedByTheseFButNotMsg.csv').values
centrality_ev_sum_f_click_m = np.zeros((30044,))
centrality_od_sum_f_click_m = np.zeros((30044,))
count_ev = 0
count_od = 0
for i in range(edges_fSendMsg_mClickedByTheseFButNotMsg.shape[0]):
    f_id, m_id = edges_fSendMsg_mClickedByTheseFButNotMsg[i][0], edges_fSendMsg_mClickedByTheseFButNotMsg[i][1]
    # ev centrality
    index_f_profile, index_m_profile = np.where(profile_f[:, 0] == f_id), np.where(profile_m[:, 0] == m_id)
    index_f, index_m = np.where(centrality_ev[:, 0] == f_id), np.where(centrality_ev[:, 0] == m_id)
    if len(index_f_profile[0]) != 0 and len(index_m_profile[0]) != 0 and len(index_f[0]) != 0 and len(index_m[0]) != 0:
        centrality_ev_sum_f_click_m[count_ev] = centrality_ev[index_m][0][1] + centrality_ev[index_f][0][1]     # sum = centrality_m + centrality_f
        count_ev += 1
    # od centrality
    index_f_profile, index_m_profile = np.where(profile_f[:, 0] == f_id), np.where(profile_m[:, 0] == m_id)
    index_f, index_m = np.where(centrality_od[:, 0] == f_id), np.where(centrality_od[:, 0] == m_id)     # sum = centrality_m + centrality_f
    if len(index_f_profile[0]) != 0 and len(index_m_profile[0]) != 0 and len(index_f[0]) != 0 and len(index_m[0]) != 0:
        centrality_od_sum_f_click_m[count_od] = centrality_od[index_m][0][1] + centrality_od[index_f][0][1]
        count_od += 1
    if (i+1) % 5000 == 0:
        print('{} / {} are done...'.format(i+1, edges_fSendMsg_mClickedByTheseFButNotMsg.shape[0]))
print('count_ev in f_click_m: ', count_ev)
print('count_od in f_click_m: ', count_od)
np.savetxt("./march_30/centrality_ev_sum_f_click_m.csv", centrality_ev_sum_f_click_m, delimiter=',', fmt='%.8f')
np.savetxt("./march_30/centrality_od_sum_f_click_m.csv", centrality_od_sum_f_click_m, delimiter=',', fmt='%.8f')
'''

centrality_ev_sum_m_click_f = pd.read_csv('./march_30/centrality_ev_sum_m_click_f.csv').values
centrality_ev_sum_m_msg_f = pd.read_csv('./march_30/centrality_ev_sum_m_msg_f.csv').values

# attributes w/ centrality
attribute_difference_m_click_f_w_centrality = np.concatenate((attribute_difference_m_click_f, centrality_ev_sum_m_click_f), axis=1)
attribute_difference_m_msg_f_w_centrality = np.concatenate((attribute_difference_m_msg_f, centrality_ev_sum_m_msg_f), axis=1)

sig_att_click_m = attribute_difference_m_click_f_w_centrality[:, 4:37]           # shape = (63540, 35)
sig_att_msg_m = attribute_difference_m_msg_f_w_centrality[:, 4:37]             # shape = (36026, 35)

x_data = np.concatenate((sig_att_click_m, sig_att_msg_m), axis=0)
y_data = np.concatenate((y_0, y_1), axis=0)

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)

# normalize
std = StandardScaler()
X_train = std.fit_transform(X_train)
X_test = std.fit_transform(X_test)

# LR
LR = LogisticRegression(C=1.0)
LR.fit(X_train, y_train)
print(LR.coef_)
print(LR.coef_.shape)
y_predict_LR = LR.predict(X_test)
print('LR Acc w/ centrality: ', accuracy_score(y_predict_LR, y_test))

# RF
RF = RandomForestClassifier()
RF.fit(X_train, y_train)
y_predict_RF = RF.predict(X_test)
print('RF Acc w/ centrality: ', accuracy_score(y_predict_RF, y_test))


# ===================== #
#        Results        #
# ===================== #
# [[ 0.10789339 -0.05652122  0.          0.         -0.07093981 -0.25249133
#   -0.03351971  0.00269015  0.03404549 -0.05905169  0.04759491 -0.00077182
#   -0.06651811 -0.0779115  -0.12209046  0.01052942 -0.09553667 -0.08761355
#   -0.02766192  0.04948721 -0.01272224  0.01915997 -0.13263841  0.13573147
#    0.02197535  0.00652964 -0.05810419  0.01998717 -0.04964754 -0.21992787
#    0.12052428]]
# LR Acc:  0.6442919317040509
# RF Acc:  0.7150987612989622
#
#
# [[ 0.1602172  -0.04205434  0.          0.         -0.08597268 -0.23816773
#   -0.03385609 -0.00842272  0.03213781 -0.06096796  0.05658485 -0.00849713
#   -0.05930825 -0.07606516 -0.10755321  0.01174681 -0.09671902 -0.06712426
#   -0.00660345  0.05578838 -0.00789058  0.0495536  -0.04694881  0.02685693
#    0.01857578  0.02763659 -0.05548229  0.02822047 -0.05475121 -0.20197982
#    0.09923714 -0.2332473 ]]
# LR Acc w/ centrality:  0.647941078004687
# RF Acc w/ centrality:  0.716973552058922



