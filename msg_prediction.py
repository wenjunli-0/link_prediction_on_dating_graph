import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity, paired_distances
import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from matplotlib import pyplot
from sklearn.metrics import classification_report


# fix random seed
seed = 20210331
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.


# ================================================= #
#           Extract Profiles in msg / click         #
# ================================================= #
# profiles of m & f in m_msg_f
'''
edges_mSendMsg_fReceivedMsg = pd.read_csv('./march_30/edges_mSendMsg_fReceivedMsg.csv').values
print(edges_mSendMsg_fReceivedMsg.shape)
m_profile_m_msg_f = np.zeros((36026, 35))
f_profile_m_msg_f = np.zeros((36026, 35))
for i in range(edges_mSendMsg_fReceivedMsg.shape[0]):
    m_id, f_id = edges_mSendMsg_fReceivedMsg[i][0], edges_mSendMsg_fReceivedMsg[i][1]
    index_m, index_f = np.where(profile_m[:, 0] == m_id), np.where(profile_f[:, 0] == f_id)
    m_profile_m_msg_f[i] = profile_m[index_m]
    f_profile_m_msg_f[i] = profile_f[index_f]
    if (i+1) % 10000 == 0:
        print('{} / {} are done...'.format(i+1, edges_mSendMsg_fReceivedMsg.shape[0]))
np.savetxt("./march_30/m_profile_m_msg_f.csv", m_profile_m_msg_f, delimiter=',', fmt='%d')
np.savetxt("./march_30/f_profile_m_msg_f.csv", f_profile_m_msg_f, delimiter=',', fmt='%d')
'''

# profiles of m & f in m_click_f
'''
edges_mSendMsg_fClickedByTheseMButNotMsg = pd.read_csv('./march_30/edges_mSendMsg_fClickedByTheseMButNotMsg.csv').values
print(edges_mSendMsg_fClickedByTheseMButNotMsg.shape)
count = 0
m_profile_m_click_f = np.zeros((63540, 35))
f_profile_m_click_f = np.zeros((63540, 35))
for i in range(edges_mSendMsg_fClickedByTheseMButNotMsg.shape[0]):
    m_id, f_id = edges_mSendMsg_fClickedByTheseMButNotMsg[i][0], edges_mSendMsg_fClickedByTheseMButNotMsg[i][1]
    index_m, index_f = np.where(profile_m[:, 0] == m_id), np.where(profile_f[:, 0] == f_id)
    if len(index_m[0]) != 0 and len(index_f[0]) != 0:
        m_profile_m_click_f[count] = profile_m[index_m]
        f_profile_m_click_f[count] = profile_f[index_f]
        count += 1
    if (i+1) % 10000 == 0:
        print('{} / {} are done...'.format(i+1, edges_mSendMsg_fClickedByTheseMButNotMsg.shape[0]))
print(count)
np.savetxt("./march_30/m_profile_m_click_f.csv", m_profile_m_click_f, delimiter=',', fmt='%d')
np.savetxt("./march_30/f_profile_m_click_f.csv", f_profile_m_click_f, delimiter=',', fmt='%d')
'''

# profiles of m & f in f_msg_m
'''
edges_fSendMsg_mReceivedMsg = pd.read_csv('./march_30/edges_fSendMsg_mReceivedMsg.csv').values
print(edges_fSendMsg_mReceivedMsg.shape)
count = 0
f_profile_f_msg_m = np.zeros((9397, 35))
m_profile_f_msg_m = np.zeros((9397, 35))
for i in range(edges_fSendMsg_mReceivedMsg.shape[0]):
    f_id, m_id = edges_fSendMsg_mReceivedMsg[i][0], edges_fSendMsg_mReceivedMsg[i][1]
    index_f, index_m = np.where(profile_f[:, 0] == f_id), np.where(profile_m[:, 0] == m_id)
    if len(index_f[0]) != 0 and len(index_m[0]) != 0:
        f_profile_f_msg_m[count] = profile_f[index_f]
        m_profile_f_msg_m[count] = profile_m[index_m]
        count += 1
    if (i+1) % 2000 == 0:
        print('{} / {} are done...'.format(i+1, edges_fSendMsg_mReceivedMsg.shape[0]))
print(count)
np.savetxt("./march_30/f_profile_f_msg_m.csv", f_profile_f_msg_m, delimiter=',', fmt='%d')
np.savetxt("./march_30/m_profile_f_msg_m.csv", m_profile_f_msg_m, delimiter=',', fmt='%d')
'''

# profiles of m & f in f_click_m
'''
edges_fSendMsg_mClickedByTheseFButNotMsg = pd.read_csv('./march_30/edges_fSendMsg_mClickedByTheseFButNotMsg.csv').values
print(edges_fSendMsg_mClickedByTheseFButNotMsg.shape)
count = 0
f_profile_f_click_m = np.zeros((30044, 35))
m_profile_f_click_m = np.zeros((30044, 35))
for i in range(edges_fSendMsg_mClickedByTheseFButNotMsg.shape[0]):
    f_id, m_id = edges_fSendMsg_mClickedByTheseFButNotMsg[i][0], edges_fSendMsg_mClickedByTheseFButNotMsg[i][1]
    index_f, index_m = np.where(profile_f[:, 0] == f_id), np.where(profile_m[:, 0] == m_id)
    if len(index_f[0]) != 0 and len(index_m[0]) != 0:
        f_profile_f_click_m[count] = profile_f[index_f]
        m_profile_f_click_m[count] = profile_m[index_m]
        count += 1
    if (i+1) % 5000 == 0:
        print('{} / {} are done...'.format(i+1, edges_fSendMsg_mClickedByTheseFButNotMsg.shape[0]))
print(count)
np.savetxt("./march_30/f_profile_f_click_m.csv", f_profile_f_click_m, delimiter=',', fmt='%d')
np.savetxt("./march_30/m_profile_f_click_m.csv", m_profile_f_click_m, delimiter=',', fmt='%d')
'''


# ================================ #
#          Compute Difference             #
# ================================ #
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


# ================================ #
#          Compute Similarity             #
# ================================ #
# cosine similarity
'''
feature_list = np.arange(4, 35)
feature_importance = np.zeros((31, 2))
feature_importance[:, 0] = feature_list

m_profile_m_click_f = pd.read_csv('./march_30/m_profile_m_click_f.csv').values     # shape = (63540, 35)   # m -> msg -> f
f_profile_m_click_f = pd.read_csv('./march_30/f_profile_m_click_f.csv').values     # shape = (63540, 35)   # m -> msg -> f
m_profile_m_msg_f = pd.read_csv('./march_30/m_profile_m_msg_f.csv').values         # shape = (36026, 35)   # m -> click -> f, but not msg
f_profile_m_msg_f = pd.read_csv('./march_30/f_profile_m_msg_f.csv').values         # shape = (36026, 35)   # m -> click -> f, but not msg
print(m_profile_m_click_f.shape)
print(f_profile_m_click_f.shape)
print(m_profile_m_msg_f.shape)
print(f_profile_m_msg_f.shape)

click_similarity = cosine_similarity(m_profile_m_click_f, f_profile_m_click_f)
np.savetxt("similarity_m_click_f.csv", click_similarity, delimiter=',', fmt='%.5f')
print(click_similarity.shape)
'''


# ================================ #
#   prediction based on Difference   #
# ================================ #
feature_list = [4, 8, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22]
feature_list = np.array(feature_list)
# feature_list = np.arange(4, 35)
# feature_list = np.arange(0, 35)
# feature_importance = np.zeros((31, 2))
# feature_importance = np.zeros((35, 2))
feature_importance = np.zeros((12, 2))
feature_importance[:, 0] = feature_list

# male
'''
attribute_difference_m_click_f = pd.read_csv('./march_30/attribute_difference_m_click_f.csv').values     # shape = (63540, 35)   # m -> msg -> f
attribute_difference_m_msg_f = pd.read_csv('./march_30/attribute_difference_m_msg_f.csv').values         # shape = (36026, 35)   # m -> click -> f, but not msg

# sig_att_click_m = attribute_difference_m_click_f[:, 4:36]           # shape = (63540, 35)
# sig_att_msg_m = attribute_difference_m_msg_f[:, 4:36]             # shape = (36026, 35)
sig_att_click_m = attribute_difference_m_click_f[:, [4, 8, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22]]           # shape = (63540, 35)
sig_att_msg_m = attribute_difference_m_msg_f[:, [4, 8, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22]]             # shape = (36026, 35)

y_0 = np.zeros((63540, ))              # y_0 - click
y_1 = np.ones((36026, ))               # y_1 - msg

x_data = np.concatenate((sig_att_click_m, sig_att_msg_m), axis=0)
y_data = np.concatenate((y_0, y_1), axis=0)

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)

# normalize
std = StandardScaler()
X_train = std.fit_transform(X_train)
X_test = std.fit_transform(X_test)

# LR
# LR = LogisticRegression(C=1.0)
# LR.fit(X_train, y_train)
# print(LR.coef_)
# y_predict_LR = LR.predict(X_test)
# print('LR Acc: ', accuracy_score(y_predict_LR, y_test))

# RF
RF = RandomForestClassifier()
RF.fit(X_train, y_train)
y_predict_RF = RF.predict(X_test)
print('RF Acc: ', accuracy_score(y_predict_RF, y_test))
print('feature importance: ', np.round(RF.feature_importances_, 4))
print(confusion_matrix(y_test, y_predict_RF))
target_names = ['click-0', 'msg-1']
print(classification_report(y_test, y_predict_RF, target_names=target_names))

feature_importance[:, 1] = RF.feature_importances_
feature_importance = feature_importance[feature_importance[:, 1].argsort()[::-1]]
print(feature_importance)
'''

# female
# '''
attribute_difference_f_click_m = pd.read_csv('./march_30/attribute_difference_f_click_m.csv').values     # shape = (63540, 35)   # m -> msg -> f
attribute_difference_f_msg_m = pd.read_csv('./march_30/attribute_difference_f_msg_m.csv').values         # shape = (36026, 35)   # m -> click -> f, but not msg

# sig_att_click_m = attribute_difference_f_click_m[:, 4:36]           # shape = (63540, 35)
# sig_att_msg_m = attribute_difference_f_msg_m[:, 4:36]             # shape = (36026, 35)
sig_att_click_m = attribute_difference_f_click_m[:, [4, 8, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22]]           # shape = (63540, 35)
sig_att_msg_m = attribute_difference_f_msg_m[:, [4, 8, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22]]             # shape = (36026, 35)

y_0 = np.zeros((9397, ))              # y - msg
y_1 = np.ones((30044, ))             # y - click

x_data = np.concatenate((sig_att_click_m, sig_att_msg_m), axis=0)
y_data = np.concatenate((y_0, y_1), axis=0)

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)

# normalize
std = StandardScaler()
X_train = std.fit_transform(X_train)
X_test = std.fit_transform(X_test)

# LR
# LR = LogisticRegression(C=1.0)
# LR.fit(X_train, y_train)
# print(LR.coef_)
# y_predict_LR = LR.predict(X_test)
# print('LR Acc: ', accuracy_score(y_predict_LR, y_test))

# RF
RF = RandomForestClassifier()
RF.fit(X_train, y_train)
y_predict_RF = RF.predict(X_test)
print('RF Acc: ', accuracy_score(y_predict_RF, y_test))
print('feature importance: ', np.round(RF.feature_importances_, 4))
feature_importance[:, 1] = RF.feature_importances_
feature_importance = feature_importance[feature_importance[:, 1].argsort()[::-1]]
print(feature_importance)
# '''


# ===================== #
#    load centrality    #
# ===================== #
'''
# find centrality for id in m_click_f, m_msg_f, f_click_m, f_msg_m
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

# male
'''
centrality_ev_sum_m_click_f = pd.read_csv('./march_30/centrality_ev_sum_m_click_f.csv').values
centrality_ev_sum_m_msg_f = pd.read_csv('./march_30/centrality_ev_sum_m_msg_f.csv').values

# attributes w/ centrality
attribute_difference_m_click_f_w_centrality = np.concatenate((attribute_difference_m_click_f, centrality_ev_sum_m_click_f), axis=1)
attribute_difference_m_msg_f_w_centrality = np.concatenate((attribute_difference_m_msg_f, centrality_ev_sum_m_msg_f), axis=1)

# sig_att_click_m = attribute_difference_m_click_f_w_centrality[:, 4:36]           # shape = (63540, 35)
# sig_att_msg_m = attribute_difference_m_msg_f_w_centrality[:, 4:36]             # shape = (36026, 35)
sig_att_click_m = attribute_difference_m_click_f_w_centrality[:, [4, 8, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 35]]           # shape = (63540, 35)
sig_att_msg_m = attribute_difference_m_msg_f_w_centrality[:, [4, 8, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 35]]             # shape = (36026, 35)


x_data = np.concatenate((sig_att_click_m, sig_att_msg_m), axis=0)
y_data = np.concatenate((y_0, y_1), axis=0)

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)

# normalize
std = StandardScaler()
X_train = std.fit_transform(X_train)
X_test = std.fit_transform(X_test)

# RF
RF = RandomForestClassifier()
RF.fit(X_train, y_train)
y_predict_RF = RF.predict(X_test)
print('RF Acc w/ centrality: ', accuracy_score(y_predict_RF, y_test))
print('feature importance: ', np.round(RF.feature_importances_, 4))
print(confusion_matrix(y_test, y_predict_RF))
target_names = ['click-0', 'msg-1']
print(classification_report(y_test, y_predict_RF, target_names=target_names))
'''

# female
'''
centrality_ev_sum_f_click_m = pd.read_csv('./march_30/centrality_od_sum_f_click_m.csv').values
centrality_ev_sum_f_msg_m = pd.read_csv('./march_30/centrality_od_sum_f_msg_m.csv').values

# attributes w/ centrality
attribute_difference_m_click_f_w_centrality = np.concatenate((attribute_difference_f_click_m, centrality_ev_sum_f_click_m), axis=1)
attribute_difference_m_msg_f_w_centrality = np.concatenate((attribute_difference_f_msg_m, centrality_ev_sum_f_msg_m), axis=1)

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
# LR = LogisticRegression(C=1.0)
# LR.fit(X_train, y_train)
# print(LR.coef_)
# y_predict_LR = LR.predict(X_test)
# print('LR Acc w/ centrality: ', accuracy_score(y_predict_LR, y_test))

# RF
RF = RandomForestClassifier()
RF.fit(X_train, y_train)
y_predict_RF = RF.predict(X_test)
print('RF Acc w/ centrality: ', accuracy_score(y_predict_RF, y_test))
print('feature importance: ', np.round(RF.feature_importances_, 4))
'''


# ========================================== #
#    select attributes based on importance   #
# ========================================== #
'''
sig_att_click_m = attribute_difference_m_click_f[:, [9, 5, 20]]           # shape = (63540, 35)
sig_att_msg_m = attribute_difference_m_msg_f[:, [9, 5, 20]]             # shape = (36026, 35)

x_data = np.concatenate((sig_att_click_m, sig_att_msg_m), axis=0)
y_data = np.concatenate((y_0, y_1), axis=0)

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)

# normalize
std = StandardScaler()
X_train = std.fit_transform(X_train)
X_test = std.fit_transform(X_test)

# LR
# LR = LogisticRegression(C=1.0)
# LR.fit(X_train, y_train)
# print(LR.coef_)
# y_predict_LR = LR.predict(X_test)
# print('LR Acc w/ selected Attributes: ', accuracy_score(y_predict_LR, y_test))

# RF
RF = RandomForestClassifier()
RF.fit(X_train, y_train)
y_predict_RF = RF.predict(X_test)
print('RF Acc w/ selected Attributes: ', accuracy_score(y_predict_RF, y_test))
'''


# ========================================== #
#    select attributes based on common sense   #
# ========================================== #
'''
attribute_difference_m_click_f = pd.read_csv('./march_30/attribute_difference_m_click_f.csv').values     # shape = (63540, 35)   # m -> msg -> f
attribute_difference_m_msg_f = pd.read_csv('./march_30/attribute_difference_m_msg_f.csv').values         # shape = (36026, 35)   # m -> click -> f, but not msg

sig_att_click_m = attribute_difference_m_click_f[:, [4, 8, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23]]           # shape = (63540, 35)
sig_att_msg_m = attribute_difference_m_msg_f[:, [4, 8, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23]]             # shape = (36026, 35)

y_0 = np.zeros((63540, ))              # y_0 - click
y_1 = np.ones((36026, ))               # y_1 - msg

x_data = np.concatenate((sig_att_click_m, sig_att_msg_m), axis=0)
y_data = np.concatenate((y_0, y_1), axis=0)

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)

# normalize
std = StandardScaler()
X_train = std.fit_transform(X_train)
X_test = std.fit_transform(X_test)

# RF
RF = RandomForestClassifier()
RF.fit(X_train, y_train)
y_predict_RF = RF.predict(X_test)
print('RF Acc: ', accuracy_score(y_predict_RF, y_test))
print('feature importance: ', np.round(RF.feature_importances_, 4))

# print(confusion_matrix(y_test, y_predict_RF))
# target_names = ['msg-0', 'click-1']
# print(classification_report(y_test, y_predict_RF, target_names=target_names))

# feature_importance[:, 1] = RF.feature_importances_
# feature_importance = feature_importance[feature_importance[:, 1].argsort()[::-1]]
# print(feature_importance)
'''


# ================================ #
#   similarity.. of  31 attributes   #
# ================================ #




