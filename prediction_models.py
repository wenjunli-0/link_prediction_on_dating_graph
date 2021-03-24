import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import random
from sklearn.metrics import classification_report


# fix random seed
seed = 20210101
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.


# LR model w/o centrality
# '''
att_edgem_m = pd.read_csv('att_edgem_m.csv')    # shape = (38624, 35)   # m who send a msg
att_edgem_f = pd.read_csv('att_edgem_f.csv')    # shape = (38624, 35)   # f who receive a msg
att_edgef_f = pd.read_csv('att_edgef_f.csv')    # shape = (10039, 35)   # f who send a msg
att_edgef_m = pd.read_csv('att_edgef_m.csv')    # shape = (10039, 35)   # m who receive a msg
att_recm_f = pd.read_csv('att_recm_f.csv')      # shape = (38624, 35)   # f who do not receive a msg
att_recf_m = pd.read_csv('att_recf_m.csv')      # shape = (10039, 35)   # m who do not receive a msg


att_edgem_m = att_edgem_m.values
att_edgem_f = att_edgem_f.values
att_edgef_f = att_edgef_f.values
att_edgef_m = att_edgef_m.values
att_recm_f = att_recm_f.values
att_recf_m = att_recf_m.values

# pre-processing
msgm_diff = att_edgem_m - att_edgem_f      # m -> msg -> f
recm_diff = att_edgem_m - att_recm_f       # m -> rec -> f

sig_att_msgmf = msgm_diff[:, 4:35]       # shape = (38624, 35)
sig_att_recmf = recm_diff[:, 4:35]       # shape = (38624, 35)

y_1 = np.ones((38624, ))              # y - msg
y_0 = np.zeros((38624, ))             # y - rec

x_data = np.concatenate((sig_att_msgmf, sig_att_recmf), axis=0)
y_data = np.concatenate((y_1, y_0), axis=0)

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)

# normalize
std = StandardScaler()
X_train = std.fit_transform(X_train)
X_test = std.fit_transform(X_test)

# fit model
lr = LogisticRegression(C=1.0)
lr.fit(X_train, y_train)

# results
print(lr.coef_)
y_predict = lr.predict(X_test)
print('LR Acc: ', lr.score(X_test, y_test))
# print('recall: ', classification_report(y_test, y_predict, labels=[2, 4], target_names=['edge-1', 'edge-0']))
# '''


# RF model w/o centrality
# '''
RF = RandomForestClassifier()
RF.fit(X_train, y_train)
y_predict = RF.predict(X_test)
print('RF Acc: ', accuracy_score(y_predict, y_test))


# =============== #
# load centrality #
# =============== #
msgm_m_ev_centrality = pd.read_csv('/Users/wenjun/SMU/IS709Network/project/project_codes/centrality/new/msgm_m_ev_centrality.csv')
msgm_f_ev_centrality = pd.read_csv('/Users/wenjun/SMU/IS709Network/project/project_codes/centrality/new/msgm_f_ev_centrality.csv')

msgm_m_ev_centrality = msgm_m_ev_centrality.values
msgm_f_ev_centrality = msgm_f_ev_centrality.values


# count none-zero centrality nodes
m = 0
for i in range(msgm_m_ev_centrality.shape[0]):
    if msgm_m_ev_centrality[i][1] != 0:
        m += 1
print('there are {} none-zero centrality in msgm_m'.format(m))
f = 0
for i in range(msgm_f_ev_centrality.shape[0]):
    if msgm_f_ev_centrality[i][1] != 0:
        f += 1
print('there are {} none-zero centrality in msgm_f'.format(f))


# load centrality as attribute-36
att_edgem_m_centrality = np.zeros((38624, 36))
att_edgem_m_centrality[:, 0:35] = att_edgem_m
att_edgem_m_centrality[:, 35] = msgm_m_ev_centrality[:, 1]

att_edgem_f_centrality = np.zeros((38624, 36))
att_edgem_f_centrality[:, 0:35] = att_edgem_f
att_edgem_f_centrality[:, 35] = msgm_f_ev_centrality[:, 1]

# diff: 0-35; sum: 36
msgm_diff = np.zeros((38624, 36))
recm_diff = np.zeros((38624, 36))
msgm_diff[:, 0:35] = att_edgem_m - att_edgem_f      # m -> msg -> f
recm_diff[:, 0:35] = att_edgem_m - att_recm_f       # m -> rec -> f

msgm_diff[:, 35] = att_edgem_m_centrality[:, 35] + att_edgem_f_centrality[:, 35]
recm_diff[:, 35] = att_edgem_m_centrality[:, 35] + att_edgem_f_centrality[:, 35]


sig_att_msgmf = msgm_diff[:, 4:36]       # shape = (38624, 36)
sig_att_recmf = recm_diff[:, 4:36]       # shape = (38624, 36)

y_1 = np.ones((38624, ))              # y - msg
y_0 = np.zeros((38624, ))             # y - rec

x_data = np.concatenate((sig_att_msgmf, sig_att_recmf), axis=0)
y_data = np.concatenate((y_1, y_0), axis=0)

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)

std = StandardScaler()
X_train = std.fit_transform(X_train)
X_test = std.transform(X_test)


# ============================ #
# LR, RF Models w/ centrality  #
# ============================ #
lr = LogisticRegression(C=1.0)
lr.fit(X_train, y_train)

print(lr.coef_)
y_predict = lr.predict(X_test)
print('LR acc: ', lr.score(X_test, y_test))


# RF model
RF = RandomForestClassifier()
RF.fit(X_train, y_train)
y_predict = RF.predict(X_test)
print('RF Acc: ', accuracy_score(y_predict, y_test))


