import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import  os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


Filepath = "2.1\\plot"
#绘图 
labels = ['A','B','C','D','E']
y_true = np.loadtxt(r'G:\Subject\1.1\{}\re_label.txt'.format(Filepath))
y_pred = np.loadtxt(r'G:\Subject\1.1\{}\pr_label.txt'.format(Filepath))

tick_marks = np.array(range(len(labels))) + 0.5
# print(tick_marks)
def plot_confusion_matrix(tp,cm,title='Confusion Matrix',cmap=plt.cm.Blues):
    plt.imshow(cm,interpolation='nearest',cmap=cmap)    #在特定窗口上显示图像
    plt.title(title)  
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations,labels,rotation=90)
    plt.yticks(xlocations,labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label \n Accuracy={}'.format(tp))

cm = confusion_matrix(y_true,y_pred)
totalt = sum(cm[i][i] for i in range(len(labels)))
totalp = totalt / cm.sum()
np.set_printoptions(precision=3) #输出精度
cm_normalized = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis] #归一化
print(cm_normalized)
plt.figure(figsize=(12,8),dpi=200)

ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm_normalized[y_val][x_val]
    if c > 0.01:
        plt.text(x_val, y_val, "%0.3f" % (c,), color='black', fontsize=10, va='center', ha='center')
# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)

plot_confusion_matrix(totalp,cm_normalized, title='Normalized confusion matrix')
# show confusion matrix
plt.savefig(r'G:\Subject\1.1\{}\confusion_matrix.eps'.format(Filepath), format='eps')
plt.show()