import json
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from nltk.util import ngrams
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot
import mmh3
import numpy
import pandas
import os
import sys
import random
import baker
from datetime import datetime
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split


# fix random seed for reproducibility
numpy.random.seed(3)

# This function will bring in two data sources and create the cross-validation index based on time
def read_data(filepath, n):
    n = int(n)
    #read data from every csv and write data in 2 lists
    print("Importing safe URLs")
    safe = pandas.read_csv(os.path.join(filepath, "safe.csv"))
    print("Importing danger URLs")
    danger = pandas.read_csv(os.path.join(filepath, "danger.csv"))

    if n > len(danger):
        n = len(danger)
    
    danger = danger.sample(n)
    safe = safe.sample(n)

    safe['label'] = 0
    danger['label'] = 1
    data = safe
    data = data.append(danger)
    data = data.sample(frac=1).reset_index(drop=True)

    y_label = [numpy.array(data['label'].values)]
    first_seen = data['first_seen'].values
    first_seen = [int((datetime.strptime(x, '%Y-%m-%d %H:%M:%S')- datetime(1970,1,1)).total_seconds()) for x in first_seen]

    #now do the time split
    p_cut = 70.0
    percentile = numpy.min((numpy.percentile(first_seen[y_label==0], p_cut), numpy.percentile(first_seen[y_label==1], p_cut)))

    train = []
    test = []
    for i, v in enumerate(first_seen):
        if v < percentile and y_label[0][i] >= 0:
            train.append(i)
        elif v >= percentile and y_label[0][i] >= 0:
            test.append(i)

    cv = [[numpy.array(train), numpy.array(test)]]

    print("Hashing feature vectors")
    X = numpy.array(vec_hash(data['url'].values))
    y = data['label'].values

    return(X, y, cv)

# This function will take the 3 n-gram of the url and hash it into a vector of length 1000
def vec_hash(data, vlen=1000):
    final = []
    for url in data:
        v = [0] * vlen
        new = list(ngrams(url, 3))
        for i in new:
            new_ = ''.join(i)
            idx = mmh3.hash(new_) % vlen
            v[idx] += 1
        final.append([numpy.array(v)])
    return final

# this will capture the training loss 
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
   
# This function is the architect of our model
def construct_model(model_type):
    print("Constructing model")
    if model_type=='deep':
        model = Sequential()
 
    	# hidden layers
        model.add(Dense(128, input_dim=1000))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(.15))
        
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.15))
        
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(.15))
        
        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(.15))

        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(.15))

        model.add(Dense(16))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(.15))

    	# final output layer
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                  optimizer='SGD',
                  metrics=['accuracy'])

    if model_type=='shallow':
    	model = Sequential()

        # One Layer
    	model.add(Dense(1,input_dim=1000, activation='sigmoid'))

    	model.compile(loss='binary_crossentropy',
                  optimizer='SGD',
                  metrics=['accuracy'])
    return model

def model_training(X_train, y_train, model):
    print("Beginning training model")
    loss = LossHistory()
    model.fit(X_train, y_train,
              epochs=80,
              batch_size=128, verbose=1, callbacks=[loss])
    return model, loss

@baker.command
def compare(filepath, n):
    # We want to split the data into training and testing
    X, y, cv = read_data(filepath, n)
    X_train = X[cv[0][0]]
    y_train = y[cv[0][0]]
    X_test = X[cv[0][1]]
    y_test = y[cv[0][1]]

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[2])

    results_path = os.path.join(filepath, "deepmodel_timesplit")
    results(X_train, y_train, X_test, y_test, cv, 'deep', results_path)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[2])

    results_path = os.path.join(filepath, "deepmodel_randsplit")
    results(X_train, y_train, X_test, y_test, 1, 'deep', results_path)
 
    plot_all(filepath)
    print("Done!")

# find the nearest value in an array to the given value
def find_nearest(array,value):
    return (numpy.abs(array-value)).argmin()

def results(X_train, y_train, X_test, y_test, cv, model_type, filepath):
    
    if not os.path.exists(filepath):
        os.makedirs(filepath)     

    # We want to train the model with train function
    model = construct_model(model_type)
    model, history = model_training(X_train, y_train, model)
    training_loss = history.losses

    # Now we want to evauluate the model by getting the FPR and TPR
    print("Predicting test dataset")
    preds =  model.predict(X_test, batch_size=64)

    # ensure format of preds is able to be handled by sklearn	
    if len(preds.shape)==1:
        preds_ = numpy.array([preds]).T
        
    if preds.shape[1]==1:
        p_neg = 1.0-preds
        preds_ = numpy.hstack((p_neg, preds))

    preds_ = preds_.astype(numpy.float) 

    #get roc curve using sklearn
    results = {}
    fpr, tpr, thresh = roc_curve(y_test, preds_[:,1], 1.0)
    curr_auc = auc(fpr, tpr)

    thresh1 = thresh[find_nearest(fpr, 1e-3)]
    thresh2 = thresh[find_nearest(fpr, 1e-4)]
   
    y_pred1 = []
    y_pred2 = []
    for k in preds: 
       if k >= thresh1:
          y_pred1.append(1)
       if k>= thresh2:
          y_pred2.append(1)
       if k< thresh1:
          y_pred1.append(0) 
       if k< thresh2:
          y_pred2.append(0)
     
    results['fpr_1e3'] = counts(y_test, y_pred1)
    results['fpr_1e4'] = counts(y_test, y_pred2)
    results['roc'] = numpy.column_stack((fpr, tpr, thresh)).tolist()
    results['auc'] = numpy.array([curr_auc]).tolist()
    results['training_loss'] = numpy.array([training_loss]).tolist()

    # plot the curve
    print("Plotting results")
    matplotlib.pyplot.plot(numpy.logspace(-10,0, 1000), numpy.logspace(-10,0, 1000), 'k--')
    matplotlib.pyplot.step(fpr, tpr, 'b-',  label='Model   (AUC = {:0.4f}), '.format(curr_auc))
    matplotlib.pyplot.xlim([0,1.0])
    matplotlib.pyplot.ylim([0, 1.0])
    matplotlib.pyplot.xlabel('False Positive Rate')
    matplotlib.pyplot.ylabel('True Positive Rate')
    if cv == 1:
        matplotlib.pyplot.title('ROC: Training=%d, Testing=%d, cv=%d' % (len(X_train), len(X_test), len(X_test)+len(X_train)))
    else:
        matplotlib.pyplot.title('ROC: Training=%d, Testing=%d, cv=%d' % (len(cv[0][0]), len(cv[0][1]), len(cv)))
    matplotlib.pyplot.legend(loc="lower right", prop={'size':8})
    matplotlib.pyplot.grid()
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.savefig(os.path.join(filepath, "ROC.png"), dpi=300)
    matplotlib.pyplot.xlim([1e-6, 1])
    matplotlib.pyplot.xscale('log')
    matplotlib.pyplot.savefig(os.path.join(filepath, "ROC_log.png"), dpi=300)
    matplotlib.pyplot.close()

    with open(os.path.join(filepath, 'output.json'), 'w') as f:
        json.dump(results, f, indent=4, separators=(',', ': '), sort_keys=True)

def counts(actual, preds):    
    tp = fp = tn = fn = 0
    for y, pred in zip(actual, preds):
        if pred == 1 and y == 1:
            tp += 1
        if pred == 0 and y == 1:
            fn += 1
        if pred == 1 and y == 0:
            fp += 1
        if pred == 0 and y == 0:
            tn += 1
    return numpy.column_stack((tp, fp, tn, fn)).tolist()

def plot_all(filepath):
    print("Plotting results")
    matplotlib.pyplot.plot(numpy.logspace(-10,0, 1000), numpy.logspace(-10,0, 1000), 'k--')
    matplotlib.pyplot.xlim([0,1.0])
    matplotlib.pyplot.ylim([0, 1.0])
    matplotlib.pyplot.xlabel('False Positive Rate')
    matplotlib.pyplot.ylabel('True Positive Rate')
    matplotlib.pyplot.title('ROC')
    line = ['-','--']
    color = ['blue', 'blue']
    models = ['deepmodel_timesplit','deepmodel_randsplit']
    labels = ['Deep Model w/Time Split', 'Deep Model w/70-30 Split']
    for idx, path in enumerate(models):
        with open(os.path.join(os.path.join(filepath, path),'output.json')) as f:
            data = json.load(f)
        fpr = [x[0] for x in data['roc']]
        tpr = [x[1] for x in data['roc']]
        matplotlib.pyplot.step(fpr, tpr, linestyle=line[idx], color=color[idx], label='Model {}  (AUC = {:0.4f}), '.format(labels[idx], data['auc'][0]))
        matplotlib.pyplot.legend(loc="lower right", prop={'size':8})
    matplotlib.pyplot.grid()
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.savefig(os.path.join(filepath, "all_ROC.png"), dpi=300)
    matplotlib.pyplot.xlim([1e-6, 1])
    matplotlib.pyplot.xscale('log')
    matplotlib.pyplot.savefig(os.path.join(filepath, "all_ROC_log.png"), dpi=300)
    matplotlib.pyplot.close()
 
def main(argv):
    baker.run()

if __name__ == '__main__':
    main(sys.argv[1:])
