'''
Sample predictive model.
You must supply at least 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
- save: saves the model.
- load: reloads the model.
'''
import pickle
import numpy as np   # We recommend to use numpy arrays
from os.path import isfile

class model:
    def __init__(self):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation.
        '''
        self.debug = 0
        self.num_train_samples=0
        self.num_feat=1
        self.num_labels=1
        self.is_trained=False
        ###### Baseline models ######
        from sklearn.naive_bayes import GaussianNB
        from sklearn.linear_model import LinearRegression
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.ensemble import GradientBoostingRegressor
        # Comment and uncomment right lines in the following to choose the model
        #self.model = GaussianNB()
        #self.model = LinearRegression()
        #self.model = DecisionTreeRegressor()
        #self.model = RandomForestRegressor()
        #self.model = KNeighborsRegressor()
        #self.model = GradientBoostingRegressor()
        from PipelineFactory import pipeline_factory
        regressor = pipeline_factory(GradientBoostingRegressor(n_estimators=150,learning_rate=0.2,max_depth=5))
        self.model = regressor
        
        
        
    def fit(self, X, y):
        '''
        This function should train the model parameters.
        Here we do nothing in this example...
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
            y: Training label matrix of dim num_train_samples * num_labels.
        Both inputs are numpy arrays.
        For classification, labels could be either numbers 0, 1, ... c-1 for c classe
        or one-hot encoded vector of zeros, with a 1 at the kth position for class k.
        The AutoML format support on-hot encoding, which also works for multi-labels problems.
        Use data_converter.convert_to_num() to convert to the category number format.
        For regression, labels are continuous values.
        '''

        if self.debug:
        	self.num_train_samples = X.shape[0]
        	if X.ndim>1: self.num_feat = X.shape[1]
        	print("FIT: dim(X)= [{:d}, {:d}]").format(self.num_train_samples, self.num_feat)
        	num_train_samples = y.shape[0]
        	if y.ndim>1: self.num_labels = y.shape[1]
        	print("FIT: dim(y)= [{:d}, {:d}]").format(num_train_samples, self.num_labels)
        	if (self.num_train_samples != num_train_samples):
        		print("ARRGH: number of samples in X and y do not match!")
        self.model.fit(X, y)
        self.is_trained=True
    
    
    def predict(self, X):
        '''
        This function should provide predictions of labels on (test) data.

        Make sure that the predicted values are in the correct format for the scoring
        metric. For example, binary classification problems often expect predictions
        in the form of a discriminant value (if the area under the ROC curve it the metric)
        rather that predictions of the class labels themselves. For multi-class or multi-labels
        problems, class probabilities are often expected if the metric is cross-entropy.
        Scikit-learn also has a function predict-proba, we do not require it.
        The function predict eventually can return probabilities.
        '''
        if self.debug:
        	num_test_samples = X.shape[0]
        	if X.ndim>1: num_feat = X.shape[1]
        	print("PREDICT: dim(X)= [{:d}, {:d}]").format(num_test_samples, num_feat)
        	if (self.num_feat != num_feat):
        		print("ARRGH: number of features in X does not match training data!")
        	print("PREDICT: dim(y)= [{:d}, {:d}]").format(num_test_samples, self.num_labels)

        y = self.model.predict(X)
        return y

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "wb"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile, "rb") as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self

  
    def hyperParamOptimizers(self,X,y,tuned_parameters,metric_score='r2',n_fold=3):
        """
        methode permettant de rechercher les hyperparametres d'un modele pour obtenir le
        meilleur score pour celui-ci.
        Args:
            X: matrice de donnee d'entrainement
            y: matrice d'etiquette d'entrainement
            
            tuned_parameters: les hyperparametre du modele a tester passe sous la forme d'un
            dictionnaire python avec, en cle une chaine de caractere qui est le nom de l'hyperparametre du
            modele et en valeur, une liste d'hyperparametre a tester.
            
            metric_score: la metric utilise pour donner le score du modele. Par defaut:r2_metric
            n_fold: nombre de paquet pour la cross-validation
        """
        from sklearn.model_selection import GridSearchCV
        self.hyperParamOpti = GridSearchCV(self.model, tuned_parameters, cv=n_fold, scoring=metric_score)
        self.hyperParamOpti.fit(X,y)
        return self.hyperParamOpti.best_params_,self.hyperParamOpti.cv_results_,self.hyperParamOpti.best_score_


    def fit2(self,clf, X, y):
        clf.fit(X, y)
        
    def predict2(self,clf, X):
        y = clf.predict(X)
        return y
    

######## Main function ########
if __name__ == "__main__":
    # Find the files containing corresponding data
    # To find these files successfully:
    # you should execute this "model.py" script in the folder "sample_code_submission"
    # and the folder "public_data" should be in the SAME folder as the starting kit
    path_to_training_data = "../../public_data/air_train.data"
    path_to_training_label = "../../public_data/air_train.solution"
    path_to_testing_data = "../../public_data/air_test.data"
    path_to_validation_data = "../../public_data/air_valid.data"

    # Find the program computing R sqaured score
    path_to_metric = "../scoring_program/libscores.py"
    import imp
    r2_score = imp.load_source('metric', path_to_metric).r2_regression

    # use numpy to load data
    X_train = np.loadtxt(path_to_training_data)
    y_train = np.loadtxt(path_to_training_label)
    X_test = np.loadtxt(path_to_testing_data)
    X_valid = np.loadtxt(path_to_validation_data)

    print "_"*100
    print "Process begin"
    
    # TRAINING ERROR
    # generate an instance of our model (clf for classifier)
    clf = model()
    # train the model
    clf.fit(X_train, y_train)
    # to compute training error, first make predictions on training set
    y_hat_train = clf.predict(X_train)
    # then compare our prediction with true labels using the metric
    training_error = r2_score(y_train, y_hat_train)

    """
    Cross-validation du modele de la classe
    """
#
#    # CROSS-VALIDATION ERROR
#    from sklearn.model_selection import KFold
#    from numpy import zeros, mean
#    # 3-fold cross-validation
#    n_fold = 3
#    kf = KFold(n_splits=n_fold)
#    kf.get_n_splits(X_train)
#    i=0
#    scores = zeros(n_fold)
#    for train_index, test_index in kf.split(X_train):
#        Xtr, Xva = X_train[train_index], X_train[test_index]
#        Ytr, Yva = y_train[train_index], y_train[test_index]
#        M = model()
#        M.fit(Xtr, Ytr)
#        Yhat = M.predict(Xva)
#        scores[i] = r2_score(Yva, Yhat)
#        print ('Fold', i+1, 'example metric = ', scores[i])
#        i=i+1
#    cross_validation_error = mean(scores)
#    
#    # Print results
#    print("\nThe scores are: ")
#    print("Training: ", training_error)
#    print ('Cross-Validation: ', cross_validation_error)
    
#    # CROSS-VALIDATION ERROR REPEATED N TIMES
#    from sklearn.model_selection import RepeatedKFold
#    n_time = 4
#    rkf = RepeatedKFold(n_splits=n_fold,n_repeats=n_time)
#    rkf.get_n_splits(X_train)
#    i=0
#    scores = zeros(n_time*n_fold)
#    for train_index, test_index in rkf.split(X_train):
#        Xtr, Xva = X_train[train_index], X_train[test_index]
#        Ytr, Yva = y_train[train_index], y_train[test_index]
#        M = model()
#        M.fit(Xtr, Ytr)
#        Yhat = M.predict(Xva)
#        scores[i] = r2_score(Yva, Yhat)
#        print ('Fold', i+1, 'example metric = ', scores[i])
#        i=i+1
#    cross_validation_error = mean(scores)
#    
#    # Print results
#    print("\nThe scores are: ")
#    print("Training: ", training_error)
#    print ('Cross-Validation: ', cross_validation_error)
    
    """
    Selection de modele parmi plusieurs autres. Pour en rajouter, il suffit de 
    rajouter un couple (model(), "nom du modele") au n-uplet "multimodel_couple".
     Cela creer un graphique de synthese a la fin de la selection du modele.
    """
    
    # MODEL SELECTION
    import time as t
    import matplotlib.pyplot as plt
    from sklearn.model_selection import KFold
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Lasso
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.linear_model import HuberRegressor
#    multimodel_couple = (
#        (DecisionTreeRegressor(), "Decision Tree Regressor"),
#        (MLPRegressor(), "MultiLayerPerceptronRegressor"),
#        (GradientBoostingRegressor(), "GradientBoostingRegressor"),
#        (AdaBoostRegressor(), "AdaBoostRegressor"),
#        (Ridge(), "Ridge"),
#        (Lasso(), "Lasso"),
#        (HuberRegressor(), "HuberRegressor")
#        )
#    
#    results = []
#    M = model()
#    # 10-fold cross-validation
#    n = 10
#    kf = KFold(n_splits=n)
#    kf.get_n_splits(X_train)
#    scores = np.zeros(n)
#    temps_train = np.zeros(n)
#    temps_test = np.zeros(n)
#    for clf, name in multimodel_couple:
#        i=0
#        print('=' * 80)
#        print(name)
#        for train_index, test_index in kf.split(X_train):
#            Xtrain, Xvalid = X_train[train_index], X_train[test_index]
#            Ytrain, Yvalid = y_train[train_index], y_train[test_index]
#            
#            t0 = t.time()
#            M.fit2(clf, Xtrain, Ytrain)
#            train_time = t.time() - t0
#            temps_train[i] = train_time
#    
#            t0 = t.time()
#            pred = M.predict2(clf, Xvalid)
#            test_time = t.time() - t0
#            temps_test[i] = test_time
#            
#            scores[i] = r2_score(Yvalid, pred)
#            print 'Fold', i+1, ': moyenne = ', scores[i]
#            i=i+1
#            
#        print 'score moyen des k-fold moyenne = ', np.mean(scores)
#        print 'temps moyen d\'entrainement : ', np.mean(temps_train)
#        print 'temps moyen de test : ', np.mean(temps_test)
#        clf_descr = str(clf).split('(')[0]
#        results.append((clf_descr,np.mean(scores),np.mean(temps_train),np.mean(temps_test),clf))
#        
#    print("=" * 80)
#    
#    
#    #DISPLAY OF RESULTS
#    indices = np.arange(len(results))
#
#    results2 = [[x[j] for x in results] for j in range(5)]
#    
#    clf_names, score, training_time, test_time, clf = results2
#    
#    training_time = np.array(training_time) / np.max(training_time)
#    test_time = np.array(test_time) / np.max(test_time)
#    
#    plt.figure(figsize=(12, 8))
#    plt.title("Score")
#    plt.barh(indices, score, .2, label="score", color='navy')
#    plt.barh(indices + .3, training_time, .2, label="moyenne training time",
#             color='c')
#    plt.barh(indices + .6, test_time, .2, label="moyenne test time", color='darkorange')
#    plt.yticks(())
#    plt.legend(loc='best')
#    plt.subplots_adjust(left=.25)
#    plt.subplots_adjust(top=.95)
#    plt.subplots_adjust(bottom=.05)
#    
#    for i, c in zip(indices, clf_names):
#        plt.text(-.3, i, c)
#    
#    plt.show()
#    print("Model selection finish")
    
    """
    Optimisation des hyper-parametre du modele. Le modele choisi ici est le "GradientBoostingRegressor"
    ATTENTION: Seul le tuned_parameters1 fonctionne, les autres produisent un temps de calcul de
    plusieurs heures. il faut optimiser cette partie ou choisir uun autre modele qui s'execute
    plus rapidement.
    """
    # HYPER-PARAMETER OPTIMIZERS
    
    print "="*80
    print "HyperParameters optimizers begin"
    
    
    tuned_parameters = [{'loss': ['ls'], 'learning_rate': [0.1,0.2],
                     'n_estimators': [100],'criterion' : ['friedman_mse'],
                     'max_depth' : [3,4,5] },
                        {'loss': ['ls'], 'learning_rate': [0.1,0.2],
                     'n_estimators': [100],'criterion' : ['mse'],
                     'max_depth' : [3,4,5] },
                         {'loss':['lad'],'learning_rate': [0.3,0.4],
                     'n_estimators': [100],'criterion' : ['friedman_mse'],
                     'max_depth' : [3,4,5]},
                        {'loss':['lad'],'learning_rate': [0.3,0.4],
                     'n_estimators': [100],'criterion' : ['mse'],
                     'max_depth' : [3,4,5]},
                         {'loss':['huber'], 'learning_rate': [0.1,0.2],
                     'n_estimators': [100],'criterion' : ['mse'],
                     'max_depth' : [3,4,5], 'alpha' : [0.7,0.8,0.9]},
                          {'loss':['quantile'], 'learning_rate': [0.1],
                     'n_estimators': [100],'criterion' : ['friedman_mse', 'mse'],
                     'max_depth' : [3,4,5], 'alpha' : [0.7,0.8,0.9]}
                        ]
    
    
    M = model()
    for dict_param in tuned_parameters:
        print("="*80)
        t0 = t.time()
        best_param, cv_result,best_score = M.hyperParamOptimizers(X_train,y_train,dict_param,n_fold=10)
        print ""
        print "Execution time of hyperParamOpti : " ,t.time()-t0
        
        print("Best parameters set found on development set:")
        print ""
        print(best_param ,"and score is " ,best_score)
        print ""
        print ""
        means = cv_result['mean_test_score']
        stds = cv_result['std_test_score']
        for mean2, std, params in zip(means, stds, cv_result['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean2, std * 2, params))

    
    print ""
    print "Process finish"
    print "_"*100
    
    print("""
To compute these errors (scores) for other models, uncomment and comment the right lines in the "Baseline models" section of the class "model".
To obtain a validation score, you should make a code submission with this model.py script on CodaLab.""")
