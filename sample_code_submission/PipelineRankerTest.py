# -*- coding: utf-8 -*-
import unittest
from random import shuffle
from PipelineRanker import PipelineRanker
from sklearn import pipeline, preprocessing as pp, tree
import numpy as np
import os

"""
Comme son nom l'indique, une classe testant les méthodes de PipelineRanker.
"""
class PipelineRankerTest(unittest.TestCase) :
    
    def test_sort(self) :
        self.assertEqual([], PipelineRanker.sort([]))
        
        # Le premier élément d'un couple n'est pas censé être utilisé,
        # Donc on peut mettre n'importe quoi.
        l = [(0,0), (0,1), (0,3), (0,4), (0,5)]
        l2 = list(reversed(l))
        self.assertEqual(l2, PipelineRanker.sort(l2))
        self.assertEqual(l2, PipelineRanker.sort(l))
        
        # On mélange aléatoirement l
        # (shuffle modifie son argument)
        shuffle(l)
        self.assertEqual(l2, PipelineRanker.sort(l))
        
    def test_display(self) :
        
        # Les pipelines
        p1 = pipeline.Pipeline([("default", pp.StandardScaler())])
        p2 = pipeline.Pipeline([("1", pp.MinMaxScaler()),
                                ("2", pp.StandardScaler())])
        
        # Test des conversions de Pipeline en string
        self.assertEqual(PipelineRanker.str_pipeline(p1), 
                         "(StandardScaler default)")
        self.assertEqual(PipelineRanker.str_pipeline(p2),
                         "(MinMaxScaler, StandardScaler)")
        
        
        regressor = tree.DecisionTreeRegressor()
        p = "./ranktest.txt"
        r = PipelineRanker(regressor, [p1, p2], p)
        
        # On supprime ranktest.txt (pour être sûr qu'il soit vierge)
        os.remove(p)
        
        # Chargement des données
        x = np.loadtxt("../public_data/air_train.data")
        y = np.loadtxt("../public_data/air_train.solution")
        
        # On sépare les données en deux
        middle = int(x.shape[0]/2)
        x_train = x[:middle, :]
        y_train = y[:middle]
        x_test = x[middle:, :]
        y_test = y[middle:]
        
        # Première écriture
        r.rank(x_train, y_train, x_test, y_test)
        f = open(p)
        nblines1 = len(f.readlines())
        self.assertNotEqual(nblines1, 0)
        f.close()
        
        # Deuxième écriture
        r.rank(x_train, y_train, x_test, y_test)
        f = open(p)
        nblines2 = len(f.readlines())
        f.close()
        
        # Normalement le nombre de lignes devrait avoir doublé
        # Sert à tester si le programme rajoute des lignes, 
        # et n'écrase pas ce qui était présent avant.
        self.assertEqual(nblines2, nblines1 * 2)

    
if __name__ == "__main__" :
    unittest.main()
