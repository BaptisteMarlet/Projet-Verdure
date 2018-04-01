# -*- coding: utf-8 -*-

import datetime
import imp
import re
from time import time

"""
solution
    La cible de prediction
    
prediction
    Les prédictions données par un régresseur (ou classifieur),
    à partir de données dont solution représente les étiquettes.
    
Renvoie le score d'après r2-metric à partir de ces deux arguments.
Utilisé par défaut par PipelineRanker pour attribuer des scores aux pipelines.
"""
default_scoring = imp.load_source('metric', "../scoring_program/libscores.py").r2_regression

"""
Censé représenter la pipeline vide.
On passe par une constante car le constructeur de Pipeline n'accepte pas 
les listes vides.
"""
empty_pipeline = None

"""
Une petite classe très simple chargée de faire un classement d'un ensemble de
pipelines sur le meme ensemble de données.

Aucune pipeline ne doit finir avec un régresseur ou un classifieur, 
car on doit utiliser le même pour tous (voir 'regressor' dans le constructeur).
L'idée est de trouver empiriquement une suite de transformations
garantissant des résultats satisfaisants, en suivant une démarche 
d'algorithme génétique manuellement. Cette classe ne s'occuperait 
que du classement des individus dans une génération.

Additionnellement, il est possible de stocker les résultats dans un fichier.
Non seulement les résultats seront affichés dans la console, mais ils seront
également rajoutés tels quels à la fin dudit fichier.
"""
class PipelineRanker :
    
    """
    regressor
        Le régresseur (ou classifieur) qui sera utilisé après la transformation
        des données par chacune des pipelines.
        
    pipeline_list
        Une liste de pipelines qui seront ajoutés dès le début.
        Si None, aucune pipeline n'est là au début.
        
    history
        Le fichier à la fin duquel ajouter les résultats.
        Si None (valeur par défaut), rien ne sera fait.
        Si le fichier n'existe pas il sera créé.
        
    scoring
        La fonction qui sera utilisée pour attribuer des scores aux pipelines.
        Par défaut, default_scoring (r2-metric).
    """
    def __init__(self, regressor, pipeline_list = None, 
                 history = None, scoring = default_scoring) :
        self.regressor = regressor
        self.pipeline_list = []
        self.history = history
        self.scoring = scoring
        
        if pipeline_list != None :
            for i in pipeline_list :
                self.add(i)
            
    """
    Méthode ajoutant une pipeline à l'ensemble que doit évaluer cet objet.
    """
    def add(self, pipeline) :
        self.pipeline_list.append(pipeline)
        
        
    """
    Renvoie le score de regressor sur les données envoyées en paramètre.
    Le régresseur est entraînésur x_train ayant pour étiquette y_train.
    Le score est déterminé à partir des prédictions sur x_test 
    et de ses étiquettes y_test.
    """
    def score(self, x_train, y_train, x_test, y_test) :
        # On entraine le régresseur...
        self.regressor.fit(x_train, y_train)
        
        # On récupère le score sur l'ensemble de test
        prediction = self.regressor.predict(x_test)
        
        # Le régresseur semble se réinitialiser à chaque fois
        # qu'on appelle fit() dessus, donc rien besoin de faire.
        # (au moins pour GradientBoostingRegressor)
        
        return self.scoring(y_test, prediction)
    
    """
    scores
        Liste contenant des tuples de la forme (pipeline, score).
        Le premier élément n'importe pas.
        
    Renvoie scores triée selon le second élément de chaque tuple.
    """
    @staticmethod
    def sort(scores) :
        result = []
        
        for i in scores :
            # On cherche où placer i dans result
            length = len(result)
            j = length - 1
            while j >= 0 and i[1] >= result[j][1] :
                j -= 1
            
            if j == -1 :
                result.insert(0, i)
            else :
                result.insert(j+1, i)
                
        return result
    
    """ 
    Retourne une chaîne de caractère représentant la pipeline en paramètre.
    Chacune des étapes est représentée par le nom de sa classe,
    suivie du nom originellement donné dans le constructeur de Pipeline
    s'il ne correspond pas au motif [0-9]* (de façon à éviter de surcharger
    inutilement les résultats).
    """
    @staticmethod
    def str_pipeline(pipeline) :
        
        if pipeline == empty_pipeline :
            return "()"
        
        regexp = re.compile("^[0-9]*$")
        def str_step(step) :
            res = step[1].__class__.__name__
            if regexp.match(step[0]) == None :
                res += " " + step[0]
                
            return res
    
        nbsteps = len(pipeline.steps)
        
        # Premier élément
        result = "("
        if nbsteps > 0 :
            result += str_step(pipeline.steps[0])
            
        # Reste des éléments (on peut ajouter la virgule avant maintenant
        # qu'on sait qu'il y a au moins un élément avant)
        i = 1
        while i < nbsteps :
            result += ", " + str_step(pipeline.steps[i])
            i += 1
            
        return result + ")"
    
    """
    Méthode chargée d'afficher les résultats.
    Les ajoute également à la fin de history si la variable ne vaut pas None
    et si le fichier correspondant existe bien.
    """
    def display(self, result) :
            
        # Lignes du classement
        lines = []
        c = 0
        for i in result :
            lines.append(str(c) + ". " 
                         + PipelineRanker.str_pipeline(i[0]) 
                         + " : " + str(i[1]))
            c += 1
        
        # Affichage dans la console
        print "\nResults :"
        for i in lines :
            print i
        
        # Ajout dans le fichier (si besoin)
        if self.history != None :
            hist = open(self.history, 'a')
            
            now = str(datetime.datetime.now())
            regClass = self.regressor.__class__.__name__
            hist.write(now + ", regressor : " + regClass + "\n")
            
            for i in lines :
                hist.write(i + "\n")
            hist.write("\n--------------------------------------------------\n\n")
            
            hist.close()
        
        
    """
    Méthode évaluant et classant toutes les pipelines ajoutées jusqu'ici.
    Renvoie la liste des (pipeline, score) triée selon le score.
    
    x_train
        Ensemble des données sur lequel le régresseur sera entraîné 
        après transformation par chacune des pipelines.
        
    y_train
        Etiquettes pour x_train.
    
    x_test
        Ensemble des données sur lequel le régresseur sera testé
        (après transformation).
        
    y_test
        Etiquettes pour x_test.
    """
    def rank(self, x_train, y_train, x_test, y_test) :
        
        # On obtient un score pour chacune des pipelines
        scores = []
        for i in self.pipeline_list :
            
            start = time()
            
            if i != empty_pipeline :
                scores.append( (i, self.score(i.fit_transform(x_train, y_train),
                                              y_train,
                                              i.fit_transform(x_test, y_test),
                                              y_test)) )
            else :
                scores.append( (i, self.score(x_train, y_train,
                                              x_test, y_test)))
            
            elapsed = time() - start
            print PipelineRanker.str_pipeline(i) + " done : " + str(scores[-1][1])
            print "Time elapsed : " + str(elapsed)
            
        result = PipelineRanker.sort(scores)
        self.display(result)
            
        return result
    
    """
    Ouvre le fichier donné en paramètre et trie toutes les lignes 
    correspondantes à des lignes de score,
    de la forme : Rang. (suite de Transformer) : score
    Fais la moyenne pour chaque type de Pipeline, et peut aussi afficher le 
    maximum, minimum et nombre d'occurences.
    """
    @staticmethod
    def sort_file(path, detailed = False) :
        file = open(path)
        lines = file.readlines()
        file.close()
        regexp = re.compile("[0-9]+\. (\(.*\)) : (.*)")
        
        # Création d'un dictionnaire qui associe à chaque type de Pipeline
        # la liste de tous les scores qu'il a obtenu
        scoresdict = dict()
        for i in lines :
            res = regexp.match(i)
            if res != None :
                group1 = res.group(1)
                if not scoresdict.has_key(group1) :
                    scoresdict[group1] = []
                scoresdict[group1].append(float(res.group(2)))
        
        # Création d'une liste dont chaque élément a la forme
        # (pipeline, score moyen, score maximal, score minimal, nb occurences)
        scores = []
        for i in scoresdict.keys() :
            # Initialisation des variables
            sl = scoresdict[i]
            count = 0
            mean = 0
            maxS = sl[0] # On a au moins un élément donc c'est bon
            minS = sl[0]
            
            # Calcul des variables
            for j in sl :
                if j > maxS :
                    maxS = j
                elif j < minS :
                    minS = j
                
                mean += j
                count +=  1
            
            # Ajout dans scores
            scores.append((i, mean/count, maxS, minS, count))
                
        result = PipelineRanker.sort(scores)
        
        detailed_pattern = "{0}. {1} : {2}, max : {3}, min : {4}, occurences : {5}"
        simple_pattern = "{0}. {1} : {2}"
        count = 0
        for i in result :
            if detailed :
                print detailed_pattern.format(count, i[0], i[1], i[2], i[3], i[4])
            else :
                print simple_pattern.format(count, i[0], i[1])
            count += 1
            
        #return result