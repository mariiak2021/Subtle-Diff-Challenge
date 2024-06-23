from .bleu.bleu import Bleu
from .cider.cider import Cider
#from .meteor.meteor import Meteor
#from .rouge.rouge import Rouge
#from .spice.spice import Spice
#from .wmd.wmd import WMD
import numpy as np
import json

def eval2(gts,res):
    scorer = Bleu(n=4)
    s1, _ = scorer.compute_score(gts, res)
    
    scorer = Cider()
    s2, _ = scorer.compute_score(gts, res)

    #scorer = Meteor()
    #s3, _ = scorer.compute_score(gts, res)

    #scorer = Rouge()
    #s4, _ = scorer.compute_score(gts, res)


    #scorer = Spice()
    #s5, _ = scorer.compute_score(gts, res)
    


    # scorer = WMD()
    # s6, _ = scorer.compute_score(gts, res)

    return {'bleu':s1[3],'cider':s2}#,'meteor':s3,'spice':s5 }#,'wmd':s6}
    #return {'bleu':s1[3], 'cider': int(s2*1000)/0.1, 'meteor': int(s3*1000)/0.3, 'rouge': int(s4*1000)/0.3}#,'cider':s2,'meteor':s3,'rouge':s4}

def get_bleu(gts,res):
    scorer = Bleu(n=4)
    s, _ = scorer.compute_score(gts, res)
    return s[2]



def get_cider(gts, res):
    scorer = Cider()
    s, _ = scorer.compute_score(gts, res)
    return s


