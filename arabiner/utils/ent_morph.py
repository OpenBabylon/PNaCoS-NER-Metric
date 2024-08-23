import sys
import os
from collections import namedtuple
from arabiner.utils.helpers import load_checkpoint
from arabiner.utils.data import get_dataloaders, text2segments
from copy import deepcopy

class EntityMorphed :
    def __init__(self, text , labels) : 
        self.text = text
        self.labels = labels
        self.stem = ''
        self.lemma = ''
        self.pos = ''
    
    def set_morph(self, stem, lemma, pos ):
        self.stem = stem
        self.lemma = lemma
        self.pos = pos

    def __str__(self):
        return  self.text + " [" + str(self.labels) +"] " + " [" + str(self.pos) +"] " 
    
    def __repr__ (self) : 
        return self.__str__()
		
class WojoodNER : 
    def __init__(self, model_path) : 
        tagger, tag_vocab, train_config = load_checkpoint(model_path)
        self.tagger = tagger
        self.tag_vocab = tag_vocab
        self.train_config = train_config 
    
    def text2dataloader(self, text) : 
        dataset, token_vocab = text2segments(text)
        vocabs = namedtuple("Vocab", ["tags", "tokens"])
        vocab = vocabs(tokens=token_vocab, tags=self.tag_vocab)
        # From the datasets generate the dataloaders
        dataloader = get_dataloaders( (dataset,), vocab, self.train_config.data_config, 16, shuffle=(False,),)[0]
        return dataloader
    
    def infer (self, dataloader) : 
        # Perform inference on the text and get back the tagged segments
        segments = self.tagger.infer(dataloader)
        return segments


    def get_ner_labels(self, text = "وزير الخارجية الإيراني: أبلغنا واشنطن وبقیة الأطراف عبر الوسيط الأوروبي بملاحظاتنا وننتظر ردهم"):
        dataloader = self.text2dataloader(text)
        segments = self.infer(dataloader)
        text_list =[]
        entity_morphed_list=[]
        for segment in segments[0] :
            ent = [x['tag'] for x in  segment.pred_tag if (x['tag']!= "O")]
            entity_morphed = EntityMorphed(segment.text,ent)
            entity_morphed_list.append(entity_morphed)
            text_list.append(segment.text)
        return text_list , entity_morphed_list
		
def set_entity_morphed(entity_morphed , disambiguate):
    if (len(disambiguate.analyses) == 0) :
        return 0
    analysis = disambiguate.analyses[0].analysis
    if 'pos' in analysis : 
        entity_morphed.pos = analysis['pos']
    if 'lex' in analysis : 
        entity_morphed.lemma = analysis['lex']
    if 'stem' in analysis : 
        entity_morphed.stem = analysis['stem']
    return 0

def merge_BI_entities(ent_list) :
    merged = []
    prevEnt = None
    lookingFor = None
    for i in range(len(ent_list)) :
        ent = ent_list[i]
        b_labels = [b for b in ent.labels if b.startswith('B-')]
        if prevEnt == None :
            if len(b_labels) > 0 :
                prevEnt = ent
                lookingFor = "I" + b_labels[0][1:]
                merged.append(ent)
            else :
                merged.append(ent)
        else :
            if lookingFor in ent.labels :
                prevEnt.text += " " + ent.text
                prevEnt.stem += " " + ent.stem
                prevEnt.lemma += " " + ent.lemma
                #prevEnt.pos += " " + ent.pos
            elif len(b_labels) > 0 : 
                prevEnt = ent
                lookingFor = "I" + b_labels[0][1:]
                merged.append(ent)
            else : 
                prevEnt = None
                lookingFor = None
                merged.append(ent)
    return merged
	
def process_sentence(wojoodTagger,mle,sent):
    #print("==="+sent+"===")
    text_list , entity_morphed_list =  wojoodTagger.get_ner_labels(sent)
    #toknized  = simple_word_tokenize(sent)
    disambig = mle.disambiguate(text_list)
    assert len(entity_morphed_list) == len(disambig)
    #diacritized = [d.analyses[0].analysis['diac'] for d in disambig]
    pos_tags = [d.analyses[0].analysis['pos'] for d in disambig if len(d.analyses) > 0]# contains verb
    if not("verb" in pos_tags):
        return None
    [set_entity_morphed(entity_morphed,disambiguate) for entity_morphed , disambiguate in zip(entity_morphed_list,disambig)]
    #lemmas = [d.analyses[0].analysis['lex'] for d in disambig]
    #stems = [d.analyses[0].analysis['stem'] for d in disambig]
    #index_list =  [i  for i in range(len(labels)) ]
    merged = merge_BI_entities(entity_morphed_list)
    #print (merged)
    return merged
    
    
def print_entities(text_list, em_list) : 
  for t,em in zip(text_list,em_list) : 
      if len(em.labels) > 0 : 
         print (t,str(em.labels))
      else : 
        print (t, "entity not detected")
        
def check_in_range(s,e,a) : 
    if (s < 0) : 
        print ("from_file must be a non-negative integer") 
        return 
    if (e < 0) : 
        print ("to_file must be a non-negative integer") 
        return 
    if (s > e ) : 
        print("from_file must be between 0 and to_file inclusive")
        return
    if (e >= len(a) ) : 
        print("to_file must be between 0 and length of files: " + str(len(a)))


def read_file(fname) : 
  text = ""
  with open (fname, "r", encoding = 'utf-8') as f:
     text = f.read()
  return text
  
  
def add_edge(sentiment_analyzer, all_relations,sourceNode,destinationNode,relationNode, together=False):
  relation =  'co' if relationNode == None else relationNode.stem
  relation_sentiment = ""
  if  relationNode != None:
    relation_sentiment = sentiment_analyzer.predict(sourceNode.stem + " " +relationNode.stem + " "  +destinationNode.stem)
    #['positive', 'negative']
  
  if together:
    relation+= "_2_"
  all_relations.append([sourceNode.stem, destinationNode.stem,relation , sourceNode.labels[0], destinationNode.labels[0],relation_sentiment])

def make_all_relations(full_entity_morphed_list, sentiment_analyzer) : 
    all_relations = []
    for entity_morphed_list in full_entity_morphed_list:
        sourceNode = None
        destinationNode = None
        relationNode = None
        for element in entity_morphed_list:
            if(element.pos=='verb'):
                relationNode = deepcopy(element)
                if sourceNode != None and destinationNode != None : 
                   add_edge(sentiment_analyzer, all_relations,sourceNode,destinationNode,relationNode, together=True) 
                   sourceNode = destinationNode
                   destinationNode = None
            elif len(element.labels)==0 : 
                    pass 
            else : 
                if sourceNode == None : 
                      sourceNode = deepcopy(element)
                else :
                    destinationNode = deepcopy(element)
                    if relationNode != None : 
                        add_edge(sentiment_analyzer,all_relations,sourceNode,destinationNode,relationNode) 
                        sourceNode = destinationNode
                        relationNode = None
                        destinationNode = None
        if sourceNode != None and destinationNode != None : 
          add_edge(sentiment_analyzer,all_relations,sourceNode,destinationNode,relationNode, together=True) 
    return all_relations     

  
  
def process_all_sentences (wojoodTagger, mle, sent_split, number_of_sentences, out, progress) :
    full_entity_morphed_list=[]
    progress_increment = 0
    for sent in sent_split[:number_of_sentences] :
        out.update(progress(progress_increment, number_of_sentences))
        progress_increment+=1
        # function process_sentence calls named entity recognition and morphologicl
        # analyis on each sentence
        try:
            merged = process_sentence(wojoodTagger, mle, sent)
            if ( merged is not None):
                full_entity_morphed_list.append(merged)
        except IndexError as ie:
            print ("Error in sentence: " +  sent + " " + str(ei) )
    return full_entity_morphed_list