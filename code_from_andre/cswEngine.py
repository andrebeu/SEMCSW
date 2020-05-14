import re
import json
from numpy import random
import itertools


QUESTION_PR = 1.0
FILLER_QUESTION_PR = 0.25
NUM_BURNIN = 0
CONDITIONED = True
# curriculum
CURR = 'block'
BLOCK_LEN = 1 # block
PR_SHIFT = 0.1 # random
## inserted block experiment
BLOCK_POSITION = 2

# see experiment for more options
SCH_FPATH="csw1000_multiname_loctransition.sch"
RFC_FPATH='csw_multiname_loctransition.rfc'

PARAMS = [CURR,QUESTION_PR,FILLER_QUESTION_PR,CONDITIONED,NUM_BURNIN,SCH_FPATH,RFC_FPATH]



""" NOTES

unconditioned graphs currently implemented as:
  randomly sampling a condition at the time of graph creation
  this will not work with two or more conditioning factors
  this was done instead of just picking first condition in every edges 
    because i wanted to randomize which condition got picked between subjects
    however, this might be unnecessary because graph is fully balanced 
    i.e. every node is visited the same proportions


currently there is a fine distinction between 'edge' and 'edges'.
  edges = {cond:edge,cond:edge}
  edge = {tonode1:pr,tonode2:pr}

careful with 'cond' as it might refer to the conditioning factor for a given
  transition but it might also refer to the condition satisfied in a given
  block in blocked learning

pass conditioning as arg to graph constructor



"""


### helper methods


def read_json(path):
  """ load schema files"""
  with open(path) as f:
    schema_info_L = json.load(f)
  return schema_info_L


def get_all_edge_conditions(schfile_dict):
  """ returns a set of all conditioning factors in a schema file
  ['loc.latent.true','loc.latent.false',...]
  currently used in construction of unconditioned graphs

  """
  all_conditions = set()
  for node_info in schfile_dict.values():
    all_conditions.update(node_info['edge'].keys())
  return list(all_conditions)


### RFC and ONT


class RFC(dict):

  def __init__(self,rfc_dict):
    """ this class increases the functionality of RFCs
    rfc.dict is {role:filler,...} 
    filler is dict {property:value,...}
    e.g. {subject:{name:bill,...},...}
    """
    self._dict = rfc_dict
    self.id = self.get_id()
  
  def __getitem__(self,key):
    return self._dict[key]

  def __str__(self):
    return str(self._dict)

  __repr__ = __str__ 


  def get_id(self):
    """ 
    returns a string that identifies the statistics of this RFC
    currently subject and location change statistics of graphs,
    so RFC id is "subject-subj_location-loc"
    used to keep track of rfcs in .js mturk script
    """
    rfc_id = "subject-%s_location-%s" %(
      self['subject']['name'],self['location']['name'].split()[0])
    return rfc_id

  def assert_cond_met(self,cond):
    """ 
    given a cond "role.property.value" e.g. "subject.latent.true"
    returns bool indicating if condition is met in this rfc
    currently only checked with bool values 
    extend this 
    """
    if cond=='uncond':
      return True
    role,prop,value = cond.split('.')
    # NB value in RFC has bool type, whereas in cond has string type
    if value.lower() == 'true': 
      value = True
    elif value.lower() == 'false': 
      value = False
    cond_met = self[role][prop] == value
    return cond_met

  def assert_roles_differ(self,other_rfc,role):
    """
    given a role (e.g. subject), return True if self and other rfc have different values for that role
    """
    other_role = other_rfc[role]
    self_role = self[role]
    return other_role != self_role



class Ontology(list):

  def __init__(self,full=True):
    if full:
      self._list = self.get_full_rfcbag()
      self._name = 'full'
  
  def __str__(self):
    return self._name + 'OntObj'

  __repr__ = __str__

  def __getitem__(self,key):
    """ implements self[key] """
    return self._list[key]

  def get_full_rfcbag(self,rfc_fpath=RFC_FPATH):
    """ returns a list of rfcs:
    using itertools.product makes an rfcbag with 
    every possible combination of fillers for available roles"""
    rfcfile_dict = read_json(rfc_fpath)
    roles = [*rfcfile_dict.keys()]
    filler_product = itertools.product(*rfcfile_dict.values())
    rfcbag = []
    for fillers in filler_product:
      rfc = RFC({x:y for x,y in zip(roles,fillers)})
      rfcbag.append(rfc)
    return rfcbag

  def sample_rfc(self,cond=None):
    """ samples RFC from ontology 
    if condition 'role.property.value' is specified, samples from subset of ontology
      that satisfies condition
    if no condition is specified, samples from full ontology
      required because np.random.choice(ont) does not work 
      due to custom class implementation of list
    """
    # initial sample from ontology
    random_rfc = random.choice(self._list) 
    if cond:
      while not random_rfc.assert_cond_met(cond):
        random_rfc = random.choice(self._list)
    return random_rfc
    


### GRAPH




class Graph():


  def __init__(self,conditioned=CONDITIONED, schfile_path=SCH_FPATH):
    self.sch_fpath = schfile_path
    self.schfile_dict = read_json(schfile_path)
    self.conditioned = conditioned
    if not self.conditioned:
      self.random_condition = random.choice(get_all_edge_conditions(self.schfile_dict))
    # initialize_nodes
    self.node_dict = {} # {node.name:nodeObj}
    self.initialize_nodes()
    # initialize edges
    self.edge_dict = {} # {frnode.name:{tonode.name:pr}}
    self.initialize_edges()

  def __getitem__(self,key):
    return self.node_dict[key]


  def initialize_nodes(self):
    """ initialize graph with {nodename:nodeobj}
    """
    for node_name,node_info in self.schfile_dict.items():
      node_sent = node_info['sent']
      self.node_dict[node_name] = Node(name=node_name, sent=node_sent)
    return None

  def initialize_edges(self):
    """ edges are placed in edge_dict
    """
    assert len(self.node_dict), 'node_dict must be initialized first'
    for node in self.node_dict.values():
      if node.name == 'END': continue
      self.edge_dict[node.name] = self.assemble_edge(node)
    return None

  def assemble_edge(self,node):
    """ edges is a dict of {cond:edge}, an edge is a dict of {tonode:pr}
    currently just changes keys of inner dict from string "tonode1" to node object
    given edge info {"cond": {"tonode1": 0.8, "tonode2": 0.2},} from sch file
    return an edge object. e.g. {"cond1": {tonode1:0.3,tonode2:0.7}, "cond2": ...} 
    """
    edge = {}
    edge_info = self.schfile_dict[node.name]['edge'] # {"cond": {"tonode1": 0.8, "tonode2": 0.2},}
    # conditioned
    if self.conditioned:
      for cond,cond_dist in edge_info.items():
        edge[cond] = {}
        for tonode_name,probability in cond_dist.items():
          tonode = self.node_dict[tonode_name]
          edge[cond][tonode] = probability
    # unconditioned
    else:
      edge['uncond'] = {}
      # use this for uncond when graph has multiple conditioning factors
      # cond_dist = list(edge_info.values())[0] 
      cond_dist = edge_info[self.random_condition]
      for tonode_name,probability in cond_dist.items():
        tonode = self.node_dict[tonode_name]
        edge['uncond'][tonode] = probability
    return edge

  def get_edge(self,frnode,rfc):
    """ 
    first checks if uncond, returns edge
    if conditioned, given an rfc, which establishes which conditions are met,
    returns an edge, i.e. a conditional distribution {tonode1:pr,tonode2:pr,...}
    """
    edges = self.edge_dict[frnode.name]
    # check if uncond
    if self.conditioned:
      # look for first condition met
      for cond in edges.keys():
        if rfc.assert_cond_met(cond):
          edge = edges[cond] 
          break
      else:
        assert False,'Failed to find edge. No conditions in edge met.'
    # if unconditioned, ignore passed rfc
    else:
      assert len(edges) == 1, 'assuming unconditioned but found more than one condition'
      edge = edges['uncond'] 
    return edge

  def sample_tonode(self,frnode,rfc):
    """  
    """
    # check what conditioning factor is met by given rfc
    # e.g. "subject.latent.false" 
    # draw next tonode from corresponding edge
    cond_dist = self.get_edge(frnode,rfc) # {tonode1:prob,tonode2:prob,...}
    next_tonode_L = list(cond_dist.keys())
    next_tonode_pr = list(cond_dist.values())
    next_tonode = random.choice(next_tonode_L,p=next_tonode_pr)
    return next_tonode


### NODE


class Node():

  def __init__(self,name,sent):
    self.name = name 
    self.sent = sent
    self.type = "story_node"
    self.edges = None # {cond:{tonode:pr}}, assigned at graph consutrction

  def __str__(self):
    return "NodeObj.%s" % self.name

  __repr__ = __str__


  def get_filled_sent(self,rfc):
    """ fills a node's sent with given rfc
        returns the resulting string
    """
    filled_sent = self.sent
    role_property_L = re.findall("\[.*?\]",filled_sent)
    for rp in role_property_L:
      r,p = rp[1:-1].split('.')
      filled_sent = re.sub("\\"+rp,rfc[r][p],filled_sent)
    return filled_sent

  def assert_sents_differ(self,rfc1,rfc2):
    """ 
    if the node's output sentences have different 
    sentences under the two rfcs, return True
    """
    return self.get_filled_sent(rfc1) != self.get_filled_sent(rfc2)



### QUESTIONS


class Question():
  """ qinfo contains frnode,tonode,rfc 
      and depending on qtype false_rfc or false_tonode 
  """

  def __init__(self,**qinfo):
    self.frnode = qinfo['frnode']
    self.true_tonode = qinfo['true_tonode']
    self.true_rfc = qinfo['true_rfc']
    self.type = "question_node"

  def __str__(self):
    """ qinfo in common to both questions
    """
    qinfo_str = """%s_%s-%s_%s-%s""" % (
      self.type,self.frnode.name,self.true_tonode.name,
      self.frnode.name,self.false_tonode.name)
    return qinfo_str
  
  __repr__ = __str__

  def get_filled_sent(self):
    """ fills in the true and false tonodes with the appropriate rfcs"""
    true_next = self.true_tonode.get_filled_sent(self.true_rfc)
    false_next = self.false_tonode.get_filled_sent(self.false_rfc)
    return {'true_next':true_next,'false_next':false_next}

class FillerQ(Question):
  """ has false rfc 
  """
  def __init__(self,**qinfo):
    """ qinfo: {frnode,tonode,rfc,false_rfc}
    """
    Question.__init__(self,**qinfo)
    self.type = 'fillerQ'
    self.false_rfc = qinfo['false_rfc']
    self.false_tonode = self.true_tonode

class TransitionQ(Question):
  """ has false tonode 
  """
  def __init__(self,**qinfo):
    """ qinfo: {frnode,tonode,rfc,false_tonode} 
    """
    Question.__init__(self,**qinfo)
    self.type = 'transQ'
    self.false_tonode = qinfo['false_tonode']
    self.false_rfc = self.true_rfc



### EXPERIMENT


class Exp():
  def __init__(self,graph,ont):
    self.graph = graph # {nodename:node}
    self.ont = ont
    self.askingQ = False # used for burnin period
    self.rand_curr = False

  ## main

  def gen_path(self,rfc_cond=None):
    """ assembles a path 
    when rfc_cond==None
    """
    # sample filling rfc from ontology 
    rfc = self.ont.sample_rfc(rfc_cond)
    # start at Begin node
    frnode = self.graph.node_dict['BEGIN']
    path = [frnode]
    while frnode.name != "END":
      # draw next tonode
      next_tonode = self.graph.sample_tonode(frnode,rfc) 
      if next_tonode.name == "END": break
      ## question asking
      if self.assert_ask_question(frnode):
        question = self.get_question(frnode,next_tonode,rfc) # returns Question or None 
        if question:
          path.append(question)
      # collect node and walk
      frnode = next_tonode
      path.append(frnode)
    # append END node
    path.append(next_tonode)
    return path,rfc

  def assert_ask_question(self,frnode):
    # don't ask during rand_curr
    if self.rand_curr and frnode.name == 'BEGIN':
      return False
    # don't ask .5 transition
    if (frnode.name[:7] == 'LOCNODE'):
      return False
    # ask
    if (random.random() < QUESTION_PR) and (self.askingQ):
      return True
  
  # Question methods 

  def get_question(self,frnode,tonode,rfc):
    """ wrapper question getter 
    calls either filler or transition question constructor
    both constructors return None when question type not available
    otherwise returns question object 
    """
    # w.p. attempt to find valid filler question
    if random.random() < FILLER_QUESTION_PR:
      question = self.get_filler_question(frnode,tonode,rfc) 
    # o.w. attempt to find transition question
    else: 
      question = self.get_transition_question(frnode,tonode,rfc)
    return question

  def assert_valid_filler_question(self,tonode,rfc1,rfc2):
    """ returns true if valid filler question
    for now filler questions are those where the sentence produced
    by tonode under the two rfc's explicitly mention about different subjects
    """
    if 'subject.name' in tonode.sent\
      and tonode.assert_sents_differ(rfc1,rfc2)\
      and rfc1.assert_roles_differ(rfc2,'subject'):
      return True 
    else:
      return False


  def get_filler_question(self,frnode,tonode,true_rfc):
    """ fillerQ: same tonode, false_rfc (different subject)
    check if a valid filler question exists by looking for a false rfc 
    which gives a different sentence and which has a different subject
    if exits, return question, else return None
    """
    if frnode.name == "BEGIN": 
      return None
    random.shuffle(self.ont._list)
    # look for rfc which produces sentences filled with different subjects
    for false_rfc in self.ont._list:
      if self.assert_valid_filler_question(tonode,true_rfc,false_rfc):
        return FillerQ(frnode=frnode,true_tonode=tonode,
                       true_rfc=true_rfc,false_rfc=false_rfc)
    return None

  def get_transition_question(self,frnode,true_tonode,true_rfc):
    """ transitionQ: false_tonode, same rfc
    check if valid transition question exists
    by looking for false_tonode that differs from true tonode
    if exists return question, else None
    """
    edge = self.graph.get_edge(frnode,true_rfc) # {tonode1:pr,tonode2:pr,}
    tonode_L = list(edge.keys()) 
    tonode_L.remove(true_tonode)
    if len(tonode_L) == 0: 
      question = None
    else: 
      false_tonode = random.choice(tonode_L)
      question = TransitionQ(
                  frnode=frnode,true_tonode=true_tonode,
                  true_rfc=true_rfc,false_tonode=false_tonode)
    return question

  # wrapper: controls blocking

  def gen_k_paths_blocked(self,num_paths,block_len,blocking_factor='location.latent'):
    """ 
    """
    self.rand_curr = False
    path_L,rfc_L = [],[]
    bfL = [blocking_factor+'.true',blocking_factor+'.false']
    random.shuffle(bfL)
    for path_num in range(num_paths):
      # alternating blocking factors
      if path_num%(block_len)==0: 
        bfL.reverse()
      block_rfc_cond = bfL[0]
      # generate path and moveon
      path,rfc = self.gen_path(rfc_cond=block_rfc_cond) 
      path_L.append(path)
      rfc_L.append(rfc)
    return path_L,rfc_L

  def gen_k_paths_randomized(self,num_paths,pr_shift,blocking_factor='location.latent'):
    """ 
    # change this so that it's parametrized by pr_shift
    wrapper for controling blocking
    """
    self.rand_curr = True
    path_L,rfc_L = [],[]
    bfL = [blocking_factor+'.true',blocking_factor+'.false']
    random.shuffle(bfL)
    for path_num in range(num_paths):
      # random shift blocking factors
      if random.binomial(1,pr_shift):
        bfL.reverse()
      rfc_cond = bfL[0]
      # generate path and moveon
      path,rfc = self.gen_path(rfc_cond=rfc_cond) 
      path_L.append(path)
      rfc_L.append(rfc)
    return path_L,rfc_L

  def gen_k_paths(self,num_paths,train_curr=CURR):
    """ 
    ** main wrapper for generating an experiment **
    this function is called in cswMturk with the default cswEngine settings
    this function can also be called in a notebook for debugging & inspection
    """
    num_train_paths = int((4/5)*num_paths)
    num_test_paths = int((1/5)*num_paths)
    # training curricula
    self.askingQ = True 
    if train_curr == 'block':
      pathL,rfcL = self.gen_k_paths_blocked(num_train_paths,block_len=BLOCK_LEN)
    elif train_curr == 'random':
      pathL,rfcL = self.gen_k_paths_randomized(num_train_paths,pr_shift=PR_SHIFT)
    # final random curricula
    te_pathL,te_rfcL = self.gen_k_paths_randomized(num_test_paths,pr_shift=0.5)
    # append train and test pathL
    pathL.extend(te_pathL)
    rfcL.extend(te_rfcL)
    return pathL,rfcL

  def gen_k_paths_inserted_block(self,block_position=BLOCK_POSITION):
    """ 
    ** main wrapper for generating an experiment **
    this function is used for generating the csw experiments where we insert
      a block at different points of an interleaved experiment
    block_position = 0 (ABII), 1 (IABI), 2 (IIAB)
    """
    # training curricula
    self.askingQ = True 
    pathL,rfcL = [],[]
    if block_position == 0: # ABII
      sec_pathL,sec_rfcL = self.gen_k_paths_blocked(num_paths=80,block_len=40)
      pathL.extend(sec_pathL)
      rfcL.extend(sec_rfcL)
      sec_pathL,sec_rfcL = self.gen_k_paths_blocked(num_paths=80,block_len=1)
      pathL.extend(sec_pathL)
      rfcL.extend(sec_rfcL)
    elif block_position == 1: # IABI
      sec_pathL,sec_rfcL = self.gen_k_paths_blocked(num_paths=40,block_len=1)
      pathL.extend(sec_pathL)
      rfcL.extend(sec_rfcL)
      sec_pathL,sec_rfcL = self.gen_k_paths_blocked(num_paths=80,block_len=40)
      pathL.extend(sec_pathL)
      rfcL.extend(sec_rfcL)
      sec_pathL,sec_rfcL = self.gen_k_paths_blocked(num_paths=40,block_len=1)
      pathL.extend(sec_pathL)
      rfcL.extend(sec_rfcL)
    elif block_position == 2: # IIAB
      sec_pathL,sec_rfcL = self.gen_k_paths_blocked(num_paths=80,block_len=1)
      pathL.extend(sec_pathL)
      rfcL.extend(sec_rfcL)
      sec_pathL,sec_rfcL = self.gen_k_paths_blocked(num_paths=80,block_len=40)
      pathL.extend(sec_pathL)
      rfcL.extend(sec_rfcL)
    # final random test phase curriculum
    te_pathL,te_rfcL = self.gen_k_paths_randomized(40,pr_shift=0.5)
    # append train and test pathL
    pathL.extend(te_pathL)
    rfcL.extend(te_rfcL)
    return pathL,rfcL
    












