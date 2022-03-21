import torch
import regex as rex
re = rex
import pandas as pd
from collections import defaultdict 
import matplotlib.pyplot as plt
import networkx as nx
import cra_function_library as cralib

#----------------------------------------------------------------------------------
def construct_edges(df, node_list: list):
  """
  Computes unique set of nodes where nodes are either person's name or email

  Arguments
  ---------
  df (DataFrame)
    Email contents with attributes.

  node_list (list)
    List of nodes. 

  Return
  ------
    node_list (list)
    Sorted list of names for the nodes
  """

  edge_dict = defaultdict(set)
  df.groupby('From')

  for node in node_list:
      # scan to:
      # scan cc:
      pass


  to_list = df['To']
  from_list = df['From']
  cc_list = df['CC']
  node_list = []

  for el in from_list:
      if type(el) == float: continue
      node_list.append(el)

  for lst in cc_list:
      split = lst.split(';')
      for el in split:
          if type(el) == float: continue
          node_list.append(el)

  for lst in to_list:
      split = lst.split(';')
      for el in split:
          if type(el) == float: continue
          node_list.append(el)

  for el in node_list:
      if len(el) > 50:
          print(el)
      
  node_list = sorted(set(node_list))

  return node_list 
#------------------------------------------------------------------------------------------------
def choose_month_year(df, year, month):
    min_date = f"{year}-{month}-01"
    max_date = f"{year}-{month}-31"
    month_df = df[(df['date_sent'].str.lower() <= max_date) & (df['date_sent'].str.lower() >= min_date)]
    return month_df

def dict_month_year(df):
    dct = {}
    years = [('%4d' % year) for year in range(2008, 2018)]
    months = [('%02d' % month) for month in range(1,13)]
    for year in years:
        for month in months:
            dct[(year, month)] = choose_month_year(df, year, month)
    return dct
#------------------------------------------------------------------------------------------------
def get_unique_nodes_ge(df, verbose=False, keep_only_emails=True):
  """
  Computes unique set of nodes where nodes are either person's name or email
  Arguments
  ---------
  df (DataFrame)
    Email contents with attributes

  verbose (bool)
    If True, print elements of To: list longer than 50 characterss.

  keep_only_emails (bool)
    If True, only keep nodes that take the form of an email (contains '@')

  Return
  ------
  node_list (list)
    Sorted list of names for the nodes

  Notes
  -----
  The returned nodes include all nodes from From, CC, and To fields. They 
  are only emails if keep_only_email is true. 
  """

  from_list = df['From']
  print("df: ", df.shape)
  to_list = df['To']
  cc_list = df['CC']
  node_list = set()
  sender_list = set()
  to_receiver_list = set()
  cc_receiver_list = set()

  for el in from_list:
      if type(el) == float: continue
      if keep_only_emails and not rex.match(r'.*@', el):
          continue
      node_list.add(el)
      sender_list.add(el)

  for lst in cc_list:
      split = lst.split(';')
      for el in split:
          if type(el) == float: continue
          if keep_only_emails and not rex.match(r'.*@', el):
            continue
          node_list.add(el)
          cc_receiver_list.add(el)

  for lst in to_list:
      split = lst.split(';')
      for el in split:
          if type(el) == float: continue
          if keep_only_emails and not rex.match(r'.*@', el):
            continue
          node_list.add(el)
          to_receiver_list.add(el)

  for el in node_list:
      if verbose and len(el) > 50:
          print(el)
      
  node_list = sorted(node_list)
  new_node_list = []

  """
  for node in node_list:
      if keep_only_emails and rex.match(r'.*@', node):
          #print(node)
          new_node_list.append(node)
      elif not keep_only_emails: 
          new_node_list.append(node)
  """

  print("sender_list: ",  len(sender_list))
  print("cc_receiver_list: ",  len(cc_receiver_list))
  print("to_receiver_list: ",  len(to_receiver_list))
  print("node_list: ", len(node_list))

  xx = cc_receiver_list.difference(sender_list)
  yy = to_receiver_list.difference(sender_list)
  print("in cc but not in sender = cc - sender: ", len(xx))
  print("in to but not in sender = to - sender: ", len(yy))

  return new_node_list 
#------------------------------------------------------------------------------------------------
def get_receiver_nodes(df, from_node, keep_only_emails=True, headers_as_list=False):
    """ 
    Compute list of nodes connected to from_node. 

    Arguments
    ---------
    df (DataFrame)
        DataFrame that contains all relevant email data.

    from_node (string or int)
        Each edges goes from from_node to the nodes returned in rec_nodes

    keep_only_emails (bool)
        If True, all receiver nodes have an associated email. 
        If False, receivers might be an email ('@') or a name (no '@')

    Return
    ------
    rec_nodes (set)
        return a set of receiver nodes, each connected to from_node
    """
    rec_nodes = set()

    for lst in df.To:
        if headers_as_list:
            split = lst
        else:
            split = lst.split(";")
        for el in split:
            if rex.match(r'.*invalid', el): continue
            if keep_only_emails and not rex.match(r'.*@', el): continue
            rec_nodes.add(el)

    for lst in df.CC:
        if headers_as_list:
            split = lst
        else:
            split = lst.split(";")
        for el in split:
            if rex.match(r'.*invalid', el): continue
            if keep_only_emails and not rex.match(r'.*@', el): continue
            rec_nodes.add(el)

    return rec_nodes
#------------------------------------------------------------------------------------------------
def make_edgelist(df, nodes):
  """
    Create an edgelist from the dataframe and a list of nodes

    Arguments
    ---------
    df (DataFrame)
        Contains the emails and their attributes

    nodes (list)
        List of names/emails (called emails) collected from From:, CC:, and To: emails headers

    Return
    -----
    edgelist: List of edges
  """
  #takes in a dataframe and a list of str
  #df is supposed to be the email dataset and nodes is the nodes generated from the dataset
  edgelist = set()
  #creates a set so that there are only unique edges
  #maybe in the future it can account for multiple edges

  for node in nodes:
    #it goes through the nodes and gets all of the emails that were sent from that node
    if node != '':
      from_who = df.loc[df['From'] == node]
      r = len(from_who.index)

      for i in range(r):
        #iterates through all of the emails that were sent from a particular node
        #and saves all of the values encased in () and appends it to the edge list
        j = from_who.iloc[i:i+1]
        x = j['To']
        #y = j['CC']
        index_1 = nodes.index(node)
        for xi in x:
          #goes through the values in To and adds it to the edge list
          if xi != '':
            result = re.findall('\(.*?\)', xi)
            for s in result:
              index_2 = nodes.index(s)
              edgelist.add(tuple([index_1, index_2]))
        '''
        for yi in y:
          #this is done for CC as well as From
          if yi != '':
            result = re.findall('\(.*?\)', yi)
            for s in result:
              index_2 = nodes.index(s)
              edgelist.add(tuple([index_1, index_2]))
        '''
  #this returns an edge list where the index is formatted [sender, reciever]
  return list(edgelist)
#------------------------------------------------------------------------------------------------

### A simple Message-Passing network w/ common aggregation schemes.
class Simple_GNN(torch.nn.Module):

    def __init__(self,in_features,int_features,out_features,depth,aggregation_mode = 'mean',dropout_prob = .1):
        '''
        [in_features]       - # of input features.
        [int_features]      - # of features in message-passing layers. Within the 
                              GNN literature, this is typically a constant.
        [out_features]      - # of output features. Corresponds to # of classes,
                              regression targets, etc.
        [depth]             - # of message-passing layers. 
        [aggregation_mode]  - choice of aggregation scheme. Can be 'mean',
                              'sum', 'max', or 'none'
        [dropout_prob]      - probability used for Dropout (see Srivastava et al., 2017)
        '''

        super(Simple_GNN,self).__init__()
        assert aggregation_mode in ['mean','sum','max','none']

        self.f_in = torch.nn.Linear(in_features,int_features)
        self.f_int = torch.nn.ModuleList([torch.nn.Sequential(*[torch.nn.Linear(int_features,int_features),
                                                                torch.nn.LeakyReLU(),
                                                                torch.nn.Dropout(dropout_prob)])
                                              for _ in range(depth)])
        self.f_out = torch.nn.Linear(int_features,out_features)

        if aggregation_mode == 'mean': self.agg = torch_scatter.scatter_mean
        elif aggregation_mode == 'sum': self.agg = torch_scatter.scatter_sum
        elif aggregation_mode == 'max': self.agg = torch_scatter.scatter_max

        self.aggregation_mode = aggregation_mode

    def forward(self,node_features,edge_index,edge_weights = None):
        '''
        [node_features]  - Matrix of node features. First (batch) dimension corresponds
                           to nodes; second to features.
        [edge_index]     - Edge list representation of a graph. Shape [num_edges]x2
        [edge_weights]   - Optional scalar edge weights. Shape [num_edges,1]
        '''

        node_features = self.f_in(node_features)
        # for idx,layer in enumerate(self.f_int):  # Original
        for idx,layer in enumerate(range(self.f_int)):
            # if self.aggregation_mode is not 'none':  # orig
            if self.aggregation_mode != 'none':
                if edge_weights != None:
                    aggregated_node_features = self.agg(edge_weights * node_features[edge_index[:,0]],
                                                                    edge_index[:,1],dim=0)
                else:
                    aggregated_node_features = self.agg(node_features[edge_index[:,0]],edge_index[:,1],dim=0) 
            else:
                aggregated_node_features = node_features
                
            if isinstance(aggregated_node_features,tuple): 
                aggregated_node_features = aggregated_node_features[0]

            node_features = node_features + layer(aggregated_node_features)
        return self.f_out(node_features),node_features
#--------------------------------------------------------------------------------------
## General purpose object for storing graph data
class Graph(object):
    def __init__(self,edge_index = None,edge_metadata = None,node_metadata = None):
        self.edge_index = edge_index
        self.edge_metadata = edge_metadata
        self.node_metadata = node_metadata
        self.num_nodes = None

    def read_edges(self,filepath):
        ''' Edges should be stored as follows:
            0 1 3.1 2.0
            0 2 1.0 0.0
            1 0 3.1 2.0
            ...
        where the first two columns are node indices and the remainder
        are edge features. '''

        with open(filepath,'r') as f:
          edges = torch.Tensor([list(map(float,line.strip().split(' '))) \
                                         for line in f.readlines()])
        if edges.shape[1] == 2:
          self.edge_index = edges.long()
        elif edges.shape[1] >= 3:
          self.edge_index = edges[:,:2].long()
          self.edge_metadata = edges[:,2::]

    def read_node_metadata(self,filepath,padding_value = 0.0):
        ''' Nodes should be stored as follows:
            0 4.1 9.2 1.1 ...
            2 3.3 1.1 9.0 ...
            ...
        where the first column is the node index and the remainder
        are node features. If a node is not listed but should still
        clearly exist (e.g, nodes '0' and '2' are present in the file
        but not node '1'), the missing node is given constant features
        set to [padding_value]'''

        with open(filepath,'r') as f:
          nodes = torch.Tensor([list(map(float,line.strip().split(' '))) \
                                         for line in f.readlines()])

        self.num_nodes = 1 + nodes[:,0].max().long()
        node_metadata = padding_value * torch.ones((self.num_nodes,nodes.shape[1]-1))
        node_metadata[nodes[:,0].long()] = nodes[:,1::]   #GE

        self.node_metadata = node_metadatav
#----------------------------------------------------------------------
### Dataset consisting of multiple _Graph_ objects
class GraphsDataset(object):
    ''' Graph metadata is expected to be saved
        under [graph_dir]/[...]/edges.txt and
        [graph_dir]/[...]/nodes.txt.
    ******* Optional Arguments *********
    [add_self_loops] - Add self loop to each graph.
                       This tends to improve importance,
                       and it resolves issues with
                       disconnected nodes.
    '''
    def __init__(self,graph_dir,add_self_loops = True):
        self.root = graph_dir
        self.graphs = os.listdir(graph_dir)
        self.add_self_loops = add_self_loops
    def __len__(self):
        return len(self.graphs)
    def __getitem__(self,idx):
        G = Graph()
        G.read_edges(self.root + '/' + self.graphs[idx]+'/edges.txt')
        G.read_node_metadata(self.root + '/' + self.graphs[idx]+'/nodes.txt')

        if self.add_self_loops:
            self_loops = torch.cat((torch.arange(G.num_nodes)[:,None],torch.arange(G.num_nodes)[:,None]),dim=-1)
            G.edge_index = torch.cat((G.edge_index,self_loops),dim=0)
            G.edge_metadata = torch.cat((G.edge_metadata,torch.ones((G.num_nodes,1))),dim=0)
        return G

### Helper function to construct batches of graphs. We assume the user wants to
### perform graph-level tasks, and so our target variable is chosen as
### G.node_metadata[0,0] i.e, the first feature of the first node.
def collate_fn(data):
    c,n = 0,0
    for idx,G in enumerate(data):
        if idx == 0:
            X,Y = G.node_metadata[:,1::],G.node_metadata[0,0][None]
            edge_index,edge_weights = G.edge_index,G.edge_metadata
            batch_index = c * torch.ones(G.num_nodes)
        else:
            X = torch.cat((X,G.node_metadata[:,1::]),dim=0)
            Y = torch.cat((Y,G.node_metadata[0,0][None]),dim=0)

            edge_index = torch.cat((edge_index,n + G.edge_index),dim=0)
            edge_weights = torch.cat((edge_weights,G.edge_metadata),dim=0)
            batch_index = torch.cat((batch_index,c * torch.ones(G.num_nodes)),dim=0)
        n += G.num_nodes
        c += 1

    return {'X':X,'edge_index':edge_index,'edge_weights':edge_weights,'Y':Y.long(),'batch_index':batch_index.long()}
#---------------------------------------------------------------------------------------------------
### Call to fit node-level model and save results
class GNN_Node_Trainer(object):
    def __init__(self,graph,**kwargs):
        '''
        [graph]  - Instance of _Graph_. Expects to be fully initialized.
        ******* Optional Arguments *********
        [train_index],[test_index] - Partitions graph into train and test sets.
                                     Defaults to random 60%/40% split.
        [num_epochs]               - # of epochs to train. Defaults to 20.
        [loss_func]                - Loss function to minimize. Defaults to
                                     Cross Entropy.
        [random_seed]              - Defaults to 0.
        '''
        torch.manual_seed(kwargs.get('random_seed',0))

        assert isinstance(graph,Graph)
        self.graph = graph

        ### If train_index and test_index are not provided, we randomly
        ### select a train/test split.
        r = torch.randperm(graph.num_nodes)
        self.train_index,self.test_index = kwargs.get('train_index',r[:int(.6 * len(r))]),\
                                  kwargs.get('test_index',r[int(.6 * len(r))::])

        self.num_epochs = kwargs.get('num_epochs',20)
        self.loss_func = kwargs.get('loss_func',lambda x,y,*args: torch.nn.functional.cross_entropy(x,y.long()))

    def __call__(self,model,**kwargs):
        '''
        [model]  - i.e, our GNN
        ******* Optional Arguments *********
        [lr],[beta],['weight_decay'] - Optimizer parameters. Default to 1e-3, (.9,.999),
                                       and 1e-2.
        [quiet]                      - Boolean. Disables logging to stdout
        [metrics_callback]           - Dict of functions to compute additional metrics.
        [title]                      - Saves metrics and model ckpt to 'gnn_results/[title]'
                                       at end of training. Defaults to 'run_0'.
        [device]                     - Should be 'cpu' or 'cuda:0'. Defaults to 'cpu'.

        Note: We assume the first column of 'self.graph.node_metedata' to be our
        target values.
        '''
        opt = torch.optim.Adam(model.parameters(),lr=kwargs.get('lr',1e-3),
                                        betas = kwargs.get('beta',(0.9, 0.999)),
                                        weight_decay=kwargs.get('weight_decay',1e-2)
                                )

        device = torch.device(kwargs.get('device','cpu'))
        model.to(device)

        ### Load data to device.
        node_features,edge_index,edge_weights = self.graph.node_metadata.to(device),\
                                                    self.graph.edge_index.to(device),\
                                                    self.graph.edge_metadata.to(device)
        X,Y = node_features[:,1::].to(device),node_features[:,0].to(device)

        metrics = {'train_loss':[],'test_loss':[]}
        metrics_callback = kwargs.get('metrics_callback',{})
        for key in metrics_callback.keys():
            metrics['train_'+key] = []
            metrics['test_'+key] = []

        pbar = tqdm.tqdm(range(self.num_epochs),position=0,disable=kwargs.get('quiet',False))
        epoch_list = []
        train_loss_list = []
        test_loss_list = []
        for idx in pbar:
            ### Get predictions and compute losses over train and test sets.
            predictions,_ = model(X,edge_index,edge_weights)
            train_loss,test_loss = self.loss_func(predictions[self.train_index],Y[self.train_index],
                                                  edge_index,edge_weights),\
                                          self.loss_func(predictions[self.test_index],Y[self.test_index],
                                                  edge_index,edge_weights)

            epoch_list.append(idx)
            train_loss_list.append(train_loss.item())
            test_loss_list.append(test_loss.item())

            pbar.set_description(f'Train Loss: {train_loss.item():.3f}\tTest Loss: {test_loss.item():.3f}')
            if torch.isnan(train_loss): raise ValueError('Training loss is NaN')

            ### Backpropagate w.r.t training loss
            train_loss.backward()
            opt.step()
            opt.zero_grad()

            ### Save losses and compute additional metrics
            metrics['train_loss'].append(train_loss.item())
            metrics['test_loss'].append(test_loss.item())
            for key in metrics_callback.keys():
                metrics['train_'+key].append(metrics_callback[key](predictions[self.train_index],Y[self.train_index],
                                                        edge_index,edge_weights))
                metrics['test_'+key].append(metrics_callback[key](predictions[self.test_index],Y[self.test_index],
                                                        edge_index,edge_weights))

        plt.figure()
        l1 = plt.plot(epoch_list, train_loss_list, c='red')
        l2 = plt.plot(epoch_list, test_loss_list, c='blue')
        plt.legend((l1,l2),('train loss','test loss'))
        plt.title('Loss vs epoch')
        plt.savefig('loss_epoch.png')
        plt.show()

        os.makedirs('gnn_results',exist_ok=True)
        title = kwargs.get('title','run_0')
        os.makedirs('gnn_results/{}'.format(title),exist_ok=True)
        torch.save(model.state_dict(),'gnn_results/{}/ckpt'.format(title))
        torch.save(metrics,'gnn_results/{}/metrics'.format(title))

        return predictions
#-------------------------------------------------------------------------------------------------------
### Call to fit graph-level model and save results
class GNN_Graph_Trainer(object):
    def __init__(self,graphs,collate_fn,**kwargs):
        '''
        [graphs]      - Instance of _GraphsDataset_. Expects to be fully initialized.
        [collate_fn]  - Collate function for batching purposes.
        ******* Optional Arguments *********
        [batch_size]               - Number of graphs per train/val batch. Defaults
                                     to 50.
        [train_%]                  - Pct. split into train set. Defaults to 80%/20%.
        [num_epochs]               - # of epochs to train. Defaults to 20.
        [loss_func]                - Loss function to minimize. Defaults to
                                     Cross Entropy.
        [random_seed]              - Defaults to 0.
        '''
        torch.manual_seed(kwargs.get('random_seed',0))

        assert isinstance(graphs,GraphsDataset)
        self.graphs = graphs
        self.collate_fn = collate_fn

        self.train_pct = kwargs.get('train_%',.8)

        self.batch_size = kwargs.get('batch_size',50)
        self.num_epochs = kwargs.get('num_epochs',20)
        self.loss_func = kwargs.get('loss_func',lambda x,y,*args: torch.nn.functional.cross_entropy(x,y.long()))

    def __call__(self,model,**kwargs):
        '''
        [model]  - i.e, our GNN
        ******* Optional Arguments *********
        [lr],[beta],['weight_decay'] - Optimizer parameters. Default to 1e-3, (.9,.999),
                                       and 1e-2.
        [quiet]                      - Boolean. Disables logging to stdout
        [metrics_callback]           - Dict of functions to compute additional metrics.
        [title]                      - Saves metrics and model ckpt to 'gnn_results/[title]'
                                       at end of training. Defaults to 'run_0'.
        [device]                     - Should be 'cpu' or 'cuda:0'. Defaults to 'cpu'.

        Note: We assume 'self.graphs[idx].node_metedata[0,0]'
        to be our target value for all idx \in [0,len(self.graphs)] .
        '''
        opt = torch.optim.Adam(model.parameters(),lr=kwargs.get('lr',1e-3),
                                        betas = kwargs.get('beta',(0.9, 0.999)),
                                        weight_decay=kwargs.get('weight_decay',1e-2)
                                )

        device = torch.device(kwargs.get('device','cpu'))
        model.to(device)

        ### Generate train/test split.
        train,test = torch.utils.data.random_split(self.graphs,[int(len(self.graphs) * self.train_pct),
                                                               len(self.graphs) - int(len(self.graphs) * self.train_pct)]
                                                  )
        train_loader,test_loader = torch.utils.data.DataLoader(train,batch_size=self.batch_size,shuffle=True,collate_fn=self.collate_fn),\
                              torch.utils.data.DataLoader(test,batch_size=self.batch_size,shuffle=True,collate_fn=self.collate_fn)

        metrics = {'train_loss':[],'test_loss':[]}
        metrics_callback = kwargs.get('metrics_callback',{})
        for key in metrics_callback.keys():
            metrics['train_'+key] = []
            metrics['test_'+key] = []

        pbar = tqdm.tqdm(range(self.num_epochs),position=0,disable=kwargs.get('quiet',False))
        for idx in pbar:

            ### Training loop
            model.train()
            for data in train_loader:
                X,Y,edge_index,edge_weights,batch_index = data['X'].to(device),data['Y'].to(device),data['edge_index'].to(device),\
                                                              data['edge_weights'].to(device),data['batch_index'].to(device)

                predictions,_ = model(X,edge_index,edge_weights)
                ### To keep things simple, we employ mean READOUT to produce a global
                ### graph embedding.
                train_loss = self.loss_func(torch_scatter.scatter_mean(predictions,batch_index,dim=0),Y,edge_index,edge_weights)

                pbar.set_description(f'Train Loss: {train_loss.item():.3f}')
                if torch.isnan(train_loss): raise ValueError('Training loss is NaN')

                train_loss.backward()
                opt.step()
                opt.zero_grad()

                ### Compute and save metrics.
                metrics['train_loss'].append(train_loss.item())
                for key in metrics_callback.keys():
                        metrics['train_'+key].append(metrics_callback[key](torch_scatter.scatter_mean(predictions,batch_index,dim=0),
                                                                    Y,edge_index,edge_weights))

            ### Evaluation loop.
            model.eval()
            with torch.no_grad():
                for data in test_loader:
                    X,Y,edge_index,edge_weights,batch_index = data['X'].to(device),data['Y'].to(device),data['edge_index'].to(device),\
                                                              data['edge_weights'].to(device),data['batch_index'].to(device)

                    predictions,_ = model(X,edge_index,edge_weights)
                    train_loss = self.loss_func(torch_scatter.scatter_mean(predictions,batch_index,dim=0),Y,edge_index,edge_weights)

                    pbar.set_description(f'Test Loss: {train_loss.item():.3f}')

                    metrics['test_loss'].append(train_loss.item())
                    for key in metrics_callback.keys():
                          metrics['test_'+key].append(metrics_callback[key](torch_scatter.scatter_mean(predictions,batch_index,dim=0),
                                                                    Y,edge_index,edge_weights))

        os.makedirs('gnn_results',exist_ok=True)
        title = kwargs.get('title','run_0')
        os.makedirs('gnn_results/{}'.format(title),exist_ok=True)
        torch.save(model.state_dict(),'gnn_results/{}/ckpt'.format(title))
        torch.save(metrics,'gnn_results/{}/metrics'.format(title))

        return predictions### Call to fit graph-level model and save results
class GNN_Graph_Trainer(object):
    def __init__(self,graphs,collate_fn,**kwargs):
        '''
        [graphs]      - Instance of _GraphsDataset_. Expects to be fully initialized.
        [collate_fn]  - Collate function for batching purposes.
        ******* Optional Arguments *********
        [batch_size]               - Number of graphs per train/val batch. Defaults
                                     to 50.
        [train_%]                  - Pct. split into train set. Defaults to 80%/20%.
        [num_epochs]               - # of epochs to train. Defaults to 20.
        [loss_func]                - Loss function to minimize. Defaults to
                                     Cross Entropy.
        [random_seed]              - Defaults to 0.
        '''
        torch.manual_seed(kwargs.get('random_seed',0))

        assert isinstance(graphs,GraphsDataset)
        self.graphs = graphs
        self.collate_fn = collate_fn

        self.train_pct = kwargs.get('train_%',.8)

        self.batch_size = kwargs.get('batch_size',50)
        self.num_epochs = kwargs.get('num_epochs',20)
        self.loss_func = kwargs.get('loss_func',lambda x,y,*args: torch.nn.functional.cross_entropy(x,y.long()))

    def __call__(self,model,**kwargs):
        '''
        [model]  - i.e, our GNN
        ******* Optional Arguments *********
        [lr],[beta],['weight_decay'] - Optimizer parameters. Default to 1e-3, (.9,.999),
                                       and 1e-2.
        [quiet]                      - Boolean. Disables logging to stdout
        [metrics_callback]           - Dict of functions to compute additional metrics.
        [title]                      - Saves metrics and model ckpt to 'gnn_results/[title]'
                                       at end of training. Defaults to 'run_0'.
        [device]                     - Should be 'cpu' or 'cuda:0'. Defaults to 'cpu'.

        Note: We assume 'self.graphs[idx].node_metedata[0,0]'
        to be our target value for all idx \in [0,len(self.graphs)] .
        '''
        opt = torch.optim.Adam(model.parameters(),lr=kwargs.get('lr',1e-3),
                                        betas = kwargs.get('beta',(0.9, 0.999)),
                                        weight_decay=kwargs.get('weight_decay',1e-2)
                                )

        device = torch.device(kwargs.get('device','cpu'))
        model.to(device)

        ### Generate train/test split.
        train,test = torch.utils.data.random_split(self.graphs,[int(len(self.graphs) * self.train_pct),
                                                               len(self.graphs) - int(len(self.graphs) * self.train_pct)]
                                                  )
        train_loader,test_loader = torch.utils.data.DataLoader(train,batch_size=self.batch_size,shuffle=True,collate_fn=self.collate_fn),\
                              torch.utils.data.DataLoader(test,batch_size=self.batch_size,shuffle=True,collate_fn=self.collate_fn)

        metrics = {'train_loss':[],'test_loss':[]}
        metrics_callback = kwargs.get('metrics_callback',{})
        for key in metrics_callback.keys():
            metrics['train_'+key] = []
            metrics['test_'+key] = []

        pbar = tqdm.tqdm(range(self.num_epochs),position=0,disable=kwargs.get('quiet',False))
        for idx in pbar:

            ### Training loop
            model.train()
            for data in train_loader:
                X,Y,edge_index,edge_weights,batch_index = data['X'].to(device),data['Y'].to(device),data['edge_index'].to(device),\
                                                              data['edge_weights'].to(device),data['batch_index'].to(device)

                predictions,_ = model(X,edge_index,edge_weights)
                ### To keep things simple, we employ mean READOUT to produce a global
                ### graph embedding.
                train_loss = self.loss_func(torch_scatter.scatter_mean(predictions,batch_index,dim=0),Y,edge_index,edge_weights)

                pbar.set_description(f'Train Loss: {train_loss.item():.3f}')
                if torch.isnan(train_loss): raise ValueError('Training loss is NaN')

                train_loss.backward()
                opt.step()
                opt.zero_grad()

                ### Compute and save metrics.
                metrics['train_loss'].append(train_loss.item())
                for key in metrics_callback.keys():
                        metrics['train_'+key].append(metrics_callback[key](torch_scatter.scatter_mean(predictions,batch_index,dim=0),
                                                                    Y,edge_index,edge_weights))

            ### Evaluation loop.
            model.eval()
            with torch.no_grad():
                for data in test_loader:
                    X,Y,edge_index,edge_weights,batch_index = data['X'].to(device),data['Y'].to(device),data['edge_index'].to(device),\
                                                              data['edge_weights'].to(device),data['batch_index'].to(device)

                    predictions,_ = model(X,edge_index,edge_weights)
                    train_loss = self.loss_func(torch_scatter.scatter_mean(predictions,batch_index,dim=0),Y,edge_index,edge_weights)

                    pbar.set_description(f'Test Loss: {train_loss.item():.3f}')

                    metrics['test_loss'].append(train_loss.item())
                    for key in metrics_callback.keys():
                          metrics['test_'+key].append(metrics_callback[key](torch_scatter.scatter_mean(predictions,batch_index,dim=0),
                                                                    Y,edge_index,edge_weights))

        os.makedirs('gnn_results',exist_ok=True)
        title = kwargs.get('title','run_0')
        os.makedirs('gnn_results/{}'.format(title),exist_ok=True)
        torch.save(model.state_dict(),'gnn_results/{}/ckpt'.format(title))
        torch.save(metrics,'gnn_results/{}/metrics'.format(title))

        return predictions
#---------------------------------------------------------------------------------------------
def sep_by_year(df):
  #this takes in a dataframe and seperates it by the year the emails were sent
  time = df['Sent'].tolist()
  year_list = []
  initial_i = 0
  currentyear = int((time[0])[0:4])
  #this is the specific index for Sent to get the year
  end = len(df)
  for i in range(end):
    #this iterates through the emails in the df and if the year increases it 
    #splits the dataset and appends it to the year_list variable
    s = time[i]
    x = int(s[0:4])
    
    if x > currentyear:
      year_span = df.iloc[initial_i:i]
      year_list.append(year_span)
      initial_i = i
      currentyear = x
  year_list.append(df.iloc[initial_i:end])
  if len(year_list) == 1:#if the year got bigger then it just returns the same df as before
    return df
  return year_list
#--------------------------------------------------------------------------------
def sep_by_month(df):
  #takes the same steps as sep_by_year except the index values have changed
  #so that it retrieves the months instead of years
  time = df['Sent'].tolist()
  month_list = []
  initial_i = 0
  currentmonth = int((time[0])[5:7])
  end = len(df)
  for i in range(end):

    s = time[i]
    x = int(s[5:7])

    if x > currentmonth:
      month_span = df.iloc[initial_i:i]
      month_list.append(month_span)
      initial_i = i
      currentmonth = x
  month_list.append(df.iloc[initial_i:end])
  if len(month_list) == 1:
    return df
  return month_list
#------------------------------------------------------------------------------
def sep_by_day(df):
  #takes the same steps as sep_by_year except the index values have changed
  #so that it retrieves the days instead of years
  time = df['Sent'].tolist()
  day_list = []
  initial_i = 0
  currentday = int((time[0])[8:10])
  end = len(df)
  for i in range(end):

    s = time[i]
    x = int(s[8:10])

    if x > currentday:
      day_span = df.iloc[initial_i:i]
      day_list.append(day_span)
      initial_i = i
      currentday = x
  day_list.append(df.iloc[initial_i:end])
  if len(day_list) == 1:
    return df
  return day_list
#------------------------------------------------------------------------------
def disp_Graph(nodes, edges, title):
  #this displays the graph based on the nodes and edges
  G = nx.DiGraph()
  r = []
  labeldict = {}
  for i in range(len(nodes)):
    r.append(i)
    labeldict[i] = nodes[i]
  G.add_nodes_from(r)
  G.add_edges_from(edges)
  plt.figure()
  plt.title(title)
  nx.draw(G,labels=labeldict, with_labels=True)
  plt.show()
#------------------------------------------------------------------------------
def create_edges_txt(edges):
  #creates a txt file that is just the edge list
  with open('edges.txt','w') as f:
    for edge in edges:
      f.write(f'{edge[0]} {edge[1]} {1}\n')
      #print(edge[0])
#------------------------------------------------------------------------------
def create_nodes_txt(nodes):#this one is more basic might delete in the future
  torch.manual_seed(0)
  with open('nodes.txt','w') as f:
    for node in nodes:
      X_str = ' '.join([str(element.item()) for element in .5 + torch.randn((16))])
      #display(X_str)
      #f.write(f'{nodes.index(node)}' + X_str + '\n')
      f.write(f'{nodes.index(node)} {nodes.index(node)} ' + X_str +'\n')
#------------------------------------------------------------------------------
def get_sent_list(edges, l):
  e_length = len(edges)
  sent_list = np.zeros(l)
  for i in range(e_length):
    x = list(edges[i])
    y = x[0]
    sent_list[y] = sent_list[y] + 1

  return sent_list
#------------------------------------------------------------------------------
def get_recieved_list(edges, l):
  e_length = len(edges)
  recieved_list = np.zeros(l)
  for i in range(e_length):
    x = list(edges[i])
    y = x[1]
    recieved_list[y] = recieved_list[y] + 1

  return recieved_list
#------------------------------------------------------------------------------
def create_nodes_txt(nodes):
  #takes a list that has the node list, the sent list, and the recieved list
  #this is a new version of create_nodes_txt that incorporates the node metadata
  torch.manual_seed(0)
  with open('nodes.txt','w') as f:
    for i in range(len(nodes[0])):
      #display(node)
      X_str = ' '.join([str(element.item()) for element in .5 + torch.randn((16))])
      s = ''
      for n in range(len(nodes)):
        s = s + str(nodes[n][i]) + ' '
      #f.write( str(nodes[0][i]) + ' ' + s + X_str +'\n')
      f.write( str(nodes[0][i]) + ' ' + s +'\n')
#------------------------------------------------------------------------------
def get_email_node_meta(df):
  body = df['Body']
  body_list = []
  char_count = []
  word_count = []
  mean_char_word = []
  std_char_word = []

  for x in body:# this interates through the values for the body and
    result = re.findall('\'.*?\'', x)#creates a list of all of the values in x that are contained in ''
    q = ' '.join(result)#this makes everything into one string
    wordCount = 0
    words = re.findall('\ .*?\ ', q)#this does the same as above but with spaces for the str
    body_list.append(q)
    num_words = np.ones(len(words))
    i = 0
    for word in words:
      num_words[i] = len(word)
      #display(len(word))
    l = len(q)

    #display(num_words)

    #error created in mean and std where Nan appears

    char_count.append(l)
    word_count.append(len(words))
    x_std = np.std(num_words)
    x_mean = np.mean(num_words)
    if math.isnan(x_mean):
      mean_char_word.append(0)
    else:
      mean_char_word.append(x_mean)

    if math.isnan(x_std):
      std_char_word.append(0)
    else:
      std_char_word.append(x_std)

  meta = [char_count, word_count, mean_char_word, std_char_word]
  return meta
#--------------------------------------------------------------------------------
def create_email_edgelist(df, sender_list):
  edgelist = []
  for sender in sender_list:
    from_who = df.loc[df['From'] == sender]
    r = len(from_who.index)

    index_l = df[df['From'] == sender].index.values
    for i in range(1,len(index_l)):
      edgelist.append(tuple([index_l[0], index_l[i]]))

  return edgelist
#--------------------------------------------------------------------------------
def create_weighted_edgelist(df, nodes):
  raw_edgelist = []
  for email in df:
    sender = email['From']
    index_1 = nodes.index(sender)
    reciever = email['To']
    result = re.findall('\(.*?\)', reciever)
    for s in result:
      index_2 = nodes.index(s)
      raw_edgelist.add(tuple([index_1, index_2]))
#--------------------------------------------------------------------------------
def create_node_list(df):
  #this takes in a pandas dataframe

  to_list = []
  from_list = []
  CC_list = []

  alist = df['To']
#this gets all of the values under To for the df

  for x in alist:# this interates through the values for To and
    result = re.findall('\(.*?\)', x)#appends all of the values in it that are contained in ()
    for s in result:
      if '@' in s:
        to_list.append(s)


  blist = df['From']#the same that is done for To is done for From  and CC

  for x in blist:
    result = re.findall('\(.*?\)', x)
    for s in result:
      if '@' in s:
        from_list.append(s)


  clist = df['CC']

  for x in clist:
    result = re.findall('\(.*?\)', x)
    for s in result:
      if '@' in s:
        CC_list.append(s)

  final_list = to_list + from_list #+ CC_list #The lists are combined together

  Myfinallist = sorted(set(final_list)) #this takes out any repeating values

  return Myfinallist #returns a list of str
#--------------------------------------------------------------------------------
def create_edge_list(df, nodes):
  #takes in a dataframe and a list of str
  #df is supposed to be the email dataset and nodes is the nodes generated from the dataset
  edgelist = set()
  #creates a set so that there are only unique edges
  #maybe in the future it can account for multiple edges

  for node in nodes:
    #it goes through the nodes and gets all of the emails that were sent from that node
    if node != '':
      from_who = df.loc[df['From'] == node]
      r = len(from_who.index)

      for i in range(r):
        #iterates through all of the emails that were sent from a particular node
        #and saves all of the values encased in () and appends it to the edge list
        j = from_who.iloc[i:i+1]
        x = j['To']
        y = j['CC']
        #display(x)
        #display('-----')
        index_1 = nodes.index(node)
        for xi in x:
          #display(xi)
          #goes through the values in To and adds it to the edge list
          if xi != '':
            result = re.findall('\(.*?\)', xi)
            for s in result:
              if s in nodes:
                index_2 = nodes.index(s)
                edgelist.add(tuple([index_1, index_2]))

        for yi in y:
          #this is done for CC as well as From
          if yi != '':
            result = re.findall('\(.*?\)', yi)
            for s in result:
              if s in nodes:
                index_2 = nodes.index(s)
                edgelist.add(tuple([index_1, index_2]))

  #this returns an edge list where the index is formatted [sender, reciever]
  return list(edgelist)
#--------------------------------------------------------------------------------
def get_rid_of_NaN(val):
  if math.isnan(val):
    return 0
  return val

def get_rid_of_emp_str(s):
  if s == '':
    return 0
  else:
    return len(s)
#--------------------------------------------------------------------------------
def get_node_recieved_list(df, nodes):
  recieved_email = []
  for node in nodes:
    alist = df['To']
    blist = df['CC']
    rec_num = 0

    for x in alist:
      result = re.findall('\(.*?\)', x)
      if node in result:
        rec_num += 1

    for x in blist:
      result = re.findall('\(.*?\)', x)
      if node in result:
        rec_num += 1

    recieved_email.append(rec_num)
  return recieved_email
#--------------------------------------------------------------------------------
def create_node_meta_list(df, nodes):
  node_meta_list = []
  for node in nodes:
    to_data = []
    cc_data = []
    char_data = []
    word_data = []

    from_who = df.loc[df['From'] == node]
    to_who = df.loc[df['To'] == node]
    cc_who = df.loc[df['CC'] == node]

    num_recieved = len(to_who) + len(cc_who)

    num_sent = len(from_who)

    r = len(from_who.index)
    for i in range(r):
        #iterates through all of the emails that were sent from a particular node
        #and saves all of the values encased in () and appends it to the edge list
      j = from_who.iloc[i:i+1]
      #display(j)
      to = j['To']
      cc = j['CC']
      body = j['Body']
      for to_i in to:
        if to_i == '':
          to_data.append(0)
        else:
          to_list_in_string = re.findall('\(.*?\)', to_i)
          to_data.append(len(to_list_in_string))
      for cc_i in cc:
        if cc_i == '':
          cc_data.append(0)
        else:
          cc_list_in_string = re.findall('\(.*?\)', cc_i)
          cc_data.append(len(cc_list_in_string))

      '''
      to_list_in_string = re.findall('\(.*?\)', to)
      cc_list_in_string = re.findall('\(.*?\)', cc)
      num_to = get_rid_of_emp_str(to_list_in_string)
      num_cc = get_rid_of_emp_str(cc_list_in_string)
      to_data.append(num_to)
      cc_data.append(num_cc)
      '''
      for bodies in body:
        b = re.findall('\'.*?\'', bodies)#creates a list of all of the values in x that are contained in ''
        body_string = ' '.join(b)
        num_char = len(body_string)
      
        char_data.append(num_char)

        words = re.findall('\ .*?\ ', body_string)
        #display(words)

        num_word = len(words)

        word_data.append(num_word)

    char_data = np.array(char_data)
    word_data = np.array(word_data)

    to_data = np.array(to_data)
    cc_data = np.array(cc_data)

    mean_char = np.mean(char_data)
    mean_word = np.mean(word_data)

    std_char = np.std(char_data)
    std_word = np.std(word_data)

    mean_to = np.mean(to_data)
    mean_cc = np.mean(cc_data)

    std_to = np.std(to_data)
    std_cc = np.std(cc_data)

    mean_char = get_rid_of_NaN(mean_char)
    mean_word = get_rid_of_NaN(mean_word)
    mean_to = get_rid_of_NaN(mean_to)
    mean_cc = get_rid_of_NaN(mean_cc)
    std_char = get_rid_of_NaN(std_char)
    std_word = get_rid_of_NaN(std_word)
    std_to = get_rid_of_NaN(std_to)
    std_cc = get_rid_of_NaN(std_cc)

    inloop_node_meta = [mean_char, mean_word, mean_to, mean_cc, std_char, std_word, std_to, std_cc, num_sent]

    node_meta_list.append(inloop_node_meta)
  return node_meta_list
#--------------------------------------------------------------------------------
def create_edge_meta_list(df, nodes, edges):
  for edge in edges:
    sender = edge[0]
    reciever = edge[1]
    num_char = []
    num_word = []
    num_to = []
    num_cc = []
    recieved_email = []

    to_N = 0
    cc_N = 0
    email_num = 0

    email_p1 = df.loc[df['From'] == sender]
    for email in email_p1:
      display(email_p1)
      display(sender)
      display(email)
      to = email['To']
      cc = email['CC']

      for to_i in to:
        para_sep_to = re.findall('\(.*?\)', to_i)
        for i in para_sep_to:
          if reciever in para_sep_to:
            body = email['Body']
            for bodies in body:
              b = re.findall('\'.*?\'', bodies)
              body_string = ' '.join(b)
              num_char = len(body_string)

              words = re.findall('\ .*?\ ', body_string)

              num_word = len(words)
              email_num += 1

              to_N += 1
      for cc_i in cc:
        para_sep_to = re.findall('\(.*?\)', cc_i)
        for i in para_sep_to:
          if reciever in para_sep_to:
            body = email['Body']
            for bodies in body:
              c = re.findall('\'.*?\'', bodies)
              body_string = ' '.join(c)
              num_char = len(body_string)

              words = re.findall('\ .*?\ ', body_string)

              num_word = len(words)
              email_num += 1

              cc_N += 1
    recieved_email.append(email_num)
  return recieved_email
#--------------------------------------------------------------------------------
def create_bi_class_to(df, node_meta):
  bi_class = []
  mean_list = np.array(node_meta[:][2])
  mean_to = np.mean(mean_list)
  for node in node_meta:
    n = node[2]
    if n > mean_to:
      bi_class.append(1)
    else:
      bi_class.append(0)
  return bi_class
#--------------------------------------------------------------------------------
def create_bi_class_char(df, node_meta):
  bi_class = []
  mean_list = np.array(node_meta[:][0])
  mean_char = np.mean(mean_list)
  for node in node_meta:
    n = node[0]
    if n > mean_char:
      bi_class.append(1)
    else:
      bi_class.append(0)
  return bi_class
#--------------------------------------------------------------------------------
def create_bi_class(df, node_meta):
  bi_class = []
  mean_list = np.array(node_meta)
  mean_val = np.mean(mean_list)
  for node in node_meta:
    if node > mean_val:
      bi_class.append(1)
    else:
      bi_class.append(0)
  return bi_class
#---------------------------------------------------------------------------
def create_node_list_txt(node, node_meta, rec_list, bi_class_list):
  #takes a list that has the node list, the sent list, and the recieved list
  #this is a new version of create_nodes_txt that incorporates the node metadata
  torch.manual_seed(0)
  with open('nodes.txt','w') as f:
    for i in range(len(node)):
      #display(node)
      X_str = ' '.join([str(element.item()) for element in .5 + torch.randn((16))])
      s = ''
      for n in range(len(node_meta[0])):
        s = s + str(node_meta[i][n]) + ' '
      #f.write( str(nodes[0][i]) + ' ' + s + X_str +'\n')
      f.write(str(i) + ' ' + str(bi_class_list[i]) + ' ' + s + str(rec_list[i]) + '\n')
#---------------------------------------------------------------------------
def print_acc(predictions, bi_class):
  x = 0
  for i in range(len(bi_class)):
    if predictions.argmax(dim=1)[i] == bi_class_char_2012[i]:
      x += 1
  acc = 100 * x / len(bi_class_char_2012) 
  display(acc)
  return acc
#---------------------------------------------------------------------------
def create_nodes_txt(nodes):
  #takes a list that has the node list, the sent list, and the recieved list
  #this is a new version of create_nodes_txt that incorporates the node metadata
  torch.manual_seed(0)
  with open('nodes.txt','w') as f:
    for i in range(len(nodes[0])):
      #display(node)
      X_str = ' '.join([str(element.item()) for element in .5 + torch.randn((16))])
      s = ''
      for n in range(len(nodes)):
        s = s + str(nodes[n][i]) + ' '
      #f.write( str(nodes[0][i]) + ' ' + s + X_str +'\n')
      f.write( str(nodes[0][i]) + ' ' + s +'\n')
#---------------------------------------------------------------------------
class MyGCN(torch.nn.Module):
    def __init__(self, G):
        super(MyGCN, self).__init__()
        nb_nodes = G.nb_nodes
        A = nx.linalg.graphmatrix.adjacency_matrix(G).toarray()
        An = np.eye(nb_nodes) + A
        D = np.sum(An, axis=0)  # degree matrix
        Dinvsq = np.diag(np.sqrt(1.0 / D))  # array
        An = Dinvsq @ An @ Dinvsq # symmetric normalization
        self.H0 = G.node_metadata
        self.Y = torch.from_numpy(G.labels)

        # # 0: training set (50%); 1: testing set (50%)
        # # mask = torch.from_numpy(np.random.randint(0, 2, H0.shape[0]))
        # mask = torch.from_numpy(
        #     np.random.choice(2, self.H0.shape[0], p=[frac_train, 1.0 - frac_train])
        # )

        nb_features = G.node_metadata.shape[1]
        
        self.relu = torch.nn.ReLU();
        self.sigmoid = torch.nn.Sigmoid();
        self.tanh = torch.nn.Tanh();
        
        self.W = []        
        W0 = torch.rand(nb_features, nb_features)
        W1 = torch.rand(nb_features, 1)
                
        # Glorot Initialization
        rmax = 1 / nb_features ** 0.5
        torch.nn.init.uniform_(W0, -rmax, rmax)  # in place
        torch.nn.init.uniform_(W1, -rmax, rmax)
        
        W0.requires_grad_(True)
        W1.requires_grad_(True)
        
        self.W0 = torch.nn.Parameter(W0)
        self.w1 = torch.nn.Parameter(W1)
        
        self.An = torch.from_numpy(An).float()

        # The input can be either torch.Tensor or numpy.ndarray
        # if isinstance(self.H, np.ndarray):
        #     self.H = torch.from_numpy(self.H).float()
        if isinstance(self.H0, np.ndarray):
            self.H0 = torch.from_numpy(self.H0).float()
        if isinstance(self.Y, np.ndarray):
            self.Y = torch.from_numpy(self.Y).float()
        if isinstance(self.An, np.ndarray):
            self.An = torch.from_numpy(self.An).float()
            
    def forward(self, H0):
        """
        Semi-supervised GCN, similar to that of Kipf & Welling (2016)
         W : list of weights of different shapes
        """
        params = list(self.parameters())
        X = self.relu(self.An @ H0 @ params[0]) #self.W0)
        return self.sigmoid(self.An @ X @ params[1]) # self.W1)
#---------------------------------------------------------------------------
class BinaryCrossEntropyLoss:
    def __init__(self, mask):
        self.mask = mask
        
    def __call__(self, hidden, target):
        H = hidden
        Y = target
        """
        summation over all edges
        Training set: defined by mask[i] = 0
        H[i,0] is the probability of Y[i] == 1 (target probability)
        H[i,1] is the probability of Y[i] == 0 (target probability)

        Parameters
        ----------
        target : torch.tensor of shape [nb_nodes, 2]


        Y : torch.tensor of shape [nb_nodes]
            labels

        Notes 
        -----
        The target must be in the range [0,1].
        """
        costf = 0
            
        for i in range(Y.shape[0]):
            if self.mask[i] == 0:  # training set
                costf -= (Y[i] * torch.log(H[i,0]) + (1-Y[i]) * torch.log(1.-H[i,0]))

        return costf
#--------------------------------------------------------------------------------
def train(A, H, H0, Y, W, mask, nb_epochs, activation, lr=1.0e-2):

    # Follow https://www.analyticsvidhya.com/blog/2021/08/linear-regression-and-gradient-descent-in-pytorch/

    accuracy_count = defaultdict(list)
    loss = [cost(H0, Y, mask, activation)]

    for epoch in tqdm(range(nb_epochs)):
        H = model(A, H0, W)
        costf = cost(H, Y, mask, activation)
        loss.append(costf.item())

        if np.isnan(costf.detach().item()):
            print("costf is NaN")
            break
        with torch.no_grad():
            costf.backward(retain_graph=False)
            for w in W:
                w -= lr * w.grad
                w.grad.zero_()

        if epoch % 100 == 0:
            predict(
                A, H0, Y, W, mask, activation="sigmoid", accuracy_count=accuracy_count
            )
            pass

    return loss, accuracy_count
#--------------------------------------------------------------------------------
def predict(G, mask, accuracy_count):
    
    H0 = torch.tensor(G.node_metadata).float()
    Y = G.labels
    
    # Follow https://www.analyticsvidhya.com/blog/2021/08/linear-regression-and-gradient-descent-in-pytorch/
    H = model(H0)

    count_correct = [0,0]
    count = [0,0]
    for i in range(H.shape[0]):
        if mask[i] == 1: # test data
            count[1] += 1
            if H[i] > 0.5 and Y[i] > 0.9:
                count_correct[1] += 1
            if H[i] < 0.5 and Y[i] < 0.1:
                count_correct[1] += 1
        else:  # mask == 0, training data
            count[0] += 1
            if H[i] > 0.5 and Y[i] > 0.9:
                count_correct[0] += 1
            if H[i] < 0.5 and Y[i] < 0.1:
                count_correct[0] += 1

    if count[0] != 0 and count[1] != 0:
        accuracy_count['train'].append(count_correct[0] / count[0])
        accuracy_count['test'].append(count_correct[1] / count[1])
    else:
        accuracy_count['train'].append(0)
        accuracy_count['test'].append(0)
#-----------------------------------------------------------------------------------
def new_train(G, model, mask, loss_fn, optimizer, nb_epochs):
    H0 = torch.tensor(G.node_metadata).float()
    labels = torch.tensor(G.labels, requires_grad=False)
    losses = []
    accuracy_count = defaultdict(list)

    for epoch in range(nb_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(H0)
        loss = loss_fn(pred, labels)
        losses.append(loss.item())

        with torch.no_grad():  # should not be necessary
            loss.backward(retain_graph=False)
            optimizer.step()

        model.eval()
        predict(G, mask, accuracy_count)

    return losses, accuracy_count
#--------------------------------------------------------------------------------
def remove_outliers(df, nb_std):
    """
    return
    ------
    b: DataFrame
        df with rescaled nb_words and nb_chars. 
        Note that I should recalculate the means and std per sender. NOT DONE. 

    tab_b: DataFrame  (no longer returned)
        This data frame contains mean, median, std, mad, max, size of input data frame. 
        With this information, the original data (i.e., the input dataframe (df) with the 
        outliers removed) can be reconstructed. 

    Notes
    -----
    Consider scaling as x / max(x), which is between 0 and 1. Note that min(x) is zero.
    Apply this caling once the outliers are removed. 
    """
    df = df.copy()
    cols = ["nb_words", "nb_chars"]
    tab = df[cols].agg(["mean", "median", "std", "mad", "max", "size"])

    stdc, stdw       = tab.loc["std", ["nb_chars", "nb_words"]]
    madc, madw       = tab.loc["mad", ["nb_chars", "nb_words"]]
    meanc, meanw     = tab.loc["mean", ["nb_chars", "nb_words"]]
    medianc, medianw = tab.loc["median", ["nb_chars", "nb_words"]]
    maxc, maxw       = tab.loc["max", ["nb_chars", "nb_words"]]
    print("max nb words: ", maxw, df['nb_words'].max())
    print("max nb chars: ", maxc, df['nb_chars'].max())

    b = df[(df["nb_chars"] < (medianc + 3 * madc)) & (df["nb_words"] < (medianw + 3 * madw))]

    # Normalize by dividing by the maximum value. This works when values are positive. 
    tab_b = b[cols].agg(["mean", "median", "std", "mad", "max", "size"])
    maxc, maxw = tab_b.loc["max", ["nb_chars", "nb_words"]]

    #tab_b = b[cols].agg(["mean", "median", "std", "mad", "max", "size"])
    #meanw, meanc = tab_b.loc["mean", ["nb_words", "nb_chars"]]
    #stdw, stdc = tab_b.loc["std", ["nb_words", "nb_chars"]]

    #b['nb_words'] = (b['nb_words'] - meanw) / stdw
    #b['nb_chars'] = (b['nb_chars'] - meanc) / stdc

    print("max nb words: ", b['nb_words'].max())
    print("max nb chars: ", b['nb_chars'].max())
    print("maxw, maxc: ", maxw, maxc)

    b['nb_words'] = b['nb_words'] / maxw
    b['nb_chars'] = b['nb_chars'] / maxc

    print("max nb words: ", b['nb_words'].max())
    print("max nb chars: ", b['nb_chars'].max())


    #print("nb_words mean: ", b['nb_words'].mean())  # should be zero. Is -0.7 (wrong)
    #print("nb_words std: ", b['nb_words'].std())  # should be one. Is 0.18 (wrong)
    #print("nb_chars mean: ", b['nb_chars'].mean()) # = 4.06 (wrong)
    #print("nb_chars std: ", b['nb_chars'].std())  # = 5.5   (wrong)

    return b
#--------------------------------------------------------------------------------
def make_edges_ge(df, keep_only_emails=True, headers_as_list=False):
    """
    Construct graph edges. 

    Arguments
    ---------
    df (DataFrame)
        Dataframe containing all email data and attributes

    Return
    ------
    edges (dictionary)  sender_node => list(receiver nodes)
        The keys and values are a list of strings. 

    index_values (set)
        List of indexes of all rows where both the sender and one of the 
        receipients have an email (contains '@')

    headers_as_list (bool)
        If True, the To and CC fields are list of recipients.  
        If False,these fields have the string type: names/emails separated by semi-colons.


    Notes
    -----
    Edges are only formed from nodes that have associated emails.

    TODO
    ----
    - Add an option to return nodes and edges as integers for faster 
      execution

    """
    dfg = df.groupby('From')
    nodes_with_edges = defaultdict(set)
    index_values = set()

    for from_node in dfg.size().index:
        if keep_only_emails and not rex.match(r'.*@', from_node):
            continue
        if not type(from_node) == str:
            continue
        db = dfg.get_group(from_node)
        index_values = index_values.union(list(db.index))
        nodes_with_edges[from_node] = get_receiver_nodes(db, from_node, keep_only_emails=keep_only_emails, headers_as_list=headers_as_list)

    # Create a list of edges as a list of tuples (node1, node2)

    edges = []
    for n, e in nodes_with_edges.items():
        for receiver in e:
            edges.append([n, receiver])

    return edges, index_values
#----------------------------------------------------------------------------------
def create_nodes_edges_ge(df, keep_only_emails):
    """
    Return
    ------
    df: (DataFrame)
        DataFrame of possibly a reduced size
    """
    print("df: ", df.shape)   # [104, 31]
    df['index'] = df.index.values # records of original file read-in
    print("df: ", df.shape)   # [104, 31]
    nodes = get_unique_nodes_ge(df, verbose=False, keep_only_emails=keep_only_emails)
    edges, index_values = make_edges_ge(df, keep_only_emails=keep_only_emails)
    df = df.loc[index_values]

    # From df, construct a data frame grouped by From. One should have len(nodes) = dfg.shape[0]
    dfg = df.groupby('From')[['nb_words','nb_chars']].agg(['mean','std'])
    print(dfg.columns)
    print("dfg: ", dfg.shape)   # [23, 4] 
    print("df: ", df.shape)   # [104, 31]
    print("len(nodes): ", len(nodes))  # 69  (should equal dfg.shape[0] = 23) WHAT IS WRONG? 
    print(dfg.head(50))
    sor = sorted(nodes)
    for s in sor:
        print(s)
    return df, nodes, edges
#--------------------------------------------------------------------------------
def save_nodes_edges(nodes, edges, node_file="nodes.txt", edge_file="edges.txt"):
    edges_df = pd.DataFrame(data=edges, columns=['node1', 'node2'])
    edges_df.to_csv(edge_file, index=0)
    # Same result

    nodes_df = pd.DataFrame({'nodes':nodes})
    nodes_df.to_csv(node_file, index=0)
#--------------------------------------------------------------------------------
def plot_emails_one_year(dct_month_year, year):
    fig, axes = plt.subplots(4,3,figsize=(12,14))
    axes = axes.flatten()
    months = ['%02d' % month for month in range(1,13)]
    for i, month in enumerate(months):
        ax = axes[i]
        G = nx.DiGraph()
        month_df = dct_month_year[year, month]
        df, nodes, edges = create_nodes_edges_ge(month_df, keep_only_emails=True)
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        cralib.add_features(G, month_df)
        nx.draw_shell(G, node_size=10, ax=ax)
        ax.set_title(f"{year}/{month}, {G.nb_nodes}/{G.nb_edges}")
        # if i > 2: break
    plt.tight_layout()
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
