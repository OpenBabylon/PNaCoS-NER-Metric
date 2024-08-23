from IPython.display import HTML

def get_category(node,connections_list):
  found = connections_list.loc[connections_list['source'] == node]
  category=''
  if len(found)>0:
    category = found['source_category'][:1].values[0]
  else:
    found = connections_list.loc[connections_list['target'] == node]
    if len(found)>0:
      category = found['target_category'][:1].values[0]
  return category

def get_category_color(node,connections_list):
  category_dict = {"B-ORG":"#ffdc1c", "B-GPE": "#00a89d" , "B-OCC":"#c8bfe7", "B-PERS":"#62a0ca"}
  category = get_category(node,connections_list)
  color = category_dict.get(category)
  return color

def is_negative(source,target, connections_list):
  ret = False
  found =connections_list[(connections_list['source'] == source) & (connections_list['target'] == target) & (connections_list['relation_sentiment'] == "['negative']") ]
  #found =connections_list[(connections_list['source'] == source) & (connections_list['target'] == target) ]
  #print(len(found))
  if len(found)>0:
    ret =True
  return ret
  
  
def get_relation(source,target,connections_list ):
  ret = None
  found =connections_list[ ((connections_list['source'] == source) & (connections_list['target'] == target)) 
    |  ((connections_list['source'] == target) & (connections_list['target'] == source) )]
  if len(found)>0:
    ret = found["relation"][:1].values[0]
  return ret
  
  
def set_network_options(G): 
    G.set_options("""
        var options = {
            "nodes": {
            "size":13,
            "font": {
            "size": 11
            }
        },
        "edges": {
        "arrowStrikethrough": false,
        "color": {
        "inherit": true
        },
        "font": {
        "size": 10,
    "align": "top"
    },
    "smooth": false
    },
        "manipulation": {
        "enabled": true,
     "initiallyActive": true
     },
     "physics": {
     "barnesHut": {
     "centralGravity": 0.2,
    "springLength": 100,
    "springConstant": 0.01,
    "damping": 0.7,
    "avoidOverlap": 1
    },
    "maxVelocity": 5,
    "minVelocity": 0.47,
    "solver": "barnesHut"
    }
    }
    """)
    
def progress(value, max=100):
    return HTML("""
        <progress
            value='{value}'
            max='{max}',
            style='width: 100%'
        >
            {value}
        </progress>
    """.format(value=value, max=max))

def init_options( ) : 
    return  {
        "nodes":{
            "font":{
                "size": 20,
                "bold":True
              },
        },
        "edges":{
            "smooth":False,
            "size": 7,
        },
        "physics":{
            "barnesHut":{
                "avoidOverlap": 0.09,
                #"springConstant": 0,
                #"avoidOverlap": 0.2
            }
        },
        "interaction":{   
            "selectConnectedEdges": True,
            "navigationButtons": True,
            "hover":True, 
            "tooltipDelay": 1000,
            "hoverConnectedEdges": True,
            'selectable': True,
        },
    }

class EdgeLimitExceeded(Exception) : 
    pass 
    
def associate_weights_to_neighbors(nodes, edges, neighbor_map, show_edges, edge_count_limit = 100000) : 
    edge_count = 0 
    N_nodes = len(nodes)
    N_edges = len(edges) 
    #Associating weights to neighbors
    try : 
        for i in range(N_nodes): #Loop through nodes
            for neighbor in neighbor_map[nodes[i]]: #and neighbors
                for j in range(N_edges): #associate weights to the edge between node and neighbor
                    if edge_count < edge_count_limit :
                        #print (edges[j])
                        #edges[j]['color'] ='red'
                        if (show_edges):
                            edges[j]['label']=edges[j]['relation']
                        edge_count+=1
                    else : 
                        raise EdgeLimitExceeded
    except EdgeLimitExceeded as e : 
        pass 
    
def customize_node_shape_color_title(net, neighbor_map, connections_list, large_node_limit) :
    i=0
    for node in net.nodes:
        if i<3:
            print(node)
        i+=1
        node['title'] =""

        #node['title'] += ' Neighbors:' + '\n'.join(neighbor_map[node['id']])
        for n in neighbor_map[node['id']]:
          relation = get_relation(node['id'] , n, connections_list)
          #relation.replace("_2_")
          node['title'] += "\n" + n + " -- "  + relation
        #node['value'] = 
        node['value'] = len(neighbor_map[node['id']])
        if node['value'] >large_node_limit:
           node['shape'] = 'box'
        color = get_category_color(node['label'], connections_list)
        if color !=None:
          node['color'] = color    
    