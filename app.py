import streamlit as st
import streamlit.components.v1 as components
import json
from collections import deque
import random

# =============================================================================
# FORD-FULKERSON ALGORITHM IMPLEMENTATION
# =============================================================================

class FlowNetwork:
    """
    Represents a flow network with capacities and flows.
    """
    
    def __init__(self):
        self.nodes = set()
        self.edges = {}  # (u, v) -> capacity
        self.flow = {}   # (u, v) -> current flow
        self.source = None
        self.sink = None
    
    def add_node(self, node):
        self.nodes.add(node)
    
    def add_edge(self, u, v, capacity):
        self.nodes.add(u)
        self.nodes.add(v)
        self.edges[(u, v)] = capacity
        self.flow[(u, v)] = 0
    
    def set_source_sink(self, source, sink):
        self.source = source
        self.sink = sink
    
    def get_capacity(self, u, v):
        return self.edges.get((u, v), 0)
    
    def get_flow(self, u, v):
        return self.flow.get((u, v), 0)
    
    def get_residual_capacity(self, u, v):
        """
        Compute residual capacity cf(u, v) as defined in equation (26.2):
        - If (u,v) is an edge: cf(u,v) = c(u,v) - f(u,v)
        - If (v,u) is an edge: cf(u,v) = f(v,u)
        - Otherwise: 0
        """
        if (u, v) in self.edges:
            return self.edges[(u, v)] - self.flow[(u, v)]
        elif (v, u) in self.edges:
            return self.flow[(v, u)]
        else:
            return 0
    
    def get_residual_edges(self):
        """
        Return all edges in the residual network Gf.
        """
        residual = []
        
        # Check all possible pairs
        for u in self.nodes:
            for v in self.nodes:
                if u != v:
                    cf = self.get_residual_capacity(u, v)
                    if cf > 0:
                        # Determine if this is a forward or reverse edge
                        is_forward = (u, v) in self.edges
                        residual.append({
                            'source': u,
                            'target': v,
                            'residual_capacity': cf,
                            'is_forward': is_forward
                        })
        
        return residual
    
    def find_augmenting_path_bfs(self):
        """
        Find augmenting path using BFS (Edmonds-Karp).
        Returns (path, bottleneck) or (None, 0) if no path exists.
        """
        if self.source is None or self.sink is None:
            return None, 0
        
        parent = {self.source: None}
        visited = {self.source}
        queue = deque([self.source])
        
        while queue:
            u = queue.popleft()
            
            if u == self.sink:
                break
            
            for v in self.nodes:
                if v not in visited:
                    cf = self.get_residual_capacity(u, v)
                    if cf > 0:
                        visited.add(v)
                        parent[v] = u
                        queue.append(v)
        
        if self.sink not in parent:
            return None, 0
        
        # Reconstruct path and find bottleneck
        path = []
        bottleneck = float('inf')
        v = self.sink
        
        while parent[v] is not None:
            u = parent[v]
            cf = self.get_residual_capacity(u, v)
            bottleneck = min(bottleneck, cf)
            path.append((u, v))
            v = u
        
        path.reverse()
        return path, bottleneck
    
    def find_augmenting_path_dfs(self):
        """
        Find augmenting path using DFS.
        Returns (path, bottleneck) or (None, 0) if no path exists.
        """
        if self.source is None or self.sink is None:
            return None, 0
        
        visited = set()
        parent = {}
        
        def dfs(u):
            if u == self.sink:
                return True
            
            visited.add(u)
            
            for v in self.nodes:
                if v not in visited:
                    cf = self.get_residual_capacity(u, v)
                    if cf > 0:
                        parent[v] = u
                        if dfs(v):
                            return True
            
            return False
        
        if not dfs(self.source):
            return None, 0
        
        # Reconstruct path and find bottleneck
        path = []
        bottleneck = float('inf')
        v = self.sink
        
        while v != self.source:
            u = parent[v]
            cf = self.get_residual_capacity(u, v)
            bottleneck = min(bottleneck, cf)
            path.append((u, v))
            v = u
        
        path.reverse()
        return path, bottleneck
    
    def augment_flow(self, path, bottleneck):
        """
        Augment flow along the given path by the bottleneck amount.
        Implements lines 5-8 of FORD-FULKERSON algorithm.
        """
        for (u, v) in path:
            if (u, v) in self.edges:
                # Forward edge: increase flow
                self.flow[(u, v)] += bottleneck
            else:
                # Reverse edge: decrease flow on (v, u)
                self.flow[(v, u)] -= bottleneck
    
    def get_max_flow_value(self):
        """
        Compute current flow value |f| = sum of flows out of source.
        """
        total = 0
        for v in self.nodes:
            if (self.source, v) in self.flow:
                total += self.flow[(self.source, v)]
        return total
    
    def get_min_cut(self):
        """
        Find the min-cut (S, T) where S = vertices reachable from source in Gf.
        """
        # BFS to find all vertices reachable from source in residual network
        S = {self.source}
        queue = deque([self.source])
        
        while queue:
            u = queue.popleft()
            for v in self.nodes:
                if v not in S:
                    cf = self.get_residual_capacity(u, v)
                    if cf > 0:
                        S.add(v)
                        queue.append(v)
        
        T = self.nodes - S
        
        # Calculate cut capacity
        cut_capacity = 0
        cut_edges = []
        for u in S:
            for v in T:
                if (u, v) in self.edges:
                    cut_capacity += self.edges[(u, v)]
                    cut_edges.append((u, v))
        
        return S, T, cut_capacity, cut_edges
    
    def copy(self):
        """Create a deep copy of the network."""
        new_network = FlowNetwork()
        new_network.nodes = self.nodes.copy()
        new_network.edges = self.edges.copy()
        new_network.flow = self.flow.copy()
        new_network.source = self.source
        new_network.sink = self.sink
        return new_network


class FordFulkersonSolver:
    """
    Ford-Fulkerson solver with step-by-step execution support.
    """
    
    def __init__(self, network, strategy='bfs'):
        self.network = network.copy()
        self.strategy = strategy
        self.history = []
        self.current_path = None
        self.current_bottleneck = 0
        self.is_complete = False
        self.iteration = 0
    
    def step(self):
        """
        Execute one iteration of Ford-Fulkerson.
        Returns True if an augmenting path was found, False otherwise.
        """
        if self.is_complete:
            return False
        
        # Find augmenting path
        if self.strategy == 'bfs':
            path, bottleneck = self.network.find_augmenting_path_bfs()
        else:
            path, bottleneck = self.network.find_augmenting_path_dfs()
        
        if path is None:
            self.is_complete = True
            self.current_path = None
            self.current_bottleneck = 0
            return False
        
        # Store current state before augmentation
        self.current_path = path
        self.current_bottleneck = bottleneck
        self.iteration += 1
        
        # Record in history
        self.history.append({
            'iteration': self.iteration,
            'path': path,
            'bottleneck': bottleneck,
            'flow_before': self.network.get_max_flow_value(),
            'flow_after': self.network.get_max_flow_value() + bottleneck
        })
        
        # Augment flow
        self.network.augment_flow(path, bottleneck)
        
        return True
    
    def run_to_completion(self):
        """Run algorithm until no augmenting path exists."""
        while self.step():
            pass
        return self.network.get_max_flow_value()


# =============================================================================
# PRESET NETWORKS
# =============================================================================

def get_preset_networks():
    """Return dictionary of preset flow networks."""
    
    presets = {}
    
    # Textbook example (Figure 26.1)
    presets["Textbook Example (Fig 26.1)"] = {
        'nodes': ['s', 'v1', 'v2', 'v3', 'v4', 't'],
        'edges': [
            ('s', 'v1', 16),
            ('s', 'v2', 13),
            ('v1', 'v2', 10),
            ('v1', 'v3', 12),
            ('v2', 'v1', 4),
            ('v2', 'v4', 14),
            ('v3', 'v2', 9),
            ('v3', 't', 20),
            ('v4', 'v3', 7),
            ('v4', 't', 4),
        ],
        'source': 's',
        'sink': 't'
    }
    
    # Simple 4-node network
    presets["Simple 4-Node"] = {
        'nodes': ['s', 'a', 'b', 't'],
        'edges': [
            ('s', 'a', 10),
            ('s', 'b', 10),
            ('a', 'b', 2),
            ('a', 't', 10),
            ('b', 't', 10),
        ],
        'source': 's',
        'sink': 't'
    }
    
    # Network demonstrating need for reverse edges
    presets["Reverse Edge Demo"] = {
        'nodes': ['s', 'a', 'b', 'c', 't'],
        'edges': [
            ('s', 'a', 10),
            ('s', 'b', 10),
            ('a', 'b', 10),
            ('a', 'c', 10),
            ('b', 'c', 10),
            ('c', 't', 20),
        ],
        'source': 's',
        'sink': 't'
    }
    
    # Worst case for naive Ford-Fulkerson (Figure 26.7 style)
    presets["Worst Case Example"] = {
        'nodes': ['s', 'u', 'v', 't'],
        'edges': [
            ('s', 'u', 100),
            ('s', 'v', 100),
            ('u', 'v', 1),
            ('u', 't', 100),
            ('v', 't', 100),
        ],
        'source': 's',
        'sink': 't'
    }
    
    # Bipartite matching example
    presets["Bipartite Matching"] = {
        'nodes': ['s', 'a1', 'a2', 'a3', 'b1', 'b2', 'b3', 't'],
        'edges': [
            ('s', 'a1', 1),
            ('s', 'a2', 1),
            ('s', 'a3', 1),
            ('a1', 'b1', 1),
            ('a1', 'b2', 1),
            ('a2', 'b2', 1),
            ('a2', 'b3', 1),
            ('a3', 'b1', 1),
            ('a3', 'b3', 1),
            ('b1', 't', 1),
            ('b2', 't', 1),
            ('b3', 't', 1),
        ],
        'source': 's',
        'sink': 't'
    }
    
    # Diamond network
    presets["Diamond Network"] = {
        'nodes': ['s', 'a', 'b', 't'],
        'edges': [
            ('s', 'a', 5),
            ('s', 'b', 7),
            ('a', 't', 6),
            ('b', 't', 8),
            ('a', 'b', 3),
        ],
        'source': 's',
        'sink': 't'
    }
    
    return presets


def create_network_from_preset(preset_data):
    """Create FlowNetwork from preset data."""
    network = FlowNetwork()
    
    for node in preset_data['nodes']:
        network.add_node(node)
    
    for u, v, cap in preset_data['edges']:
        network.add_edge(u, v, cap)
    
    network.set_source_sink(preset_data['source'], preset_data['sink'])
    
    return network


def generate_random_network(num_nodes, edge_probability, min_capacity, max_capacity, seed=None):
    """Generate a random flow network."""
    if seed is not None:
        random.seed(seed)
    
    network = FlowNetwork()
    
    # Create nodes
    nodes = ['s'] + [f'v{i}' for i in range(1, num_nodes - 1)] + ['t']
    for node in nodes:
        network.add_node(node)
    
    # Create edges (ensuring path from s to t exists)
    # First, ensure a path exists
    for i in range(len(nodes) - 1):
        if random.random() < 0.7:  # High probability for sequential edges
            cap = random.randint(min_capacity, max_capacity)
            network.add_edge(nodes[i], nodes[i + 1], cap)
    
    # Add random edges
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if i < j and (u, v) not in network.edges:
                if random.random() < edge_probability:
                    cap = random.randint(min_capacity, max_capacity)
                    network.add_edge(u, v, cap)
    
    network.set_source_sink('s', 't')
    
    return network


# =============================================================================
# CYTOSCAPE.JS VISUALIZATION
# =============================================================================

def render_network_cytoscape(network, augmenting_path=None, show_residual=False,
                              S_set=None, T_set=None, cut_edges=None, height=450):
    """
    Render flow network using Cytoscape.js.
    
    Parameters:
    - network: FlowNetwork instance
    - augmenting_path: list of (u, v) tuples for current augmenting path
    - show_residual: whether to show residual network overlay
    - S_set, T_set: sets of nodes for min-cut visualization
    - cut_edges: list of edges in the min-cut
    - height: height of the visualization
    """
    
    elements = []
    
    # Determine node positions using layered layout
    node_layers = assign_layers(network)
    
    # Add nodes
    for node in network.nodes:
        node_class = []
        
        if node == network.source:
            node_class.append("source")
        elif node == network.sink:
            node_class.append("sink")
        
        if S_set is not None and node in S_set:
            node_class.append("in-S")
        elif T_set is not None and node in T_set:
            node_class.append("in-T")
        
        elements.append({
            "data": {
                "id": node,
                "label": node,
                "layer": node_layers.get(node, 1)
            },
            "classes": " ".join(node_class)
        })
    
    # Track which edges are on augmenting path
    aug_path_set = set(augmenting_path) if augmenting_path else set()
    cut_edges_set = set(cut_edges) if cut_edges else set()
    
    # Add original edges
    for (u, v), capacity in network.edges.items():
        flow = network.get_flow(u, v)
        
        edge_class = []
        if (u, v) in aug_path_set:
            edge_class.append("augmenting")
        if flow == capacity and capacity > 0:
            edge_class.append("saturated")
        if (u, v) in cut_edges_set:
            edge_class.append("cut-edge")
        
        elements.append({
            "data": {
                "id": f"{u}->{v}",
                "source": u,
                "target": v,
                "label": f"{flow}/{capacity}",
                "flow": flow,
                "capacity": capacity,
                "edgeType": "original"
            },
            "classes": " ".join(edge_class)
        })
    
    # Add residual edges if requested
    if show_residual:
        for edge in network.get_residual_edges():
            u, v = edge['source'], edge['target']
            cf = edge['residual_capacity']
            
            # Skip if this is an original edge with residual shown
            if (u, v) in network.edges:
                continue
            
            # This is a reverse edge
            edge_class = ["residual"]
            if (u, v) in aug_path_set:
                edge_class.append("augmenting")
            
            elements.append({
                "data": {
                    "id": f"{u}~>{v}",
                    "source": u,
                    "target": v,
                    "label": str(cf),
                    "residual": cf,
                    "edgeType": "residual"
                },
                "classes": " ".join(edge_class)
            })
    
    elements_json = json.dumps(elements)
    
    # Build bottleneck info for path display
    path_info = ""
    if augmenting_path:
        path_str = " → ".join([augmenting_path[0][0]] + [e[1] for e in augmenting_path])
        bottleneck = min(network.get_residual_capacity(u, v) for u, v in augmenting_path)
        path_info = f"Path: {path_str}<br>Bottleneck: {bottleneck}"
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.23.0/cytoscape.min.js"></script>
        <script src="https://unpkg.com/dagre@0.8.5/dist/dagre.min.js"></script>
        <script src="https://unpkg.com/cytoscape-dagre@2.5.0/cytoscape-dagre.js"></script>
        <style>
            body {{
                margin: 0;
                padding: 0;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }}
            #cy {{
                width: 100%;
                height: {height - 50}px;
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 8px;
            }}
            #info {{
                padding: 10px;
                font-size: 14px;
                color: #495057;
                background: #e9ecef;
                border-radius: 0 0 8px 8px;
                min-height: 20px;
            }}
            .legend {{
                display: flex;
                gap: 20px;
                flex-wrap: wrap;
                align-items: center;
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                gap: 5px;
            }}
            .legend-color {{
                width: 20px;
                height: 4px;
                border-radius: 2px;
            }}
            .legend-node {{
                width: 12px;
                height: 12px;
                border-radius: 50%;
            }}
        </style>
    </head>
    <body>
        <div id="cy"></div>
        <div id="info">
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-node" style="background: #28a745;"></div>
                    <span>Source</span>
                </div>
                <div class="legend-item">
                    <div class="legend-node" style="background: #dc3545;"></div>
                    <span>Sink</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #007bff;"></div>
                    <span>Augmenting Path</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #dc3545;"></div>
                    <span>Saturated Edge</span>
                </div>
                {"<div class='legend-item'><div class='legend-color' style='background: #17a2b8; border: 1px dashed #17a2b8;'></div><span>Residual Edge</span></div>" if show_residual else ""}
                {f"<div class='legend-item' style='margin-left: auto;'><strong>{path_info}</strong></div>" if path_info else ""}
            </div>
        </div>
        <script>
            var cy = cytoscape({{
                container: document.getElementById('cy'),
                elements: {elements_json},
                style: [
                    {{
                        selector: 'node',
                        style: {{
                            'background-color': '#6c757d',
                            'label': 'data(label)',
                            'text-valign': 'center',
                            'text-halign': 'center',
                            'color': '#fff',
                            'font-size': '14px',
                            'font-weight': 'bold',
                            'width': '45px',
                            'height': '45px',
                            'border-width': 3,
                            'border-color': '#495057'
                        }}
                    }},
                    {{
                        selector: 'node.source',
                        style: {{
                            'background-color': '#28a745',
                            'border-color': '#1e7e34'
                        }}
                    }},
                    {{
                        selector: 'node.sink',
                        style: {{
                            'background-color': '#dc3545',
                            'border-color': '#bd2130'
                        }}
                    }},
                    {{
                        selector: 'node.in-S',
                        style: {{
                            'border-color': '#ffc107',
                            'border-width': 5
                        }}
                    }},
                    {{
                        selector: 'node.in-T',
                        style: {{
                            'border-color': '#17a2b8',
                            'border-width': 5
                        }}
                    }},
                    {{
                        selector: 'edge',
                        style: {{
                            'width': 3,
                            'line-color': '#adb5bd',
                            'target-arrow-color': '#adb5bd',
                            'target-arrow-shape': 'triangle',
                            'curve-style': 'bezier',
                            'label': 'data(label)',
                            'font-size': '12px',
                            'text-background-color': '#fff',
                            'text-background-opacity': 0.9,
                            'text-background-padding': '3px',
                            'text-margin-y': -10,
                            'color': '#212529'
                        }}
                    }},
                    {{
                        selector: 'edge.augmenting',
                        style: {{
                            'line-color': '#007bff',
                            'target-arrow-color': '#007bff',
                            'width': 5,
                            'z-index': 999
                        }}
                    }},
                    {{
                        selector: 'edge.saturated',
                        style: {{
                            'line-color': '#dc3545',
                            'target-arrow-color': '#dc3545'
                        }}
                    }},
                    {{
                        selector: 'edge.residual',
                        style: {{
                            'line-color': '#17a2b8',
                            'target-arrow-color': '#17a2b8',
                            'line-style': 'dashed',
                            'width': 2,
                            'curve-style': 'bezier',
                            'control-point-step-size': 50
                        }}
                    }},
                    {{
                        selector: 'edge.cut-edge',
                        style: {{
                            'line-color': '#ffc107',
                            'target-arrow-color': '#ffc107',
                            'width': 6
                        }}
                    }}
                ],
                layout: {{
                    name: 'dagre',
                    rankDir: 'LR',
                    nodeSep: 70,
                    rankSep: 120,
                    padding: 30
                }}
            }});
            
            // Enable node dragging
            cy.nodes().grabify();
            
            // Fit to container
            cy.fit(undefined, 30);
        </script>
    </body>
    </html>
    """
    
    components.html(html_code, height=height)


def assign_layers(network):
    """
    Assign layers to nodes for visualization.
    Uses BFS from source.
    """
    if not network.source:
        return {node: 0 for node in network.nodes}
    
    layers = {network.source: 0}
    queue = deque([network.source])
    
    while queue:
        u = queue.popleft()
        for v in network.nodes:
            if v not in layers:
                if (u, v) in network.edges or (v, u) in network.edges:
                    layers[v] = layers[u] + 1
                    queue.append(v)
    
    # Assign remaining nodes
    max_layer = max(layers.values()) if layers else 0
    for node in network.nodes:
        if node not in layers:
            layers[node] = max_layer // 2
    
    # Ensure sink is at the end
    if network.sink in layers:
        max_layer = max(layers.values())
        layers[network.sink] = max_layer
    
    return layers


# =============================================================================
# STREAMLIT APP
# =============================================================================

def initialize_session_state():
    """Initialize session state variables."""
    
    if 'network' not in st.session_state:
        st.session_state.network = None
    
    if 'solver' not in st.session_state:
        st.session_state.solver = None
    
    if 'manual_edges' not in st.session_state:
        st.session_state.manual_edges = []
    
    if 'manual_nodes' not in st.session_state:
        st.session_state.manual_nodes = ['s', 't']


def main():
    st.set_page_config(
        page_title="Ford-Fulkerson Visualizer",
        page_icon="🌊",
        layout="wide"
    )
    
    initialize_session_state()
    
    st.title("🌊 Ford-Fulkerson Maximum Flow Visualizer")
    st.markdown("""
    This interactive tool visualizes the **Ford-Fulkerson method** for finding 
    maximum flow in a flow network. Watch how augmenting paths are found and 
    flow is pushed through the network step by step.
    """)
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Network input method
        st.subheader("1. Network Input")
        input_method = st.radio(
            "Select input method:",
            ["Preset Examples", "Manual Entry", "Random Generation"],
            key="input_method"
        )
        
        network = None
        
        if input_method == "Preset Examples":
            presets = get_preset_networks()
            preset_name = st.selectbox(
                "Choose a preset network:",
                list(presets.keys())
            )
            
            if st.button("Load Preset", type="primary"):
                network = create_network_from_preset(presets[preset_name])
                st.session_state.network = network
                st.session_state.solver = None
                st.success(f"Loaded: {preset_name}")
        
        elif input_method == "Manual Entry":
            st.markdown("**Add Nodes:**")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                new_node = st.text_input("Node name:", key="new_node_input")
            with col2:
                st.write("")
                st.write("")
                if st.button("Add", key="add_node_btn"):
                    if new_node and new_node not in st.session_state.manual_nodes:
                        st.session_state.manual_nodes.append(new_node)
                        st.rerun()
            
            st.write(f"Current nodes: {', '.join(st.session_state.manual_nodes)}")
            
            st.markdown("**Add Edges:**")
            
            if len(st.session_state.manual_nodes) >= 2:
                col1, col2, col3 = st.columns(3)
                with col1:
                    edge_from = st.selectbox("From:", st.session_state.manual_nodes, key="edge_from")
                with col2:
                    edge_to = st.selectbox("To:", st.session_state.manual_nodes, key="edge_to")
                with col3:
                    edge_cap = st.number_input("Capacity:", min_value=1, value=10, key="edge_cap")
                
                if st.button("Add Edge", key="add_edge_btn"):
                    if edge_from != edge_to:
                        st.session_state.manual_edges.append((edge_from, edge_to, edge_cap))
                        st.rerun()
            
            if st.session_state.manual_edges:
                st.markdown("**Current Edges:**")
                for i, (u, v, c) in enumerate(st.session_state.manual_edges):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"{u} → {v} (capacity: {c})")
                    with col2:
                        if st.button("❌", key=f"del_edge_{i}"):
                            st.session_state.manual_edges.pop(i)
                            st.rerun()
            
            source = st.selectbox("Source node:", st.session_state.manual_nodes, key="source_select")
            sink_options = [n for n in st.session_state.manual_nodes if n != source]
            sink = st.selectbox("Sink node:", sink_options, key="sink_select") if sink_options else None
            
            if st.button("Build Network", type="primary") and sink:
                network = FlowNetwork()
                for node in st.session_state.manual_nodes:
                    network.add_node(node)
                for u, v, c in st.session_state.manual_edges:
                    network.add_edge(u, v, c)
                network.set_source_sink(source, sink)
                st.session_state.network = network
                st.session_state.solver = None
                st.success("Network built successfully!")
            
            if st.button("Clear All", key="clear_manual"):
                st.session_state.manual_nodes = ['s', 't']
                st.session_state.manual_edges = []
                st.session_state.network = None
                st.session_state.solver = None
                st.rerun()
        
        else:  # Random Generation
            num_nodes = st.slider("Number of nodes:", 4, 12, 6)
            edge_prob = st.slider("Edge probability:", 0.2, 0.8, 0.4)
            min_cap = st.number_input("Min capacity:", 1, 50, 1)
            max_cap = st.number_input("Max capacity:", min_cap, 100, 20)
            seed = st.number_input("Random seed (0 for random):", 0, 9999, 42)
            
            if st.button("Generate Network", type="primary"):
                actual_seed = seed if seed > 0 else None
                network = generate_random_network(num_nodes, edge_prob, min_cap, max_cap, actual_seed)
                st.session_state.network = network
                st.session_state.solver = None
                st.success("Network generated!")
        
        st.divider()
        
        # Algorithm configuration
        st.subheader("2. Algorithm Settings")
        
        strategy = st.radio(
            "Augmenting path strategy:",
            ["BFS (Edmonds-Karp)", "DFS"],
            help="BFS finds shortest paths, DFS may find longer paths"
        )
        
        strategy_key = 'bfs' if 'BFS' in strategy else 'dfs'
        
        st.divider()
        
        # Visualization options
        st.subheader("3. Display Options")
        
        show_residual = st.checkbox("Show residual edges", value=True,
                                    help="Display reverse edges in the residual network")
        
        show_cut = st.checkbox("Show min-cut (when complete)", value=True,
                               help="Highlight the minimum cut after algorithm completes")
    
    # Main content area
    if st.session_state.network is None:
        st.info("👈 Please configure and load a network from the sidebar to begin.")
        
        # Show example
        st.subheader("Example: Textbook Network")
        example_network = create_network_from_preset(get_preset_networks()["Textbook Example (Fig 26.1)"])
        render_network_cytoscape(example_network, show_residual=False, height=400)
        
        return
    
    network = st.session_state.network
    
    # Control buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Reset", type="secondary", use_container_width=True):
            # Reset flow to zero
            for edge in network.flow:
                network.flow[edge] = 0
            st.session_state.solver = None
            st.rerun()
    
    with col2:
        if st.button("▶️ Next Step", type="primary", use_container_width=True):
            if st.session_state.solver is None:
                st.session_state.solver = FordFulkersonSolver(network, strategy_key)
            
            if not st.session_state.solver.is_complete:
                st.session_state.solver.step()
                # Sync network state
                st.session_state.network = st.session_state.solver.network
            st.rerun()
    
    with col3:
        if st.button("⏭️ Run to Completion", type="primary", use_container_width=True):
            if st.session_state.solver is None:
                st.session_state.solver = FordFulkersonSolver(network, strategy_key)
            
            st.session_state.solver.run_to_completion()
            st.session_state.network = st.session_state.solver.network
            st.rerun()
    
    with col4:
        if st.session_state.solver:
            if st.session_state.solver.is_complete:
                st.success(f"✅ Complete! Max Flow: {network.get_max_flow_value()}")
            else:
                st.info(f"Iteration: {st.session_state.solver.iteration}")
    
    # Main visualization
    st.subheader("Flow Network")
    
    solver = st.session_state.solver
    augmenting_path = None
    S_set, T_set, cut_edges = None, None, None
    
    if solver and not solver.is_complete and solver.current_path:
        augmenting_path = solver.current_path
    
    if solver and solver.is_complete and show_cut:
        S_set, T_set, cut_capacity, cut_edges = network.get_min_cut()
    
    render_network_cytoscape(
        network,
        augmenting_path=augmenting_path,
        show_residual=show_residual,
        S_set=S_set,
        T_set=T_set,
        cut_edges=cut_edges,
        height=500
    )
    
    # Information panels
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Network Statistics")
        
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        
        with stats_col1:
            st.metric("Nodes", len(network.nodes))
        with stats_col2:
            st.metric("Edges", len(network.edges))
        with stats_col3:
            st.metric("Current Flow", network.get_max_flow_value())
        
        # Edge details
        with st.expander("Edge Details", expanded=False):
            edge_data = []
            for (u, v), cap in network.edges.items():
                flow = network.get_flow(u, v)
                residual = cap - flow
                edge_data.append({
                    "Edge": f"{u} → {v}",
                    "Capacity": cap,
                    "Flow": flow,
                    "Residual": residual,
                    "Saturated": "✓" if flow == cap else ""
                })
            st.table(edge_data)
    
    with col2:
        st.subheader("📜 Algorithm History")
        
        if solver and solver.history:
            for record in reversed(solver.history[-10:]):  # Show last 10
                path_str = " → ".join([record['path'][0][0]] + [e[1] for e in record['path']])
                st.markdown(f"""
                **Iteration {record['iteration']}:**  
                Path: `{path_str}`  
                Bottleneck: {record['bottleneck']}  
                Flow: {record['flow_before']} → {record['flow_after']}
                """)
                st.divider()
        else:
            st.info("No iterations yet. Click 'Next Step' to begin.")
        
        # Min-cut information
        if solver and solver.is_complete:
            S, T, cut_cap, edges = network.get_min_cut()
            
            st.subheader("✂️ Minimum Cut")
            st.markdown(f"""
            - **S** (source side): {{{', '.join(sorted(S))}}}
            - **T** (sink side): {{{', '.join(sorted(T))}}}
            - **Cut capacity**: {cut_cap}
            - **Cut edges**: {', '.join([f'{u}→{v}' for u, v in edges])}
            
            ✅ **Max-Flow Min-Cut Theorem verified:**  
            Max Flow ({network.get_max_flow_value()}) = Min Cut ({cut_cap})
            """)
    
    # Educational content
    with st.expander("📚 About the Ford-Fulkerson Method", expanded=False):
        st.markdown("""
        ### The Ford-Fulkerson Method
        
        The Ford-Fulkerson method finds the **maximum flow** from a source node *s* 
        to a sink node *t* in a flow network.
        
        #### Key Concepts:
        
        1. **Residual Network (Gf):** Shows how flow can still be changed
           - **Forward edges:** remaining capacity = c(u,v) - f(u,v)
           - **Backward edges:** allow "undoing" flow = f(v,u)
        
        2. **Augmenting Path:** A path from *s* to *t* in the residual network
           - **Bottleneck:** minimum residual capacity along the path
        
        3. **Min-Cut:** A partition (S, T) where s ∈ S and t ∈ T
           - **Cut capacity:** sum of capacities of edges from S to T
        
        #### The Algorithm:
        
        ```
        FORD-FULKERSON(G, s, t):
            Initialize flow f to 0
            While there exists an augmenting path p in Gf:
                Find bottleneck capacity cf(p)
                Augment flow along p by cf(p)
            Return f
        ```
        
        #### Max-Flow Min-Cut Theorem:
        
        The value of the maximum flow equals the capacity of the minimum cut.
        
        When the algorithm terminates (no augmenting path exists), we have found 
        both the maximum flow and a minimum cut!
        
        #### Path Selection Strategies:
        
        - **BFS (Edmonds-Karp):** O(VE²) time - finds shortest augmenting paths
        - **DFS:** O(E|f*|) time - may find longer paths, can be slower
        """)


if __name__ == "__main__":
    main()
