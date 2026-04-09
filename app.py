import streamlit as st
import streamlit.components.v1 as components
import json
from collections import deque
import random

# =============================================================================
# FORD-FULKERSON ALGORITHM
# =============================================================================

class FlowNetwork:
    def __init__(self):
        self.nodes = []  # Use list to maintain order
        self.node_set = set()
        self.edges = {}
        self.flow = {}
        self.source = None
        self.sink = None
    
    def add_node(self, node):
        if node not in self.node_set:
            self.nodes.append(node)
            self.node_set.add(node)
    
    def add_edge(self, u, v, capacity):
        self.add_node(u)
        self.add_node(v)
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
        Residual capacity:
        - Forward edge: c(u,v) - f(u,v)
        - Reverse edge: f(v,u)
        """
        if (u, v) in self.edges:
            return self.edges[(u, v)] - self.flow[(u, v)]
        elif (v, u) in self.edges:
            return self.flow[(v, u)]
        return 0
    
    def get_neighbors(self, u, prefer_longer_paths=False):
        """
        Get neighbors with positive residual capacity.
        If prefer_longer_paths=True (for DFS worst case demo),
        sort to prefer intermediate nodes over sink.
        """
        neighbors = []
        for v in self.nodes:
            if v != u and self.get_residual_capacity(u, v) > 0:
                neighbors.append(v)
        
        if prefer_longer_paths:
            # Sort to prefer non-sink nodes first (creates longer paths)
            # This demonstrates worst-case DFS behavior
            neighbors.sort(key=lambda x: (x == self.sink, x))
        
        return neighbors
    
    def find_augmenting_path_bfs(self):
        """
        BFS finds shortest augmenting path (Edmonds-Karp).
        Guarantees O(VE²) time complexity.
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
                if v not in visited and self.get_residual_capacity(u, v) > 0:
                    visited.add(v)
                    parent[v] = u
                    queue.append(v)
        
        if self.sink not in parent:
            return None, 0
        
        # Reconstruct path
        path = []
        bottleneck = float('inf')
        v = self.sink
        
        while parent[v] is not None:
            u = parent[v]
            bottleneck = min(bottleneck, self.get_residual_capacity(u, v))
            path.append((u, v))
            v = u
        
        path.reverse()
        return path, bottleneck
    
    def find_augmenting_path_dfs(self, prefer_longer_paths=True):
        """
        DFS finds any augmenting path.
        With prefer_longer_paths=True, demonstrates worst-case O(E|f*|) behavior.
        """
        if self.source is None or self.sink is None:
            return None, 0
        
        visited = set()
        parent = {}
        
        def dfs(u):
            if u == self.sink:
                return True
            
            visited.add(u)
            
            # Get neighbors - prefer longer paths for worst case demo
            neighbors = self.get_neighbors(u, prefer_longer_paths)
            
            for v in neighbors:
                if v not in visited:
                    parent[v] = u
                    if dfs(v):
                        return True
            
            return False
        
        if not dfs(self.source):
            return None, 0
        
        # Reconstruct path
        path = []
        bottleneck = float('inf')
        v = self.sink
        
        while v != self.source:
            u = parent[v]
            bottleneck = min(bottleneck, self.get_residual_capacity(u, v))
            path.append((u, v))
            v = u
        
        path.reverse()
        return path, bottleneck
    
    def augment_flow(self, path, bottleneck):
        """
        Augment flow along path.
        Forward edges: increase flow
        Reverse edges: decrease flow (cancellation)
        """
        for (u, v) in path:
            if (u, v) in self.edges:
                # Forward edge
                self.flow[(u, v)] += bottleneck
            else:
                # Reverse edge - decrease flow on original edge
                self.flow[(v, u)] -= bottleneck
    
    def get_max_flow_value(self):
        """Total flow out of source."""
        return sum(self.flow.get((self.source, v), 0) for v in self.nodes)
    
    def get_min_cut(self):
        """
        Find min-cut: S = vertices reachable from source in residual network.
        """
        S = {self.source}
        queue = deque([self.source])
        
        while queue:
            u = queue.popleft()
            for v in self.nodes:
                if v not in S and self.get_residual_capacity(u, v) > 0:
                    S.add(v)
                    queue.append(v)
        
        T = set(self.nodes) - S
        cut_edges = [(u, v) for u in S for v in T if (u, v) in self.edges]
        cut_capacity = sum(self.edges[(u, v)] for u, v in cut_edges)
        
        return S, T, cut_capacity, cut_edges
    
    def copy(self):
        new = FlowNetwork()
        new.nodes = self.nodes.copy()
        new.node_set = self.node_set.copy()
        new.edges = self.edges.copy()
        new.flow = self.flow.copy()
        new.source = self.source
        new.sink = self.sink
        return new


def build_animation_data(network, strategy='bfs'):
    """
    Run Ford-Fulkerson and build complete animation timeline.
    """
    network = network.copy()
    
    data = {
        'nodes': list(network.nodes),
        'source': network.source,
        'sink': network.sink,
        'strategy': strategy.upper(),
        'edges': [
            {
                'id': f"{u}-{v}",
                'source': u,
                'target': v,
                'capacity': cap
            }
            for (u, v), cap in network.edges.items()
        ],
        'iterations': []
    }
    
    iteration_num = 0
    max_iterations = 1000  # Safety limit
    
    while iteration_num < max_iterations:
        # Find augmenting path using selected strategy
        if strategy == 'bfs':
            path, bottleneck = network.find_augmenting_path_bfs()
        else:
            path, bottleneck = network.find_augmenting_path_dfs(prefer_longer_paths=True)
        
        if path is None:
            break
        
        iteration_num += 1
        
        # Check if path uses reverse edges
        uses_reverse = any((u, v) not in network.edges for u, v in path)
        
        iteration = {
            'iteration': iteration_num,
            'path': [{'source': u, 'target': v, 'is_reverse': (u, v) not in network.edges} 
                    for u, v in path],
            'bottleneck': bottleneck,
            'uses_reverse_edge': uses_reverse,
            'flow_before': {f"{u}-{v}": network.flow[(u, v)] for u, v in network.edges},
        }
        
        # Augment flow
        network.augment_flow(path, bottleneck)
        
        iteration['flow_after'] = {f"{u}-{v}": network.flow[(u, v)] for u, v in network.edges}
        iteration['total_flow'] = network.get_max_flow_value()
        
        data['iterations'].append(iteration)
    
    # Final state
    S, T, cut_cap, cut_edges = network.get_min_cut()
    data['final'] = {
        'max_flow': network.get_max_flow_value(),
        'S': list(S),
        'T': list(T),
        'cut_capacity': cut_cap,
        'cut_edges': [{'source': u, 'target': v} for u, v in cut_edges],
        'total_iterations': iteration_num
    }
    
    return data


# =============================================================================
# PRESET NETWORKS
# =============================================================================

def get_presets():
    return {
        "Textbook Example (CLRS Fig 26.1)": {
            'nodes': ['s', 'v1', 'v2', 'v3', 'v4', 't'],
            'edges': [
                ('s', 'v1', 16), ('s', 'v2', 13),
                ('v1', 'v2', 10), ('v1', 'v3', 12),
                ('v2', 'v1', 4), ('v2', 'v4', 14),
                ('v3', 'v2', 9), ('v3', 't', 20),
                ('v4', 'v3', 7), ('v4', 't', 4),
            ],
            'source': 's', 'sink': 't',
            'description': 'Classic example from Introduction to Algorithms textbook.'
        },
        "Simple Diamond": {
            'nodes': ['s', 'a', 'b', 't'],
            'edges': [
                ('s', 'a', 10), ('s', 'b', 10),
                ('a', 'b', 2), ('a', 't', 10), ('b', 't', 10),
            ],
            'source': 's', 'sink': 't',
            'description': 'Simple network showing flow distribution through two paths.'
        },
        "Worst Case (BFS vs DFS)": {
            # Node order matters for DFS worst case!
            # We want DFS to explore u→v before u→t
            'nodes': ['s', 'u', 'v', 't'],
            'edges': [
                ('s', 'u', 100), ('s', 'v', 100),
                ('u', 'v', 1),  # The problematic edge!
                ('u', 't', 100), ('v', 't', 100),
            ],
            'source': 's', 'sink': 't',
            'description': '''WORST CASE DEMONSTRATION:
            
• BFS: Finds shortest paths s→u→t and s→v→t (2 iterations, flow=200)
• DFS: May alternate through u→v edge (up to 200 iterations!)

The u→v edge with capacity 1 creates the problem:
- DFS might find s→u→v→t first (bottleneck=1)
- Then s→v→u→t using reverse edge (bottleneck=1)
- Repeating 200 times instead of 2!'''
        },
        "Reverse Edge Demo": {
            'nodes': ['s', 'a', 'b', 'c', 't'],
            'edges': [
                ('s', 'a', 10), ('s', 'b', 5),
                ('a', 'b', 15), ('b', 'c', 15),
                ('a', 't', 5), ('c', 't', 15),
            ],
            'source': 's', 'sink': 't',
            'description': 'Shows how reverse edges enable flow rerouting.'
        },
        "Bipartite Matching": {
            'nodes': ['s', 'a1', 'a2', 'a3', 'b1', 'b2', 'b3', 't'],
            'edges': [
                ('s', 'a1', 1), ('s', 'a2', 1), ('s', 'a3', 1),
                ('a1', 'b1', 1), ('a1', 'b2', 1),
                ('a2', 'b2', 1), ('a2', 'b3', 1),
                ('a3', 'b1', 1), ('a3', 'b3', 1),
                ('b1', 't', 1), ('b2', 't', 1), ('b3', 't', 1),
            ],
            'source': 's', 'sink': 't',
            'description': 'Maximum bipartite matching as max-flow. Each unit of flow = one match.'
        },
        "Linear Chain": {
            'nodes': ['s', 'a', 'b', 'c', 't'],
            'edges': [
                ('s', 'a', 10), ('a', 'b', 8),
                ('b', 'c', 6), ('c', 't', 10),
            ],
            'source': 's', 'sink': 't',
            'description': 'Simple chain - bottleneck determines max flow.'
        },
    }


def create_network_from_preset(preset):
    network = FlowNetwork()
    # Add nodes in order (important for DFS worst case)
    for node in preset['nodes']:
        network.add_node(node)
    for u, v, cap in preset['edges']:
        network.add_edge(u, v, cap)
    network.set_source_sink(preset['source'], preset['sink'])
    return network


def generate_random_network(num_nodes, edge_prob, min_cap, max_cap, seed=None):
    if seed is not None:
        random.seed(seed)
    
    network = FlowNetwork()
    nodes = ['s'] + [f'v{i}' for i in range(1, num_nodes - 1)] + ['t']
    
    for node in nodes:
        network.add_node(node)
    
    # Ensure path exists
    for i in range(len(nodes) - 1):
        if random.random() < 0.7:
            cap = random.randint(min_cap, max_cap)
            network.add_edge(nodes[i], nodes[i + 1], cap)
    
    # Add random edges
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if i < j and (u, v) not in network.edges:
                if random.random() < edge_prob:
                    cap = random.randint(min_cap, max_cap)
                    network.add_edge(u, v, cap)
    
    network.set_source_sink('s', 't')
    return network


# =============================================================================
# ANIMATION COMPONENT
# =============================================================================

def render_auto_animation(animation_data, step_duration=2000, height=750):
    """Render self-playing animation with full algorithm visualization."""
    
    data_json = json.dumps(animation_data)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.23.0/cytoscape.min.js"></script>
        <script src="https://unpkg.com/dagre@0.8.5/dist/dagre.min.js"></script>
        <script src="https://unpkg.com/cytoscape-dagre@2.5.0/cytoscape-dagre.js"></script>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                color: #fff;
                overflow: hidden;
            }}
            #container {{
                display: flex;
                flex-direction: column;
                height: {height}px;
            }}
            #header {{
                padding: 15px 20px;
                background: rgba(255,255,255,0.05);
                border-bottom: 1px solid rgba(255,255,255,0.1);
                display: flex;
                justify-content: space-between;
                align-items: center;
                flex-wrap: wrap;
                gap: 10px;
            }}
            #title {{
                font-size: 20px;
                font-weight: 600;
                color: #4fc3f7;
            }}
            #strategy-badge {{
                background: #ff9800;
                color: #000;
                padding: 4px 12px;
                border-radius: 12px;
                font-size: 12px;
                font-weight: 700;
            }}
            #stats {{
                display: flex;
                gap: 20px;
            }}
            .stat {{
                text-align: center;
            }}
            .stat-value {{
                font-size: 24px;
                font-weight: 700;
                color: #4fc3f7;
            }}
            .stat-label {{
                font-size: 10px;
                text-transform: uppercase;
                letter-spacing: 1px;
                color: rgba(255,255,255,0.6);
            }}
            #cy {{
                flex: 1;
                background: radial-gradient(ellipse at center, #1a1a2e 0%, #0f0f1a 100%);
            }}
            #status-bar {{
                padding: 15px 20px;
                background: rgba(0,0,0,0.3);
                border-top: 1px solid rgba(255,255,255,0.1);
            }}
            #phase-indicator {{
                display: flex;
                align-items: center;
                gap: 15px;
                margin-bottom: 12px;
            }}
            #phase-icon {{
                width: 40px;
                height: 40px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 20px;
                background: #4fc3f7;
                animation: pulse 1s ease-in-out infinite;
            }}
            @keyframes pulse {{
                0%, 100% {{ transform: scale(1); opacity: 1; }}
                50% {{ transform: scale(1.1); opacity: 0.8; }}
            }}
            #phase-text {{
                flex: 1;
            }}
            #phase-title {{
                font-size: 15px;
                font-weight: 600;
                margin-bottom: 3px;
            }}
            #phase-description {{
                font-size: 12px;
                color: rgba(255,255,255,0.7);
            }}
            #progress-container {{
                height: 6px;
                background: rgba(255,255,255,0.1);
                border-radius: 3px;
                overflow: hidden;
            }}
            #progress-bar {{
                height: 100%;
                background: linear-gradient(90deg, #4fc3f7, #00e676);
                border-radius: 3px;
                transition: width 0.3s ease;
                width: 0%;
            }}
            #path-display {{
                margin-top: 12px;
                padding: 10px 15px;
                background: rgba(79, 195, 247, 0.15);
                border: 1px solid rgba(79, 195, 247, 0.3);
                border-radius: 8px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 13px;
                display: none;
            }}
            #path-display.visible {{
                display: block;
            }}
            .path-arrow {{
                color: #4fc3f7;
                margin: 0 6px;
            }}
            .path-arrow.reverse {{
                color: #ff9800;
            }}
            .path-node {{
                background: rgba(79, 195, 247, 0.3);
                padding: 2px 8px;
                border-radius: 4px;
            }}
            .path-node.reverse {{
                background: rgba(255, 152, 0, 0.3);
            }}
            .bottleneck {{
                color: #ffeb3b;
                margin-left: 15px;
                font-weight: 600;
            }}
            .reverse-indicator {{
                color: #ff9800;
                margin-left: 10px;
                font-size: 11px;
            }}
            #legend {{
                position: absolute;
                top: 70px;
                right: 15px;
                background: rgba(0,0,0,0.6);
                padding: 12px;
                border-radius: 8px;
                font-size: 11px;
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                gap: 8px;
                margin-bottom: 6px;
            }}
            .legend-color {{
                width: 18px;
                height: 3px;
                border-radius: 2px;
            }}
            .legend-node {{
                width: 12px;
                height: 12px;
                border-radius: 50%;
            }}
            #completion {{
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0,0,0,0.9);
                display: none;
                align-items: center;
                justify-content: center;
                flex-direction: column;
                gap: 20px;
                z-index: 1000;
                padding: 20px;
            }}
            #completion.visible {{
                display: flex;
            }}
            #completion h2 {{
                font-size: 28px;
                color: #00e676;
            }}
            #completion-stats {{
                display: flex;
                gap: 30px;
                flex-wrap: wrap;
                justify-content: center;
            }}
            #completion .stat-value {{
                font-size: 42px;
                color: #00e676;
            }}
            #theorem-box {{
                background: rgba(0, 230, 118, 0.15);
                border: 1px solid rgba(0, 230, 118, 0.3);
                padding: 15px 25px;
                border-radius: 8px;
                text-align: center;
            }}
            #restart-btn {{
                margin-top: 15px;
                padding: 12px 30px;
                font-size: 16px;
                font-weight: 600;
                background: #4fc3f7;
                color: #1a1a2e;
                border: none;
                border-radius: 25px;
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
            }}
            #restart-btn:hover {{
                transform: translateY(-2px);
                box-shadow: 0 5px 20px rgba(79, 195, 247, 0.4);
            }}
            .particle {{
                position: absolute;
                width: 10px;
                height: 10px;
                background: #4fc3f7;
                border-radius: 50%;
                pointer-events: none;
                box-shadow: 0 0 10px #4fc3f7, 0 0 20px #4fc3f7;
                z-index: 100;
            }}
            .particle.reverse {{
                background: #ff9800;
                box-shadow: 0 0 10px #ff9800, 0 0 20px #ff9800;
            }}
        </style>
    </head>
    <body>
        <div id="container">
            <div id="header">
                <div style="display: flex; align-items: center; gap: 15px;">
                    <div id="title">🌊 Ford-Fulkerson Algorithm</div>
                    <div id="strategy-badge"></div>
                </div>
                <div id="stats">
                    <div class="stat">
                        <div class="stat-value" id="iteration-value">0</div>
                        <div class="stat-label">Iteration</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="flow-value">0</div>
                        <div class="stat-label">Flow</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="total-iterations">-</div>
                        <div class="stat-label">Total</div>
                    </div>
                </div>
            </div>
            
            <div id="cy"></div>
            
            <div id="legend">
                <div class="legend-item">
                    <div class="legend-node" style="background: #00e676;"></div>
                    <span>Source</span>
                </div>
                <div class="legend-item">
                    <div class="legend-node" style="background: #ff5252;"></div>
                    <span>Sink</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #4fc3f7;"></div>
                    <span>Forward Edge</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #ff9800;"></div>
                    <span>Reverse Edge</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #ff5252;"></div>
                    <span>Saturated</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #ffeb3b;"></div>
                    <span>Min-Cut</span>
                </div>
            </div>
            
            <div id="status-bar">
                <div id="phase-indicator">
                    <div id="phase-icon">▶</div>
                    <div id="phase-text">
                        <div id="phase-title">Initializing...</div>
                        <div id="phase-description">Setting up the flow network</div>
                    </div>
                </div>
                <div id="progress-container">
                    <div id="progress-bar"></div>
                </div>
                <div id="path-display"></div>
            </div>
        </div>
        
        <div id="completion">
            <h2>✅ Algorithm Complete!</h2>
            <div id="completion-stats">
                <div class="stat">
                    <div class="stat-value" id="final-flow">0</div>
                    <div class="stat-label">Maximum Flow</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="final-cut">0</div>
                    <div class="stat-label">Min-Cut</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="final-iterations">0</div>
                    <div class="stat-label">Iterations</div>
                </div>
            </div>
            <div id="theorem-box">
                <strong>Max-Flow Min-Cut Theorem:</strong><br>
                Maximum Flow = Minimum Cut Capacity ✓
            </div>
            <button id="restart-btn" onclick="restartAnimation()">🔄 Restart Animation</button>
        </div>
        
        <script>
            const DATA = {data_json};
            const STEP_DURATION = {step_duration};
            
            let cy = null;
            let residualEdges = new Set();  // Track added residual edges
            
            function initCytoscape() {{
                const elements = [];
                
                DATA.nodes.forEach(node => {{
                    let classes = '';
                    if (node === DATA.source) classes = 'source';
                    if (node === DATA.sink) classes = 'sink';
                    elements.push({{
                        data: {{ id: node, label: node }},
                        classes: classes
                    }});
                }});
                
                DATA.edges.forEach(edge => {{
                    elements.push({{
                        data: {{
                            id: edge.id,
                            source: edge.source,
                            target: edge.target,
                            label: `0/${{edge.capacity}}`,
                            flow: 0,
                            capacity: edge.capacity
                        }}
                    }});
                }});
                
                cy = cytoscape({{
                    container: document.getElementById('cy'),
                    elements: elements,
                    style: [
                        {{
                            selector: 'node',
                            style: {{
                                'background-color': '#455a64',
                                'label': 'data(label)',
                                'text-valign': 'center',
                                'text-halign': 'center',
                                'color': '#fff',
                                'font-size': '18px',
                                'font-weight': 'bold',
                                'width': '55px',
                                'height': '55px',
                                'border-width': 3,
                                'border-color': '#607d8b',
                                'text-outline-color': '#000',
                                'text-outline-width': 1
                            }}
                        }},
                        {{
                            selector: 'node.source',
                            style: {{
                                'background-color': '#00e676',
                                'border-color': '#00c853'
                            }}
                        }},
                        {{
                            selector: 'node.sink',
                            style: {{
                                'background-color': '#ff5252',
                                'border-color': '#d50000'
                            }}
                        }},
                        {{
                            selector: 'node.active',
                            style: {{
                                'border-color': '#4fc3f7',
                                'border-width': 5
                            }}
                        }},
                        {{
                            selector: 'node.cut-s',
                            style: {{
                                'border-color': '#ffeb3b',
                                'border-width': 6
                            }}
                        }},
                        {{
                            selector: 'node.cut-t',
                            style: {{
                                'border-color': '#ff9800',
                                'border-width': 6
                            }}
                        }},
                        {{
                            selector: 'edge',
                            style: {{
                                'width': 4,
                                'line-color': '#546e7a',
                                'target-arrow-color': '#546e7a',
                                'target-arrow-shape': 'triangle',
                                'curve-style': 'bezier',
                                'label': 'data(label)',
                                'font-size': '13px',
                                'font-weight': 'bold',
                                'text-background-color': '#1a1a2e',
                                'text-background-opacity': 0.9,
                                'text-background-padding': '3px',
                                'text-margin-y': -12,
                                'color': '#fff'
                            }}
                        }},
                        {{
                            selector: 'edge.highlighted',
                            style: {{
                                'line-color': '#4fc3f7',
                                'target-arrow-color': '#4fc3f7',
                                'width': 7,
                                'z-index': 999
                            }}
                        }},
                        {{
                            selector: 'edge.highlighted-reverse',
                            style: {{
                                'line-color': '#ff9800',
                                'target-arrow-color': '#ff9800',
                                'width': 7,
                                'z-index': 999
                            }}
                        }},
                        {{
                            selector: 'edge.residual',
                            style: {{
                                'line-color': '#ff9800',
                                'target-arrow-color': '#ff9800',
                                'line-style': 'dashed',
                                'width': 2,
                                'opacity': 0.7
                            }}
                        }},
                        {{
                            selector: 'edge.saturated',
                            style: {{
                                'line-color': '#ff5252',
                                'target-arrow-color': '#ff5252'
                            }}
                        }},
                        {{
                            selector: 'edge.cut-edge',
                            style: {{
                                'line-color': '#ffeb3b',
                                'target-arrow-color': '#ffeb3b',
                                'width': 8
                            }}
                        }}
                    ],
                    layout: {{
                        name: 'dagre',
                        rankDir: 'LR',
                        nodeSep: 80,
                        rankSep: 120,
                        padding: 50
                    }}
                }});
                
                // Display strategy
                document.getElementById('strategy-badge').textContent = DATA.strategy;
                document.getElementById('total-iterations').textContent = DATA.iterations.length;
                
                residualEdges.clear();
            }}
            
            function delay(ms) {{
                return new Promise(resolve => setTimeout(resolve, ms));
            }}
            
            function updatePhase(icon, title, description) {{
                document.getElementById('phase-icon').textContent = icon;
                document.getElementById('phase-title').textContent = title;
                document.getElementById('phase-description').textContent = description;
            }}
            
            function updateProgress(percent) {{
                document.getElementById('progress-bar').style.width = percent + '%';
            }}
            
            function showPath(path, bottleneck, usesReverse) {{
                const display = document.getElementById('path-display');
                let html = '';
                
                path.forEach((e, i) => {{
                    const nodeClass = e.is_reverse ? 'path-node reverse' : 'path-node';
                    const arrowClass = e.is_reverse ? 'path-arrow reverse' : 'path-arrow';
                    
                    if (i === 0) {{
                        html += `<span class="${{nodeClass}}">${{e.source}}</span>`;
                    }}
                    html += `<span class="${{arrowClass}}">${{e.is_reverse ? '⟵' : '→'}}</span>`;
                    html += `<span class="${{nodeClass}}">${{e.target}}</span>`;
                }});
                
                html += `<span class="bottleneck">Bottleneck: ${{bottleneck}}</span>`;
                
                if (usesReverse) {{
                    html += `<span class="reverse-indicator">⚠️ Uses reverse edge (flow cancellation)</span>`;
                }}
                
                display.innerHTML = html;
                display.classList.add('visible');
            }}
            
            function hidePath() {{
                document.getElementById('path-display').classList.remove('visible');
            }}
            
            async function animateParticles(path) {{
                const promises = [];
                
                for (const edge of path) {{
                    let edgeId, sourceNode, targetNode;
                    
                    if (edge.is_reverse) {{
                        // For reverse edges, we need to find the visual edge
                        // The reverse edge goes from target to source of original
                        edgeId = `${{edge.target}}-${{edge.source}}`;
                        sourceNode = edge.source;
                        targetNode = edge.target;
                    }} else {{
                        edgeId = `${{edge.source}}-${{edge.target}}`;
                        sourceNode = edge.source;
                        targetNode = edge.target;
                    }}
                    
                    for (let i = 0; i < 4; i++) {{
                        promises.push(
                            delay(i * 80).then(() => createParticle(sourceNode, targetNode, edge.is_reverse))
                        );
                    }}
                }}
                
                await Promise.all(promises);
            }}
            
            function createParticle(sourceId, targetId, isReverse) {{
                return new Promise(resolve => {{
                    const particle = document.createElement('div');
                    particle.className = 'particle' + (isReverse ? ' reverse' : '');
                    document.body.appendChild(particle);
                    
                    const cyContainer = document.getElementById('cy');
                    const rect = cyContainer.getBoundingClientRect();
                    
                    const sourceNode = cy.getElementById(sourceId);
                    const targetNode = cy.getElementById(targetId);
                    
                    if (!sourceNode.length || !targetNode.length) {{
                        particle.remove();
                        resolve();
                        return;
                    }}
                    
                    const sourcePos = sourceNode.renderedPosition();
                    const targetPos = targetNode.renderedPosition();
                    
                    const startX = rect.left + sourcePos.x;
                    const startY = rect.top + sourcePos.y;
                    const endX = rect.left + targetPos.x;
                    const endY = rect.top + targetPos.y;
                    
                    particle.style.left = startX + 'px';
                    particle.style.top = startY + 'px';
                    
                    particle.animate([
                        {{ left: startX + 'px', top: startY + 'px', opacity: 1, transform: 'scale(1)' }},
                        {{ left: endX + 'px', top: endY + 'px', opacity: 0.3, transform: 'scale(0.5)' }}
                    ], {{
                        duration: 500,
                        easing: 'ease-out'
                    }}).onfinish = () => {{
                        particle.remove();
                        resolve();
                    }};
                }});
            }}
            
            function addResidualEdge(source, target, capacity) {{
                const edgeId = `res-${{source}}-${{target}}`;
                
                if (!residualEdges.has(edgeId) && cy.getElementById(edgeId).length === 0) {{
                    cy.add({{
                        data: {{
                            id: edgeId,
                            source: source,
                            target: target,
                            label: capacity.toString(),
                            capacity: capacity
                        }},
                        classes: 'residual'
                    }});
                    residualEdges.add(edgeId);
                    
                    // Animate fade in
                    const edge = cy.getElementById(edgeId);
                    edge.style('opacity', 0);
                    edge.animate({{ style: {{ opacity: 0.7 }} }}, {{ duration: 300 }});
                }}
            }}
            
            function updateResidualEdge(source, target, capacity) {{
                const edgeId = `res-${{source}}-${{target}}`;
                const edge = cy.getElementById(edgeId);
                
                if (edge.length > 0) {{
                    if (capacity > 0) {{
                        edge.data('label', capacity.toString());
                    }} else {{
                        edge.remove();
                        residualEdges.delete(edgeId);
                    }}
                }}
            }}
            
            async function runAnimation() {{
                updatePhase('🚀', 'Starting Algorithm', `Using ${{DATA.strategy}} to find augmenting paths`);
                updateProgress(0);
                await delay(STEP_DURATION);
                
                for (let i = 0; i < DATA.iterations.length; i++) {{
                    const iter = DATA.iterations[i];
                    
                    document.getElementById('iteration-value').textContent = iter.iteration;
                    updateProgress(((i + 0.5) / DATA.iterations.length) * 100);
                    
                    // Phase 1: Finding path
                    updatePhase('🔍', `Iteration ${{iter.iteration}}: Finding Path`, 
                               `${{DATA.strategy}} searching residual network...`);
                    await delay(STEP_DURATION * 0.4);
                    
                    // Highlight path sequentially
                    for (const edge of iter.path) {{
                        let edgeId;
                        let highlightClass;
                        
                        if (edge.is_reverse) {{
                            // Highlight the reverse direction
                            edgeId = `res-${{edge.source}}-${{edge.target}}`;
                            highlightClass = 'highlighted-reverse';
                            
                            // If residual edge doesn't exist visually, highlight original
                            if (cy.getElementById(edgeId).length === 0) {{
                                edgeId = `${{edge.target}}-${{edge.source}}`;
                                highlightClass = 'highlighted-reverse';
                            }}
                        }} else {{
                            edgeId = `${{edge.source}}-${{edge.target}}`;
                            highlightClass = 'highlighted';
                        }}
                        
                        const cyEdge = cy.getElementById(edgeId);
                        if (cyEdge.length > 0) {{
                            cyEdge.addClass(highlightClass);
                        }}
                        
                        cy.getElementById(edge.source).addClass('active');
                        cy.getElementById(edge.target).addClass('active');
                        await delay(250);
                    }}
                    
                    showPath(iter.path, iter.bottleneck, iter.uses_reverse_edge);
                    await delay(STEP_DURATION * 0.4);
                    
                    // Phase 2: Push flow
                    const pushMsg = iter.uses_reverse_edge 
                        ? `Pushing ${{iter.bottleneck}} (includes flow cancellation)` 
                        : `Pushing ${{iter.bottleneck}} units`;
                    updatePhase('💧', pushMsg, 'Augmenting flow along path...');
                    
                    await animateParticles(iter.path);
                    await delay(STEP_DURATION * 0.3);
                    
                    // Phase 3: Update flows
                    updatePhase('📝', 'Updating Network', 'Adjusting edge flows and residual capacities...');
                    
                    // Update original edge flows
                    for (const [edgeId, newFlow] of Object.entries(iter.flow_after)) {{
                        const cyEdge = cy.getElementById(edgeId);
                        if (cyEdge.length > 0) {{
                            const capacity = cyEdge.data('capacity');
                            const oldFlow = parseInt(cyEdge.data('flow')) || 0;
                            
                            cyEdge.data('label', `${{newFlow}}/${{capacity}}`);
                            cyEdge.data('flow', newFlow);
                            
                            // Mark saturated
                            if (newFlow === capacity) {{
                                cyEdge.addClass('saturated');
                            }} else {{
                                cyEdge.removeClass('saturated');
                            }}
                            
                            // Add/update residual edge for reverse direction
                            const [u, v] = edgeId.split('-');
                            if (newFlow > 0) {{
                                addResidualEdge(v, u, newFlow);
                            }} else {{
                                updateResidualEdge(v, u, newFlow);
                            }}
                        }}
                    }}
                    
                    document.getElementById('flow-value').textContent = iter.total_flow;
                    
                    await delay(STEP_DURATION * 0.4);
                    
                    // Clear highlights
                    cy.elements().removeClass('highlighted highlighted-reverse active');
                    hidePath();
                    
                    updateProgress(((i + 1) / DATA.iterations.length) * 100);
                    await delay(STEP_DURATION * 0.2);
                }}
                
                // Final: Show min-cut
                updatePhase('✂️', 'No More Augmenting Paths', 'Computing minimum cut...');
                await delay(STEP_DURATION);
                
                // Highlight min-cut
                DATA.final.S.forEach(node => {{
                    cy.getElementById(node).addClass('cut-s');
                }});
                DATA.final.T.forEach(node => {{
                    cy.getElementById(node).addClass('cut-t');
                }});
                DATA.final.cut_edges.forEach(edge => {{
                    const edgeId = `${{edge.source}}-${{edge.target}}`;
                    cy.getElementById(edgeId).addClass('cut-edge');
                }});
                
                updatePhase('✅', 'Complete!', 
                           `Max Flow = ${{DATA.final.max_flow}}, Min Cut = ${{DATA.final.cut_capacity}}`);
                updateProgress(100);
                
                await delay(STEP_DURATION);
                
                // Show completion
                document.getElementById('final-flow').textContent = DATA.final.max_flow;
                document.getElementById('final-cut').textContent = DATA.final.cut_capacity;
                document.getElementById('final-iterations').textContent = DATA.final.total_iterations;
                document.getElementById('completion').classList.add('visible');
            }}
            
            function restartAnimation() {{
                document.getElementById('completion').classList.remove('visible');
                if (cy) cy.destroy();
                document.getElementById('iteration-value').textContent = '0';
                document.getElementById('flow-value').textContent = '0';
                hidePath();
                initCytoscape();
                runAnimation();
            }}
            
            // Start
            initCytoscape();
            setTimeout(runAnimation, 800);
        </script>
    </body>
    </html>
    """
    
    components.html(html, height=height)


# =============================================================================
# SESSION STATE
# =============================================================================

def init_session_state():
    if 'manual_nodes' not in st.session_state:
        st.session_state.manual_nodes = ['s', 't']
    if 'manual_edges' not in st.session_state:
        st.session_state.manual_edges = []
    if 'network_ready' not in st.session_state:
        st.session_state.network_ready = False
    if 'current_network' not in st.session_state:
        st.session_state.current_network = None
    if 'animation_started' not in st.session_state:
        st.session_state.animation_started = False


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.set_page_config(
        page_title="Ford-Fulkerson Visualizer",
        page_icon="🌊",
        layout="wide"
    )
    
    init_session_state()
    
    st.markdown("""
        <style>
            .block-container { padding-top: 1rem; }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🌊 Ford-Fulkerson Maximum Flow Visualizer")
    
    # ========== SIDEBAR ==========
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Step 1: Input Method
        st.subheader("1️⃣ Network Input")
        input_method = st.radio(
            "Select input method:",
            ["📚 Preset Examples", "✏️ Manual Entry", "🎲 Random Generation"],
            label_visibility="collapsed"
        )
        
        network = None
        network_name = ""
        
        # ---------- PRESET ----------
        if input_method == "📚 Preset Examples":
            presets = get_presets()
            preset_name = st.selectbox("Choose network:", list(presets.keys()))
            
            # Show description
            if 'description' in presets[preset_name]:
                with st.expander("ℹ️ About this network"):
                    st.markdown(presets[preset_name]['description'])
            
            if st.button("📥 Load Network", type="primary", use_container_width=True):
                network = create_network_from_preset(presets[preset_name])
                network_name = preset_name
                st.session_state.current_network = network
                st.session_state.network_name = network_name
                st.session_state.network_ready = True
                st.session_state.animation_started = False
                st.rerun()
        
        # ---------- MANUAL ----------
        elif input_method == "✏️ Manual Entry":
            with st.expander("📍 Nodes", expanded=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    new_node = st.text_input("Name:", key="new_node", placeholder="e.g., v1")
                with col2:
                    st.write("")
                    st.write("")
                    if st.button("➕", key="add_node"):
                        if new_node and new_node.strip() not in st.session_state.manual_nodes:
                            st.session_state.manual_nodes.append(new_node.strip())
                            st.rerun()
                
                st.caption(f"Nodes: {', '.join(st.session_state.manual_nodes)}")
            
            with st.expander("🔗 Edges", expanded=True):
                if len(st.session_state.manual_nodes) >= 2:
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        edge_from = st.selectbox("From", st.session_state.manual_nodes, key="ef")
                    with c2:
                        edge_to = st.selectbox("To", [n for n in st.session_state.manual_nodes if n != edge_from], key="et")
                    with c3:
                        edge_cap = st.number_input("Cap", 1, 1000, 10, key="ec")
                    
                    if st.button("➕ Add Edge", use_container_width=True):
                        if (edge_from, edge_to) not in [(e[0], e[1]) for e in st.session_state.manual_edges]:
                            st.session_state.manual_edges.append((edge_from, edge_to, edge_cap))
                            st.rerun()
                
                for i, (u, v, c) in enumerate(st.session_state.manual_edges):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.text(f"{u} → {v} ({c})")
                    with col2:
                        if st.button("🗑️", key=f"de{i}"):
                            st.session_state.manual_edges.pop(i)
                            st.rerun()
            
            if len(st.session_state.manual_nodes) >= 2 and st.session_state.manual_edges:
                source = st.selectbox("Source:", st.session_state.manual_nodes, key="src")
                sink = st.selectbox("Sink:", [n for n in st.session_state.manual_nodes if n != source], key="snk")
                
                if st.button("🔨 Build Network", type="primary", use_container_width=True):
                    network = FlowNetwork()
                    for node in st.session_state.manual_nodes:
                        network.add_node(node)
                    for u, v, c in st.session_state.manual_edges:
                        network.add_edge(u, v, c)
                    network.set_source_sink(source, sink)
                    
                    st.session_state.current_network = network
                    st.session_state.network_name = "Custom Network"
                    st.session_state.network_ready = True
                    st.session_state.animation_started = False
                    st.rerun()
            
            if st.button("🗑️ Clear All"):
                st.session_state.manual_nodes = ['s', 't']
                st.session_state.manual_edges = []
                st.session_state.network_ready = False
                st.rerun()
        
        # ---------- RANDOM ----------
        else:
            num_nodes = st.slider("Nodes:", 4, 12, 6)
            edge_prob = st.slider("Edge probability:", 0.2, 0.8, 0.4)
            c1, c2 = st.columns(2)
            with c1:
                min_cap = st.number_input("Min cap:", 1, 50, 1)
            with c2:
                max_cap = st.number_input("Max cap:", min_cap, 100, 20)
            seed = st.number_input("Seed (0=random):", 0, 9999, 42)
            
            if st.button("🎲 Generate", type="primary", use_container_width=True):
                network = generate_random_network(num_nodes, edge_prob, min_cap, max_cap, seed if seed > 0 else None)
                st.session_state.current_network = network
                st.session_state.network_name = f"Random (seed={seed})"
                st.session_state.network_ready = True
                st.session_state.animation_started = False
                st.rerun()
        
        # Step 2: Algorithm Settings (only show when network is ready)
        if st.session_state.network_ready:
            st.divider()
            st.subheader("2️⃣ Algorithm Settings")
            
            strategy = st.radio(
                "Path finding strategy:",
                ["BFS (Edmonds-Karp)", "DFS (Basic Ford-Fulkerson)"],
                help="""
                **BFS**: Finds shortest augmenting paths. Guaranteed O(VE²) time.
                
                **DFS**: Finds any path. Can be O(E·|f*|) in worst case.
                
                Try both on "Worst Case" example to see the difference!
                """
            )
            strategy_key = 'bfs' if 'BFS' in strategy else 'dfs'
            
            speed = st.slider(
                "Animation speed (ms/phase):",
                500, 4000, 1500, 250,
                help="Lower = faster animation"
            )
            
            st.divider()
            st.subheader("3️⃣ Run Animation")
            
            if st.button("▶️ START ANIMATION", type="primary", use_container_width=True):
                st.session_state.animation_started = True
                st.session_state.strategy = strategy_key
                st.session_state.speed = speed
                st.rerun()
            
            if st.button("🔄 Reset Network"):
                st.session_state.network_ready = False
                st.session_state.animation_started = False
                st.rerun()
    
    # ========== MAIN CONTENT ==========
    if not st.session_state.network_ready:
        st.info("👈 Configure a network in the sidebar to begin")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            ### 📚 Presets
            - Textbook examples
            - Worst-case demos
            - Bipartite matching
            """)
        with col2:
            st.markdown("""
            ### ✏️ Manual
            - Custom nodes/edges
            - Any topology
            - Full control
            """)
        with col3:
            st.markdown("""
            ### 🎲 Random
            - Configurable size
            - Edge probability
            - Reproducible
            """)
    
    elif not st.session_state.animation_started:
        network = st.session_state.current_network
        
        st.subheader(f"📊 Network: {st.session_state.network_name}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Nodes", len(network.nodes))
        with col2:
            st.metric("Edges", len(network.edges))
        with col3:
            st.metric("Source", network.source)
        with col4:
            st.metric("Sink", network.sink)
        
        st.info("👈 Select algorithm (BFS/DFS) and click START ANIMATION")
        
        with st.expander("🔍 Edge Details"):
            for (u, v), cap in network.edges.items():
                st.text(f"{u} → {v} : capacity = {cap}")
    
    else:
        # Show animation
        network = st.session_state.current_network
        strategy = st.session_state.strategy
        speed = st.session_state.speed
        
        st.subheader(f"🎬 {st.session_state.network_name} | Strategy: {strategy.upper()}")
        
        # Build and render
        with st.spinner("Computing algorithm timeline..."):
            animation_data = build_animation_data(network, strategy)
        
        # Show summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Iterations", animation_data['final']['total_iterations'])
        with col2:
            st.metric("Max Flow", animation_data['final']['max_flow'])
        with col3:
            st.metric("Strategy", strategy.upper())
        
        # Warning for many iterations
        if animation_data['final']['total_iterations'] > 20:
            st.warning(f"⚠️ {animation_data['final']['total_iterations']} iterations! This demonstrates worst-case DFS behavior. Try BFS for comparison.")
        
        render_auto_animation(animation_data, step_duration=speed, height=700)
        
        with st.expander("📋 Iteration Details"):
            for iter_data in animation_data['iterations']:
                path_parts = []
                for e in iter_data['path']:
                    arrow = "⟵" if e['is_reverse'] else "→"
                    path_parts.append(f"{e['source']} {arrow} {e['target']}")
                path_str = " | ".join(path_parts)
                
                reverse_note = " 🔄 (uses reverse edge)" if iter_data['uses_reverse_edge'] else ""
                
                st.markdown(f"""
                **Iteration {iter_data['iteration']}**{reverse_note}
                - Path: `{path_str}`
                - Bottleneck: {iter_data['bottleneck']}
                - Total flow: {iter_data['total_flow']}
                """)


if __name__ == "__main__":
    main()
