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
        self.nodes = []
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
        if (u, v) in self.edges:
            return self.edges[(u, v)] - self.flow[(u, v)]
        elif (v, u) in self.edges:
            return self.flow[(v, u)]
        return 0
    
    def get_neighbors(self, u, prefer_longer_paths=False):
        neighbors = []
        for v in self.nodes:
            if v != u and self.get_residual_capacity(u, v) > 0:
                neighbors.append(v)
        
        if prefer_longer_paths:
            neighbors.sort(key=lambda x: (x == self.sink, x))
        
        return neighbors
    
    def find_augmenting_path_bfs(self):
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
        if self.source is None or self.sink is None:
            return None, 0
        
        visited = set()
        parent = {}
        
        def dfs(u):
            if u == self.sink:
                return True
            visited.add(u)
            neighbors = self.get_neighbors(u, prefer_longer_paths)
            for v in neighbors:
                if v not in visited:
                    parent[v] = u
                    if dfs(v):
                        return True
            return False
        
        if not dfs(self.source):
            return None, 0
        
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
        for (u, v) in path:
            if (u, v) in self.edges:
                self.flow[(u, v)] += bottleneck
            else:
                self.flow[(v, u)] -= bottleneck
    
    def get_max_flow_value(self):
        return sum(self.flow.get((self.source, v), 0) for v in self.nodes)
    
    def get_min_cut(self):
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
    
    def decompose_flow(self):
        """
        Decompose the current flow into a set of s-t paths with their flow values.
        Returns a list of (path, flow_amount) tuples.
        """
        # Create a copy of flow values to work with
        remaining_flow = {edge: flow for edge, flow in self.flow.items() if flow > 0}
        
        paths = []
        
        while True:
            # Find a path from source to sink with positive flow
            path, min_flow = self._find_flow_path(remaining_flow)
            
            if path is None:
                break
            
            # Record this path
            paths.append({
                'path': path,
                'flow': min_flow
            })
            
            # Subtract flow from this path
            for (u, v) in path:
                remaining_flow[(u, v)] -= min_flow
                if remaining_flow[(u, v)] == 0:
                    del remaining_flow[(u, v)]
        
        return paths
    
    def _find_flow_path(self, flow_dict):
        """Find a path from source to sink using only edges with positive flow."""
        parent = {self.source: None}
        visited = {self.source}
        queue = deque([self.source])
        
        while queue:
            u = queue.popleft()
            if u == self.sink:
                break
            
            for v in self.nodes:
                if v not in visited and flow_dict.get((u, v), 0) > 0:
                    visited.add(v)
                    parent[v] = u
                    queue.append(v)
        
        if self.sink not in parent:
            return None, 0
        
        # Reconstruct path and find minimum flow
        path = []
        min_flow = float('inf')
        v = self.sink
        
        while parent[v] is not None:
            u = parent[v]
            min_flow = min(min_flow, flow_dict.get((u, v), 0))
            path.append((u, v))
            v = u
        
        path.reverse()
        return path, min_flow
    
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
    """Run Ford-Fulkerson and build complete animation timeline."""
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
    max_iterations = 1000
    
    while iteration_num < max_iterations:
        if strategy == 'bfs':
            path, bottleneck = network.find_augmenting_path_bfs()
        else:
            path, bottleneck = network.find_augmenting_path_dfs(prefer_longer_paths=True)
        
        if path is None:
            break
        
        iteration_num += 1
        uses_reverse = any((u, v) not in network.edges for u, v in path)
        
        iteration = {
            'iteration': iteration_num,
            'path': [{'source': u, 'target': v, 'is_reverse': (u, v) not in network.edges} 
                    for u, v in path],
            'bottleneck': bottleneck,
            'uses_reverse_edge': uses_reverse,
            'flow_before': {f"{u}-{v}": network.flow[(u, v)] for u, v in network.edges},
        }
        
        network.augment_flow(path, bottleneck)
        
        iteration['flow_after'] = {f"{u}-{v}": network.flow[(u, v)] for u, v in network.edges}
        iteration['total_flow'] = network.get_max_flow_value()
        
        data['iterations'].append(iteration)
    
    # Final state
    S, T, cut_cap, cut_edges = network.get_min_cut()
    
    # Flow decomposition
    flow_paths = network.decompose_flow()
    
    data['final'] = {
        'max_flow': network.get_max_flow_value(),
        'S': list(S),
        'T': list(T),
        'cut_capacity': cut_cap,
        'cut_edges': [{'source': u, 'target': v} for u, v in cut_edges],
        'total_iterations': iteration_num,
        'final_flows': {f"{u}-{v}": network.flow[(u, v)] for u, v in network.edges},
        'flow_paths': flow_paths
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
            'nodes': ['s', 'u', 'v', 't'],
            'edges': [
                ('s', 'u', 100), ('s', 'v', 100),
                ('u', 'v', 1),
                ('u', 't', 100), ('v', 't', 100),
            ],
            'source': 's', 'sink': 't',
            'description': '''WORST CASE: BFS uses 2 iterations, DFS may use up to 200!'''
        },
        "User Example (Max Flow = 110)": {
            'nodes': ['s', 'v1', 'v2', 'v3', 'v4', 't'],
            'edges': [
                ('s', 'v1', 10), ('s', 'v2', 110),
                ('v1', 'v2', 30), ('v1', 'v4', 40),
                ('v2', 'v3', 10), ('v2', 'v4', 40), ('v2', 't', 70),
                ('v3', 'v4', 60),
                ('v4', 't', 40),
            ],
            'source': 's', 'sink': 't',
            'description': 'Your example network with max flow = 110.'
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
            'description': 'Maximum bipartite matching as max-flow.'
        },
    }


def create_network_from_preset(preset):
    network = FlowNetwork()
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
    
    for i in range(len(nodes) - 1):
        if random.random() < 0.7:
            cap = random.randint(min_cap, max_cap)
            network.add_edge(nodes[i], nodes[i + 1], cap)
    
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
    """Render self-playing animation with flow decomposition at end."""
    
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
                padding: 12px 20px;
                background: rgba(255,255,255,0.05);
                border-bottom: 1px solid rgba(255,255,255,0.1);
                display: flex;
                justify-content: space-between;
                align-items: center;
                flex-wrap: wrap;
                gap: 10px;
            }}
            #title {{
                font-size: 18px;
                font-weight: 600;
                color: #4fc3f7;
            }}
            #strategy-badge {{
                background: #ff9800;
                color: #000;
                padding: 4px 12px;
                border-radius: 12px;
                font-size: 11px;
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
                font-size: 22px;
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
                padding: 12px 20px;
                background: rgba(0,0,0,0.3);
                border-top: 1px solid rgba(255,255,255,0.1);
            }}
            #phase-indicator {{
                display: flex;
                align-items: center;
                gap: 12px;
                margin-bottom: 10px;
            }}
            #phase-icon {{
                width: 36px;
                height: 36px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 18px;
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
                font-size: 14px;
                font-weight: 600;
                margin-bottom: 2px;
            }}
            #phase-description {{
                font-size: 11px;
                color: rgba(255,255,255,0.7);
            }}
            #progress-container {{
                height: 5px;
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
                margin-top: 10px;
                padding: 8px 12px;
                background: rgba(79, 195, 247, 0.15);
                border: 1px solid rgba(79, 195, 247, 0.3);
                border-radius: 6px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 12px;
                display: none;
            }}
            #path-display.visible {{
                display: block;
            }}
            .path-arrow {{
                color: #4fc3f7;
                margin: 0 5px;
            }}
            .path-arrow.reverse {{
                color: #ff9800;
            }}
            .path-node {{
                background: rgba(79, 195, 247, 0.3);
                padding: 2px 6px;
                border-radius: 3px;
            }}
            .path-node.reverse {{
                background: rgba(255, 152, 0, 0.3);
            }}
            .bottleneck {{
                color: #ffeb3b;
                margin-left: 12px;
                font-weight: 600;
            }}
            .reverse-indicator {{
                color: #ff9800;
                margin-left: 8px;
                font-size: 10px;
            }}
            #legend {{
                position: absolute;
                top: 60px;
                right: 12px;
                background: rgba(0,0,0,0.6);
                padding: 10px;
                border-radius: 6px;
                font-size: 10px;
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                gap: 6px;
                margin-bottom: 4px;
            }}
            .legend-color {{
                width: 16px;
                height: 3px;
                border-radius: 2px;
            }}
            .legend-node {{
                width: 10px;
                height: 10px;
                border-radius: 50%;
            }}
            #completion {{
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0,0,0,0.92);
                display: none;
                align-items: flex-start;
                justify-content: center;
                z-index: 1000;
                padding: 30px;
                overflow-y: auto;
            }}
            #completion.visible {{
                display: flex;
            }}
            #completion-content {{
                max-width: 700px;
                width: 100%;
            }}
            #completion h2 {{
                font-size: 26px;
                color: #00e676;
                text-align: center;
                margin-bottom: 20px;
            }}
            #completion-stats {{
                display: flex;
                gap: 25px;
                justify-content: center;
                margin-bottom: 25px;
                flex-wrap: wrap;
            }}
            #completion .stat-value {{
                font-size: 36px;
                color: #00e676;
            }}
            #theorem-box {{
                background: rgba(0, 230, 118, 0.15);
                border: 1px solid rgba(0, 230, 118, 0.3);
                padding: 12px 20px;
                border-radius: 8px;
                text-align: center;
                margin-bottom: 25px;
            }}
            #flow-decomposition {{
                background: rgba(79, 195, 247, 0.1);
                border: 1px solid rgba(79, 195, 247, 0.3);
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 20px;
            }}
            #flow-decomposition h3 {{
                color: #4fc3f7;
                margin-bottom: 15px;
                font-size: 16px;
            }}
            .flow-path {{
                background: rgba(255,255,255,0.05);
                border-radius: 6px;
                padding: 10px 12px;
                margin-bottom: 8px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 13px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                flex-wrap: wrap;
                gap: 10px;
            }}
            .flow-path-nodes {{
                flex: 1;
            }}
            .flow-path-amount {{
                background: #00e676;
                color: #000;
                padding: 4px 12px;
                border-radius: 12px;
                font-weight: 700;
                font-size: 12px;
            }}
            .flow-path .path-arrow {{
                color: #4fc3f7;
                margin: 0 8px;
            }}
            .flow-path .path-node {{
                background: rgba(79, 195, 247, 0.3);
                padding: 3px 8px;
                border-radius: 4px;
            }}
            #final-flows {{
                background: rgba(255,255,255,0.05);
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 20px;
            }}
            #final-flows h3 {{
                color: #fff;
                margin-bottom: 12px;
                font-size: 14px;
            }}
            .edge-flow {{
                display: inline-block;
                margin: 4px;
                padding: 6px 10px;
                background: rgba(255,255,255,0.08);
                border-radius: 4px;
                font-size: 12px;
            }}
            .edge-flow.saturated {{
                background: rgba(255, 82, 82, 0.3);
                border: 1px solid rgba(255, 82, 82, 0.5);
            }}
            .edge-flow.unused {{
                opacity: 0.5;
            }}
            #restart-btn {{
                display: block;
                margin: 0 auto;
                padding: 12px 30px;
                font-size: 14px;
                font-weight: 600;
                background: #4fc3f7;
                color: #1a1a2e;
                border: none;
                border-radius: 20px;
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
                <div style="display: flex; align-items: center; gap: 12px;">
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
                    <span>Forward</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #ff9800;"></div>
                    <span>Reverse</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #ff5252;"></div>
                    <span>Saturated</span>
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
            <div id="completion-content">
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
                    <strong>Max-Flow Min-Cut Theorem:</strong> Maximum Flow = Minimum Cut Capacity ✓
                </div>
                
                <div id="flow-decomposition">
                    <h3>📊 Flow Decomposition (Final Paths Used)</h3>
                    <div id="flow-paths"></div>
                </div>
                
                <div id="final-flows">
                    <h3>📋 Final Edge Flows</h3>
                    <div id="edge-flows"></div>
                </div>
                
                <button id="restart-btn" onclick="restartAnimation()">🔄 Restart Animation</button>
            </div>
        </div>
        
        <script>
            const DATA = {data_json};
            const STEP_DURATION = {step_duration};
            
            let cy = null;
            let residualEdges = new Set();
            
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
                                'font-size': '16px',
                                'font-weight': 'bold',
                                'width': '50px',
                                'height': '50px',
                                'border-width': 3,
                                'border-color': '#607d8b',
                                'text-outline-color': '#000',
                                'text-outline-width': 1
                            }}
                        }},
                        {{
                            selector: 'node.source',
                            style: {{ 'background-color': '#00e676', 'border-color': '#00c853' }}
                        }},
                        {{
                            selector: 'node.sink',
                            style: {{ 'background-color': '#ff5252', 'border-color': '#d50000' }}
                        }},
                        {{
                            selector: 'node.active',
                            style: {{ 'border-color': '#4fc3f7', 'border-width': 5 }}
                        }},
                        {{
                            selector: 'node.cut-s',
                            style: {{ 'border-color': '#ffeb3b', 'border-width': 6 }}
                        }},
                        {{
                            selector: 'node.cut-t',
                            style: {{ 'border-color': '#ff9800', 'border-width': 6 }}
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
                                'font-size': '12px',
                                'font-weight': 'bold',
                                'text-background-color': '#1a1a2e',
                                'text-background-opacity': 0.9,
                                'text-background-padding': '3px',
                                'text-margin-y': -10,
                                'color': '#fff'
                            }}
                        }},
                        {{
                            selector: 'edge.highlighted',
                            style: {{ 'line-color': '#4fc3f7', 'target-arrow-color': '#4fc3f7', 'width': 6, 'z-index': 999 }}
                        }},
                        {{
                            selector: 'edge.highlighted-reverse',
                            style: {{ 'line-color': '#ff9800', 'target-arrow-color': '#ff9800', 'width': 6, 'z-index': 999 }}
                        }},
                        {{
                            selector: 'edge.residual',
                            style: {{ 'line-color': '#ff9800', 'target-arrow-color': '#ff9800', 'line-style': 'dashed', 'width': 2, 'opacity': 0.7 }}
                        }},
                        {{
                            selector: 'edge.saturated',
                            style: {{ 'line-color': '#ff5252', 'target-arrow-color': '#ff5252' }}
                        }},
                        {{
                            selector: 'edge.cut-edge',
                            style: {{ 'line-color': '#ffeb3b', 'target-arrow-color': '#ffeb3b', 'width': 7 }}
                        }}
                    ],
                    layout: {{ name: 'dagre', rankDir: 'LR', nodeSep: 70, rankSep: 100, padding: 40 }}
                }});
                
                document.getElementById('strategy-badge').textContent = DATA.strategy;
                document.getElementById('total-iterations').textContent = DATA.iterations.length;
                residualEdges.clear();
            }}
            
            function delay(ms) {{ return new Promise(resolve => setTimeout(resolve, ms)); }}
            
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
                    if (i === 0) html += `<span class="${{nodeClass}}">${{e.source}}</span>`;
                    html += `<span class="${{arrowClass}}">${{e.is_reverse ? '⟵' : '→'}}</span>`;
                    html += `<span class="${{nodeClass}}">${{e.target}}</span>`;
                }});
                html += `<span class="bottleneck">Bottleneck: ${{bottleneck}}</span>`;
                if (usesReverse) html += `<span class="reverse-indicator">⚠️ Uses reverse edge</span>`;
                display.innerHTML = html;
                display.classList.add('visible');
            }}
            
            function hidePath() {{ document.getElementById('path-display').classList.remove('visible'); }}
            
            async function animateParticles(path) {{
                const promises = [];
                for (const edge of path) {{
                    for (let i = 0; i < 4; i++) {{
                        promises.push(delay(i * 70).then(() => createParticle(edge.source, edge.target, edge.is_reverse)));
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
                    
                    if (!sourceNode.length || !targetNode.length) {{ particle.remove(); resolve(); return; }}
                    
                    const sourcePos = sourceNode.renderedPosition();
                    const targetPos = targetNode.renderedPosition();
                    const startX = rect.left + sourcePos.x, startY = rect.top + sourcePos.y;
                    const endX = rect.left + targetPos.x, endY = rect.top + targetPos.y;
                    
                    particle.style.left = startX + 'px';
                    particle.style.top = startY + 'px';
                    
                    particle.animate([
                        {{ left: startX + 'px', top: startY + 'px', opacity: 1, transform: 'scale(1)' }},
                        {{ left: endX + 'px', top: endY + 'px', opacity: 0.3, transform: 'scale(0.5)' }}
                    ], {{ duration: 450, easing: 'ease-out' }}).onfinish = () => {{ particle.remove(); resolve(); }};
                }});
            }}
            
            function addResidualEdge(source, target, capacity) {{
                const edgeId = `res-${{source}}-${{target}}`;
                if (!residualEdges.has(edgeId) && cy.getElementById(edgeId).length === 0) {{
                    cy.add({{ data: {{ id: edgeId, source, target, label: capacity.toString(), capacity }}, classes: 'residual' }});
                    residualEdges.add(edgeId);
                    cy.getElementById(edgeId).style('opacity', 0);
                    cy.getElementById(edgeId).animate({{ style: {{ opacity: 0.7 }} }}, {{ duration: 300 }});
                }}
            }}
            
            function renderFlowDecomposition() {{
                const pathsContainer = document.getElementById('flow-paths');
                const flowPaths = DATA.final.flow_paths;
                
                if (flowPaths.length === 0) {{
                    pathsContainer.innerHTML = '<p style="color: rgba(255,255,255,0.6);">No flow paths (max flow = 0)</p>';
                    return;
                }}
                
                let html = '';
                flowPaths.forEach((fp, idx) => {{
                    const pathNodes = [fp.path[0][0]];
                    fp.path.forEach(edge => pathNodes.push(edge[1]));
                    
                    const pathHtml = pathNodes.map(n => `<span class="path-node">${{n}}</span>`).join('<span class="path-arrow">→</span>');
                    
                    html += `
                        <div class="flow-path">
                            <div class="flow-path-nodes">${{pathHtml}}</div>
                            <div class="flow-path-amount">${{fp.flow}} units</div>
                        </div>
                    `;
                }});
                
                pathsContainer.innerHTML = html;
            }}
            
            function renderFinalFlows() {{
                const container = document.getElementById('edge-flows');
                const finalFlows = DATA.final.final_flows;
                
                let html = '';
                for (const [edgeId, flow] of Object.entries(finalFlows)) {{
                    const edge = DATA.edges.find(e => e.id === edgeId);
                    const capacity = edge ? edge.capacity : 0;
                    
                    let classes = 'edge-flow';
                    if (flow === capacity && flow > 0) classes += ' saturated';
                    if (flow === 0) classes += ' unused';
                    
                    const [u, v] = edgeId.split('-');
                    html += `<span class="${{classes}}">${{u}}→${{v}}: ${{flow}}/${{capacity}}</span>`;
                }}
                
                container.innerHTML = html;
            }}
            
            async function runAnimation() {{
                updatePhase('🚀', 'Starting Algorithm', `Using ${{DATA.strategy}} to find augmenting paths`);
                updateProgress(0);
                await delay(STEP_DURATION);
                
                for (let i = 0; i < DATA.iterations.length; i++) {{
                    const iter = DATA.iterations[i];
                    document.getElementById('iteration-value').textContent = iter.iteration;
                    updateProgress(((i + 0.5) / DATA.iterations.length) * 100);
                    
                    updatePhase('🔍', `Iteration ${{iter.iteration}}: Finding Path`, `${{DATA.strategy}} searching...`);
                    await delay(STEP_DURATION * 0.35);
                    
                    for (const edge of iter.path) {{
                        let edgeId = edge.is_reverse ? `res-${{edge.source}}-${{edge.target}}` : `${{edge.source}}-${{edge.target}}`;
                        let highlightClass = edge.is_reverse ? 'highlighted-reverse' : 'highlighted';
                        if (cy.getElementById(edgeId).length === 0 && edge.is_reverse) {{
                            edgeId = `${{edge.target}}-${{edge.source}}`;
                        }}
                        const cyEdge = cy.getElementById(edgeId);
                        if (cyEdge.length > 0) cyEdge.addClass(highlightClass);
                        cy.getElementById(edge.source).addClass('active');
                        cy.getElementById(edge.target).addClass('active');
                        await delay(200);
                    }}
                    
                    showPath(iter.path, iter.bottleneck, iter.uses_reverse_edge);
                    await delay(STEP_DURATION * 0.35);
                    
                    updatePhase('💧', `Pushing ${{iter.bottleneck}} units`, iter.uses_reverse_edge ? 'Includes flow cancellation' : 'Augmenting flow...');
                    await animateParticles(iter.path);
                    await delay(STEP_DURATION * 0.25);
                    
                    updatePhase('📝', 'Updating Network', 'Adjusting flows...');
                    
                    for (const [edgeId, newFlow] of Object.entries(iter.flow_after)) {{
                        const cyEdge = cy.getElementById(edgeId);
                        if (cyEdge.length > 0) {{
                            const capacity = cyEdge.data('capacity');
                            cyEdge.data('label', `${{newFlow}}/${{capacity}}`);
                            cyEdge.data('flow', newFlow);
                            if (newFlow === capacity) cyEdge.addClass('saturated');
                            else cyEdge.removeClass('saturated');
                            
                            const [u, v] = edgeId.split('-');
                            if (newFlow > 0) addResidualEdge(v, u, newFlow);
                        }}
                    }}
                    
                    document.getElementById('flow-value').textContent = iter.total_flow;
                    await delay(STEP_DURATION * 0.3);
                    
                    cy.elements().removeClass('highlighted highlighted-reverse active');
                    hidePath();
                    updateProgress(((i + 1) / DATA.iterations.length) * 100);
                    await delay(STEP_DURATION * 0.15);
                }}
                
                updatePhase('✂️', 'No More Paths', 'Computing minimum cut...');
                await delay(STEP_DURATION * 0.7);
                
                DATA.final.S.forEach(node => cy.getElementById(node).addClass('cut-s'));
                DATA.final.T.forEach(node => cy.getElementById(node).addClass('cut-t'));
                DATA.final.cut_edges.forEach(edge => {{
                    cy.getElementById(`${{edge.source}}-${{edge.target}}`).addClass('cut-edge');
                }});
                
                updatePhase('✅', 'Complete!', `Max Flow = ${{DATA.final.max_flow}}`);
                updateProgress(100);
                await delay(STEP_DURATION * 0.7);
                
                document.getElementById('final-flow').textContent = DATA.final.max_flow;
                document.getElementById('final-cut').textContent = DATA.final.cut_capacity;
                document.getElementById('final-iterations').textContent = DATA.final.total_iterations;
                
                renderFlowDecomposition();
                renderFinalFlows();
                
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
            
            initCytoscape();
            setTimeout(runAnimation, 700);
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
    
    st.markdown("""<style>.block-container { padding-top: 1rem; }</style>""", unsafe_allow_html=True)
    
    st.title("🌊 Ford-Fulkerson Maximum Flow Visualizer")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("1️⃣ Network Input")
        input_method = st.radio(
            "Method:",
            ["📚 Preset", "✏️ Manual", "🎲 Random"],
            label_visibility="collapsed"
        )
        
        network = None
        
        if input_method == "📚 Preset":
            presets = get_presets()
            preset_name = st.selectbox("Network:", list(presets.keys()))
            
            if 'description' in presets[preset_name]:
                with st.expander("ℹ️ Info"):
                    st.markdown(presets[preset_name]['description'])
            
            if st.button("📥 Load", type="primary", use_container_width=True):
                network = create_network_from_preset(presets[preset_name])
                st.session_state.current_network = network
                st.session_state.network_name = preset_name
                st.session_state.network_ready = True
                st.session_state.animation_started = False
                st.rerun()
        
        elif input_method == "✏️ Manual":
            with st.expander("📍 Nodes", expanded=True):
                c1, c2 = st.columns([3, 1])
                with c1:
                    new_node = st.text_input("Name:", key="nn", placeholder="v1")
                with c2:
                    st.write("")
                    st.write("")
                    if st.button("➕", key="an"):
                        if new_node and new_node.strip() not in st.session_state.manual_nodes:
                            st.session_state.manual_nodes.append(new_node.strip())
                            st.rerun()
                st.caption(f"Nodes: {', '.join(st.session_state.manual_nodes)}")
            
            with st.expander("🔗 Edges", expanded=True):
                if len(st.session_state.manual_nodes) >= 2:
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        ef = st.selectbox("From", st.session_state.manual_nodes, key="ef")
                    with c2:
                        et = st.selectbox("To", [n for n in st.session_state.manual_nodes if n != ef], key="et")
                    with c3:
                        ec = st.number_input("Cap", 1, 1000, 10, key="ec")
                    
                    if st.button("➕ Add", use_container_width=True):
                        if (ef, et) not in [(e[0], e[1]) for e in st.session_state.manual_edges]:
                            st.session_state.manual_edges.append((ef, et, ec))
                            st.rerun()
                
                for i, (u, v, c) in enumerate(st.session_state.manual_edges):
                    c1, c2 = st.columns([4, 1])
                    with c1:
                        st.text(f"{u}→{v} ({c})")
                    with c2:
                        if st.button("🗑️", key=f"de{i}"):
                            st.session_state.manual_edges.pop(i)
                            st.rerun()
            
            if len(st.session_state.manual_nodes) >= 2 and st.session_state.manual_edges:
                src = st.selectbox("Source:", st.session_state.manual_nodes, key="src")
                snk = st.selectbox("Sink:", [n for n in st.session_state.manual_nodes if n != src], key="snk")
                
                if st.button("🔨 Build", type="primary", use_container_width=True):
                    network = FlowNetwork()
                    for node in st.session_state.manual_nodes:
                        network.add_node(node)
                    for u, v, c in st.session_state.manual_edges:
                        network.add_edge(u, v, c)
                    network.set_source_sink(src, snk)
                    st.session_state.current_network = network
                    st.session_state.network_name = "Custom"
                    st.session_state.network_ready = True
                    st.session_state.animation_started = False
                    st.rerun()
            
            if st.button("🗑️ Clear"):
                st.session_state.manual_nodes = ['s', 't']
                st.session_state.manual_edges = []
                st.session_state.network_ready = False
                st.rerun()
        
        else:  # Random
            nn = st.slider("Nodes:", 4, 12, 6)
            ep = st.slider("Edge prob:", 0.2, 0.8, 0.4)
            c1, c2 = st.columns(2)
            with c1:
                minc = st.number_input("Min:", 1, 50, 1)
            with c2:
                maxc = st.number_input("Max:", minc, 100, 20)
            seed = st.number_input("Seed:", 0, 9999, 42)
            
            if st.button("🎲 Generate", type="primary", use_container_width=True):
                network = generate_random_network(nn, ep, minc, maxc, seed if seed > 0 else None)
                st.session_state.current_network = network
                st.session_state.network_name = f"Random"
                st.session_state.network_ready = True
                st.session_state.animation_started = False
                st.rerun()
        
        if st.session_state.network_ready:
            st.divider()
            st.subheader("2️⃣ Algorithm")
            
            strategy = st.radio(
                "Strategy:",
                ["BFS (Edmonds-Karp)", "DFS"],
                help="BFS: O(VE²) guaranteed. DFS: May be slower."
            )
            strategy_key = 'bfs' if 'BFS' in strategy else 'dfs'
            
            speed = st.slider("Speed (ms):", 500, 4000, 1500, 250)
            
            st.divider()
            
            if st.button("▶️ START", type="primary", use_container_width=True):
                st.session_state.animation_started = True
                st.session_state.strategy = strategy_key
                st.session_state.speed = speed
                st.rerun()
            
            if st.button("🔄 Reset"):
                st.session_state.network_ready = False
                st.session_state.animation_started = False
                st.rerun()
    
    # Main content
    if not st.session_state.network_ready:
        st.info("👈 Configure a network to begin")
    
    elif not st.session_state.animation_started:
        network = st.session_state.current_network
        st.subheader(f"📊 {st.session_state.network_name}")
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Nodes", len(network.nodes))
        with c2:
            st.metric("Edges", len(network.edges))
        with c3:
            st.metric("Source", network.source)
        with c4:
            st.metric("Sink", network.sink)
        
        st.info("👈 Select algorithm and click START")
        
        with st.expander("🔍 Edges"):
            for (u, v), cap in network.edges.items():
                st.text(f"{u} → {v} : {cap}")
    
    else:
        network = st.session_state.current_network
        strategy = st.session_state.strategy
        speed = st.session_state.speed
        
        with st.spinner("Computing..."):
            animation_data = build_animation_data(network, strategy)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Iterations", animation_data['final']['total_iterations'])
        with c2:
            st.metric("Max Flow", animation_data['final']['max_flow'])
        with c3:
            st.metric("Strategy", strategy.upper())
        
        render_auto_animation(animation_data, step_duration=speed, height=680)
        
        # Show details in Streamlit too
        with st.expander("📋 Iteration Details"):
            for iter_data in animation_data['iterations']:
                path_parts = []
                for e in iter_data['path']:
                    arrow = "⟵" if e['is_reverse'] else "→"
                    path_parts.append(f"{e['source']} {arrow} {e['target']}")
                path_str = " | ".join(path_parts)
                reverse_note = " 🔄" if iter_data['uses_reverse_edge'] else ""
                
                st.markdown(f"**Iter {iter_data['iteration']}**{reverse_note}: `{path_str}` — Bottleneck: {iter_data['bottleneck']}, Total: {iter_data['total_flow']}")
        
        with st.expander("📊 Flow Decomposition (Final Paths)"):
            flow_paths = animation_data['final']['flow_paths']
            
            if flow_paths:
                st.markdown("**These paths carry the final maximum flow:**")
                
                total_check = 0
                for i, fp in enumerate(flow_paths, 1):
                    path_nodes = [fp['path'][0][0]] + [edge[1] for edge in fp['path']]
                    path_str = " → ".join(path_nodes)
                    st.markdown(f"**Path {i}:** `{path_str}` — **{fp['flow']} units**")
                    total_check += fp['flow']
                
                st.divider()
                st.markdown(f"**Total flow from decomposition:** {total_check}")
                st.markdown(f"**Maximum flow (verified):** {animation_data['final']['max_flow']}")
            else:
                st.write("No flow paths (max flow = 0)")
        
        with st.expander("📋 Final Edge Flows"):
            final_flows = animation_data['final']['final_flows']
            
            flow_data = []
            for (u, v), cap in network.edges.items():
                flow = final_flows.get(f"{u}-{v}", 0)
                status = "🔴 Saturated" if flow == cap and flow > 0 else ("⚪ Unused" if flow == 0 else "🟢 Active")
                flow_data.append({
                    "Edge": f"{u} → {v}",
                    "Flow": flow,
                    "Capacity": cap,
                    "Utilization": f"{(flow/cap*100):.0f}%" if cap > 0 else "N/A",
                    "Status": status
                })
            
            st.table(flow_data)


if __name__ == "__main__":
    main()
