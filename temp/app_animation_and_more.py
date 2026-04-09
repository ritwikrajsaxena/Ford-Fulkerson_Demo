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
        self.nodes = set()
        self.edges = {}
        self.flow = {}
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
        if (u, v) in self.edges:
            return self.edges[(u, v)] - self.flow[(u, v)]
        elif (v, u) in self.edges:
            return self.flow[(v, u)]
        return 0
    
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
    
    def find_augmenting_path_dfs(self):
        if self.source is None or self.sink is None:
            return None, 0
        
        visited = set()
        parent = {}
        
        def dfs(u):
            if u == self.sink:
                return True
            visited.add(u)
            for v in self.nodes:
                if v not in visited and self.get_residual_capacity(u, v) > 0:
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
        T = self.nodes - S
        cut_edges = [(u, v) for u in S for v in T if (u, v) in self.edges]
        cut_capacity = sum(self.edges[(u, v)] for u, v in cut_edges)
        return S, T, cut_capacity, cut_edges
    
    def copy(self):
        new = FlowNetwork()
        new.nodes = self.nodes.copy()
        new.edges = self.edges.copy()
        new.flow = self.flow.copy()
        new.source = self.source
        new.sink = self.sink
        return new


def build_animation_data(network, strategy='bfs'):
    """Build complete animation timeline."""
    network = network.copy()
    
    data = {
        'nodes': list(network.nodes),
        'source': network.source,
        'sink': network.sink,
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
    while True:
        if strategy == 'bfs':
            path, bottleneck = network.find_augmenting_path_bfs()
        else:
            path, bottleneck = network.find_augmenting_path_dfs()
        
        if path is None:
            break
        
        iteration_num += 1
        iteration = {
            'iteration': iteration_num,
            'path': [{'source': u, 'target': v} for u, v in path],
            'bottleneck': bottleneck,
            'flow_before': {f"{u}-{v}": network.flow[(u, v)] for u, v in network.edges},
        }
        
        network.augment_flow(path, bottleneck)
        
        iteration['flow_after'] = {f"{u}-{v}": network.flow[(u, v)] for u, v in network.edges}
        iteration['total_flow'] = network.get_max_flow_value()
        
        data['iterations'].append(iteration)
    
    S, T, cut_cap, cut_edges = network.get_min_cut()
    data['final'] = {
        'max_flow': network.get_max_flow_value(),
        'S': list(S),
        'T': list(T),
        'cut_capacity': cut_cap,
        'cut_edges': [{'source': u, 'target': v} for u, v in cut_edges]
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
            'source': 's', 'sink': 't'
        },
        "Simple Diamond": {
            'nodes': ['s', 'a', 'b', 't'],
            'edges': [
                ('s', 'a', 10), ('s', 'b', 10),
                ('a', 'b', 2), ('a', 't', 10), ('b', 't', 10),
            ],
            'source': 's', 'sink': 't'
        },
        "Linear Chain": {
            'nodes': ['s', 'a', 'b', 'c', 't'],
            'edges': [
                ('s', 'a', 10), ('a', 'b', 8),
                ('b', 'c', 6), ('c', 't', 10),
            ],
            'source': 's', 'sink': 't'
        },
        "Parallel Paths": {
            'nodes': ['s', 'a', 'b', 'c', 'd', 't'],
            'edges': [
                ('s', 'a', 10), ('s', 'b', 10),
                ('a', 'c', 10), ('b', 'd', 10),
                ('c', 't', 10), ('d', 't', 10),
            ],
            'source': 's', 'sink': 't'
        },
        "Worst Case (DFS vs BFS)": {
            'nodes': ['s', 'u', 'v', 't'],
            'edges': [
                ('s', 'u', 100), ('s', 'v', 100),
                ('u', 'v', 1),
                ('u', 't', 100), ('v', 't', 100),
            ],
            'source': 's', 'sink': 't'
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
            'source': 's', 'sink': 't'
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
    """Generate a random flow network."""
    if seed is not None:
        random.seed(seed)
    
    network = FlowNetwork()
    nodes = ['s'] + [f'v{i}' for i in range(1, num_nodes - 1)] + ['t']
    
    for node in nodes:
        network.add_node(node)
    
    # Ensure path exists from s to t
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
# AUTO-PLAYING ANIMATION COMPONENT
# =============================================================================

def render_auto_animation(animation_data, step_duration=2000, height=700):
    """Render self-playing animation."""
    
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
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
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
            }}
            #title {{
                font-size: 20px;
                font-weight: 600;
                color: #4fc3f7;
            }}
            #stats {{
                display: flex;
                gap: 25px;
            }}
            .stat {{
                text-align: center;
            }}
            .stat-value {{
                font-size: 28px;
                font-weight: 700;
                color: #4fc3f7;
            }}
            .stat-label {{
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 1px;
                color: rgba(255,255,255,0.6);
            }}
            #cy {{
                flex: 1;
                background: radial-gradient(ellipse at center, #1a1a2e 0%, #0f0f1a 100%);
            }}
            #status-bar {{
                padding: 20px;
                background: rgba(0,0,0,0.3);
                border-top: 1px solid rgba(255,255,255,0.1);
            }}
            #phase-indicator {{
                display: flex;
                align-items: center;
                gap: 15px;
                margin-bottom: 15px;
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
                font-size: 16px;
                font-weight: 600;
                margin-bottom: 4px;
            }}
            #phase-description {{
                font-size: 13px;
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
                margin-top: 15px;
                padding: 12px 15px;
                background: rgba(79, 195, 247, 0.15);
                border: 1px solid rgba(79, 195, 247, 0.3);
                border-radius: 8px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 14px;
                display: none;
            }}
            #path-display.visible {{
                display: block;
            }}
            .path-arrow {{
                color: #4fc3f7;
                margin: 0 8px;
            }}
            .path-node {{
                background: rgba(79, 195, 247, 0.3);
                padding: 2px 8px;
                border-radius: 4px;
            }}
            .bottleneck {{
                color: #ffeb3b;
                margin-left: 15px;
                font-weight: 600;
            }}
            #completion {{
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0,0,0,0.85);
                display: none;
                align-items: center;
                justify-content: center;
                flex-direction: column;
                gap: 20px;
                z-index: 1000;
            }}
            #completion.visible {{
                display: flex;
            }}
            #completion h2 {{
                font-size: 32px;
                color: #00e676;
            }}
            #completion-stats {{
                display: flex;
                gap: 40px;
            }}
            #completion .stat-value {{
                font-size: 48px;
                color: #00e676;
            }}
            #restart-btn {{
                margin-top: 20px;
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
            #legend {{
                position: absolute;
                top: 80px;
                right: 20px;
                background: rgba(0,0,0,0.5);
                padding: 15px;
                border-radius: 8px;
                font-size: 12px;
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                gap: 10px;
                margin-bottom: 8px;
            }}
            .legend-color {{
                width: 20px;
                height: 4px;
                border-radius: 2px;
            }}
            .legend-node {{
                width: 14px;
                height: 14px;
                border-radius: 50%;
            }}
        </style>
    </head>
    <body>
        <div id="container">
            <div id="header">
                <div id="title">🌊 Ford-Fulkerson Algorithm</div>
                <div id="stats">
                    <div class="stat">
                        <div class="stat-value" id="iteration-value">0</div>
                        <div class="stat-label">Iteration</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="flow-value">0</div>
                        <div class="stat-label">Current Flow</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="total-iterations">-</div>
                        <div class="stat-label">Total Steps</div>
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
                    <span>Augmenting Path</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #ff5252;"></div>
                    <span>Saturated Edge</span>
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
                    <div class="stat-label">Min-Cut Capacity</div>
                </div>
            </div>
            <p style="color: rgba(255,255,255,0.7); margin-top: 10px;">
                Max-Flow Min-Cut Theorem: Maximum Flow = Minimum Cut ✓
            </p>
            <button id="restart-btn" onclick="restartAnimation()">🔄 Restart Animation</button>
        </div>
        
        <script>
            const DATA = {data_json};
            const STEP_DURATION = {step_duration};
            
            let cy = null;
            let currentIteration = 0;
            let isRunning = false;
            
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
                                'border-width': 5,
                                'background-color': '#4fc3f7'
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
                                'font-size': '14px',
                                'font-weight': 'bold',
                                'text-background-color': '#1a1a2e',
                                'text-background-opacity': 0.9,
                                'text-background-padding': '4px',
                                'text-margin-y': -15,
                                'color': '#fff'
                            }}
                        }},
                        {{
                            selector: 'edge.highlighted',
                            style: {{
                                'line-color': '#4fc3f7',
                                'target-arrow-color': '#4fc3f7',
                                'width': 8,
                                'z-index': 999
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
                                'width': 10
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
                
                document.getElementById('total-iterations').textContent = DATA.iterations.length;
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
            
            function showPath(path, bottleneck) {{
                const display = document.getElementById('path-display');
                const pathStr = path.map((e, i) => {{
                    if (i === 0) {{
                        return `<span class="path-node">${{e.source}}</span><span class="path-arrow">→</span><span class="path-node">${{e.target}}</span>`;
                    }}
                    return `<span class="path-arrow">→</span><span class="path-node">${{e.target}}</span>`;
                }}).join('');
                
                display.innerHTML = pathStr + `<span class="bottleneck">Bottleneck: ${{bottleneck}}</span>`;
                display.classList.add('visible');
            }}
            
            function hidePath() {{
                document.getElementById('path-display').classList.remove('visible');
            }}
            
            async function animateParticles(path) {{
                const promises = [];
                
                for (const edge of path) {{
                    const edgeId = `${{edge.source}}-${{edge.target}}`;
                    const cyEdge = cy.getElementById(edgeId);
                    
                    if (cyEdge.length > 0) {{
                        for (let i = 0; i < 5; i++) {{
                            promises.push(
                                delay(i * 100).then(() => createParticle(cyEdge))
                            );
                        }}
                    }}
                }}
                
                await Promise.all(promises);
            }}
            
            function createParticle(cyEdge) {{
                return new Promise(resolve => {{
                    const particle = document.createElement('div');
                    particle.className = 'particle';
                    document.body.appendChild(particle);
                    
                    const cyContainer = document.getElementById('cy');
                    const rect = cyContainer.getBoundingClientRect();
                    
                    const sourcePos = cyEdge.source().renderedPosition();
                    const targetPos = cyEdge.target().renderedPosition();
                    
                    const startX = rect.left + sourcePos.x;
                    const startY = rect.top + sourcePos.y;
                    const endX = rect.left + targetPos.x;
                    const endY = rect.top + targetPos.y;
                    
                    particle.style.left = startX + 'px';
                    particle.style.top = startY + 'px';
                    
                    const duration = 600;
                    
                    particle.animate([
                        {{ left: startX + 'px', top: startY + 'px', opacity: 1, transform: 'scale(1)' }},
                        {{ left: endX + 'px', top: endY + 'px', opacity: 0.3, transform: 'scale(0.5)' }}
                    ], {{
                        duration: duration,
                        easing: 'ease-out'
                    }}).onfinish = () => {{
                        particle.remove();
                        resolve();
                    }};
                }});
            }}
            
            async function runAnimation() {{
                isRunning = true;
                currentIteration = 0;
                
                updatePhase('🚀', 'Starting Algorithm', 'Initial flow is 0 on all edges');
                updateProgress(0);
                await delay(STEP_DURATION);
                
                for (let i = 0; i < DATA.iterations.length; i++) {{
                    currentIteration = i + 1;
                    const iter = DATA.iterations[i];
                    
                    document.getElementById('iteration-value').textContent = currentIteration;
                    
                    const progress = ((i + 0.5) / DATA.iterations.length) * 100;
                    updateProgress(progress);
                    
                    updatePhase('🔍', `Iteration ${{currentIteration}}: Finding Augmenting Path`, 
                               'Searching for a path from source to sink in residual network...');
                    await delay(STEP_DURATION * 0.5);
                    
                    for (const edge of iter.path) {{
                        const edgeId = `${{edge.source}}-${{edge.target}}`;
                        const cyEdge = cy.getElementById(edgeId);
                        if (cyEdge.length > 0) {{
                            cyEdge.addClass('highlighted');
                        }}
                        cy.getElementById(edge.source).addClass('active');
                        cy.getElementById(edge.target).addClass('active');
                        await delay(300);
                    }}
                    
                    showPath(iter.path, iter.bottleneck);
                    await delay(STEP_DURATION * 0.5);
                    
                    updatePhase('💧', `Pushing Flow: ${{iter.bottleneck}} units`, 
                               `Found path with bottleneck capacity ${{iter.bottleneck}}`);
                    
                    await animateParticles(iter.path);
                    await delay(STEP_DURATION * 0.3);
                    
                    updatePhase('📝', 'Updating Flow Values', 
                               'Updating edge flows and checking for saturation...');
                    
                    for (const [edgeId, newFlow] of Object.entries(iter.flow_after)) {{
                        const cyEdge = cy.getElementById(edgeId);
                        if (cyEdge.length > 0) {{
                            const capacity = cyEdge.data('capacity');
                            cyEdge.data('label', `${{newFlow}}/${{capacity}}`);
                            cyEdge.data('flow', newFlow);
                            
                            if (newFlow === capacity) {{
                                cyEdge.addClass('saturated');
                            }} else {{
                                cyEdge.removeClass('saturated');
                            }}
                        }}
                    }}
                    
                    document.getElementById('flow-value').textContent = iter.total_flow;
                    
                    await delay(STEP_DURATION * 0.5);
                    
                    cy.elements().removeClass('highlighted active');
                    hidePath();
                    
                    updateProgress(((i + 1) / DATA.iterations.length) * 100);
                    await delay(STEP_DURATION * 0.3);
                }}
                
                updatePhase('✂️', 'No More Augmenting Paths!', 
                           'Computing minimum cut...');
                await delay(STEP_DURATION);
                
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
                
                updatePhase('✅', 'Algorithm Complete!', 
                           `Max Flow = ${{DATA.final.max_flow}}, Min Cut = ${{DATA.final.cut_capacity}}`);
                updateProgress(100);
                
                await delay(STEP_DURATION);
                
                document.getElementById('final-flow').textContent = DATA.final.max_flow;
                document.getElementById('final-cut').textContent = DATA.final.cut_capacity;
                document.getElementById('completion').classList.add('visible');
                
                isRunning = false;
            }}
            
            function restartAnimation() {{
                document.getElementById('completion').classList.remove('visible');
                
                if (cy) {{
                    cy.destroy();
                }}
                
                document.getElementById('iteration-value').textContent = '0';
                document.getElementById('flow-value').textContent = '0';
                hidePath();
                
                initCytoscape();
                runAnimation();
            }}
            
            initCytoscape();
            setTimeout(runAnimation, 1000);
        </script>
    </body>
    </html>
    """
    
    components.html(html, height=height)


# =============================================================================
# SESSION STATE INITIALIZATION
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


# =============================================================================
# STREAMLIT APP
# =============================================================================

def main():
    st.set_page_config(
        page_title="Ford-Fulkerson Complete",
        page_icon="🌊",
        layout="wide"
    )
    
    init_session_state()
    
    # Custom CSS for cleaner look
    st.markdown("""
        <style>
            .block-container {
                padding-top: 1rem;
            }
            .stTabs [data-baseweb="tab-list"] {
                gap: 24px;
            }
            .stTabs [data-baseweb="tab"] {
                height: 50px;
                padding: 0 24px;
                font-weight: 600;
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🌊 Ford-Fulkerson Maximum Flow Visualizer")
    st.markdown("Complete interactive tool with **auto-playing animation** for any network configuration")
    
    # Sidebar for network configuration
    with st.sidebar:
        st.header("⚙️ Network Configuration")
        
        input_method = st.radio(
            "Select input method:",
            ["📚 Preset Examples", "✏️ Manual Entry", "🎲 Random Generation"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        network = None
        network_name = ""
        
        # ========== PRESET EXAMPLES ==========
        if input_method == "📚 Preset Examples":
            presets = get_presets()
            preset_name = st.selectbox(
                "Choose a preset network:",
                list(presets.keys())
            )
            
            with st.expander("ℹ️ Network Info"):
                preset = presets[preset_name]
                st.write(f"**Nodes:** {len(preset['nodes'])}")
                st.write(f"**Edges:** {len(preset['edges'])}")
                st.write(f"**Source:** {preset['source']}")
                st.write(f"**Sink:** {preset['sink']}")
            
            if st.button("✅ Load This Network", type="primary", use_container_width=True):
                network = create_network_from_preset(presets[preset_name])
                network_name = preset_name
                st.session_state.current_network = network
                st.session_state.network_name = network_name
                st.session_state.network_ready = True
                st.success(f"Loaded: {preset_name}")
                st.rerun()
        
        # ========== MANUAL ENTRY ==========
        elif input_method == "✏️ Manual Entry":
            st.subheader("Build Your Network")
            
            # Node management
            with st.expander("📍 Manage Nodes", expanded=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    new_node = st.text_input("Node name:", key="new_node_input", placeholder="e.g., v1")
                with col2:
                    st.write("")
                    st.write("")
                    if st.button("➕", key="add_node_btn", use_container_width=True):
                        if new_node and new_node.strip() and new_node not in st.session_state.manual_nodes:
                            st.session_state.manual_nodes.append(new_node.strip())
                            st.rerun()
                
                if len(st.session_state.manual_nodes) > 0:
                    st.write("**Current nodes:**")
                    cols = st.columns(4)
                    for i, node in enumerate(st.session_state.manual_nodes):
                        with cols[i % 4]:
                            if st.button(f"❌ {node}", key=f"del_node_{i}"):
                                # Check if node is used in edges
                                edges_using_node = [e for e in st.session_state.manual_edges 
                                                   if e[0] == node or e[1] == node]
                                if edges_using_node:
                                    st.error(f"Can't delete: {node} is used in edges")
                                else:
                                    st.session_state.manual_nodes.remove(node)
                                    st.rerun()
            
            # Edge management
            with st.expander("🔗 Manage Edges", expanded=True):
                if len(st.session_state.manual_nodes) >= 2:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        edge_from = st.selectbox("From:", st.session_state.manual_nodes, key="edge_from")
                    with col2:
                        other_nodes = [n for n in st.session_state.manual_nodes if n != edge_from]
                        edge_to = st.selectbox("To:", other_nodes, key="edge_to")
                    with col3:
                        edge_cap = st.number_input("Capacity:", min_value=1, value=10, key="edge_cap")
                    
                    if st.button("➕ Add Edge", key="add_edge_btn", use_container_width=True):
                        if (edge_from, edge_to) not in [(e[0], e[1]) for e in st.session_state.manual_edges]:
                            st.session_state.manual_edges.append((edge_from, edge_to, edge_cap))
                            st.rerun()
                        else:
                            st.warning("Edge already exists!")
                else:
                    st.info("Add at least 2 nodes first")
                
                if st.session_state.manual_edges:
                    st.write("**Current edges:**")
                    for i, (u, v, c) in enumerate(st.session_state.manual_edges):
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.write(f"`{u}` → `{v}` (capacity: {c})")
                        with col2:
                            if st.button("🗑️", key=f"del_edge_{i}"):
                                st.session_state.manual_edges.pop(i)
                                st.rerun()
            
            # Source/Sink selection
            st.subheader("Source & Sink")
            if len(st.session_state.manual_nodes) >= 2:
                source = st.selectbox("Source node:", st.session_state.manual_nodes, key="source_select")
                sink_options = [n for n in st.session_state.manual_nodes if n != source]
                sink = st.selectbox("Sink node:", sink_options, key="sink_select")
                
                # Build network button
                st.divider()
                if st.button("🚀 Build & Visualize", type="primary", use_container_width=True):
                    if len(st.session_state.manual_edges) == 0:
                        st.error("Add at least one edge!")
                    else:
                        network = FlowNetwork()
                        for node in st.session_state.manual_nodes:
                            network.add_node(node)
                        for u, v, c in st.session_state.manual_edges:
                            network.add_edge(u, v, c)
                        network.set_source_sink(source, sink)
                        
                        st.session_state.current_network = network
                        st.session_state.network_name = "Custom Network"
                        st.session_state.network_ready = True
                        st.success("Network built successfully!")
                        st.rerun()
                
                if st.button("🗑️ Clear All", key="clear_manual"):
                    st.session_state.manual_nodes = ['s', 't']
                    st.session_state.manual_edges = []
                    st.session_state.network_ready = False
                    st.rerun()
            else:
                st.info("Add at least 2 nodes to continue")
        
        # ========== RANDOM GENERATION ==========
        else:  # Random Generation
            st.subheader("Random Network Settings")
            
            num_nodes = st.slider("Number of nodes:", 4, 12, 6)
            edge_prob = st.slider("Edge probability:", 0.2, 0.8, 0.4, 0.1)
            
            col1, col2 = st.columns(2)
            with col1:
                min_cap = st.number_input("Min capacity:", 1, 50, 1)
            with col2:
                max_cap = st.number_input("Max capacity:", min_cap, 100, 20)
            
            seed = st.number_input("Random seed (0 for random):", 0, 9999, 42)
            
            if st.button("🎲 Generate Network", type="primary", use_container_width=True):
                actual_seed = seed if seed > 0 else None
                network = generate_random_network(num_nodes, edge_prob, min_cap, max_cap, actual_seed)
                
                st.session_state.current_network = network
                st.session_state.network_name = f"Random Network (seed={seed if seed > 0 else 'random'})"
                st.session_state.network_ready = True
                st.success("Network generated!")
                st.rerun()
        
        # ========== ALGORITHM SETTINGS ==========
        if st.session_state.network_ready:
            st.divider()
            st.header("🎬 Animation Settings")
            
            strategy = st.radio(
                "Path finding strategy:",
                ["BFS (Edmonds-Karp)", "DFS"],
                help="BFS finds shortest paths"
            )
            strategy_key = 'bfs' if 'BFS' in strategy else 'dfs'
            
            step_duration = st.slider(
                "Animation speed (ms per phase):",
                min_value=500,
                max_value=4000,
                value=2000,
                step=250,
                help="Lower = faster"
            )
            
            st.divider()
            
            if st.button("▶️ Start Animation", type="primary", use_container_width=True):
                st.session_state.start_animation = True
                st.session_state.strategy = strategy_key
                st.session_state.speed = step_duration
                st.rerun()
    
    # ========== MAIN CONTENT AREA ==========
    if not st.session_state.network_ready:
        # Show welcome message
        st.info("👈 Please configure a network from the sidebar to begin")
        
        # Show preview of a preset
        st.subheader("Preview: Textbook Example")
        presets = get_presets()
        preview_network = create_network_from_preset(presets["Textbook Example (CLRS Fig 26.1)"])
        preview_data = build_animation_data(preview_network, 'bfs')
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Nodes", len(preview_data['nodes']))
        with col2:
            st.metric("Edges", len(preview_data['edges']))
        with col3:
            st.metric("Iterations", len(preview_data['iterations']))
        with col4:
            st.metric("Max Flow", preview_data['final']['max_flow'])
        
        st.markdown("---")
        
        # Show features
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 📚 Preset Examples
            - Classic textbook networks
            - Educational examples
            - Worst-case scenarios
            - Bipartite matching
            """)
        
        with col2:
            st.markdown("""
            ### ✏️ Manual Entry
            - Build custom networks
            - Add/remove nodes
            - Define edge capacities
            - Choose source & sink
            """)
        
        with col3:
            st.markdown("""
            ### 🎲 Random Generation
            - Configurable size
            - Edge probability
            - Capacity ranges
            - Reproducible seeds
            """)
        
    else:
        # Show network info
        network = st.session_state.current_network
        
        st.subheader(f"📊 {st.session_state.network_name}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Nodes", len(network.nodes))
        with col2:
            st.metric("Edges", len(network.edges))
        with col3:
            st.metric("Source", network.source)
        with col4:
            st.metric("Sink", network.sink)
        
        # Check if animation should start
        if 'start_animation' in st.session_state and st.session_state.start_animation:
            st.markdown("---")
            
            # Build animation data
            with st.spinner("Computing algorithm timeline..."):
                animation_data = build_animation_data(
                    network,
                    st.session_state.get('strategy', 'bfs')
                )
            
            # Show animation
            render_auto_animation(
                animation_data,
                step_duration=st.session_state.get('speed', 2000),
                height=700
            )
            
            # Show iteration details
            with st.expander("📋 Algorithm Execution Details", expanded=False):
                st.write(f"**Total Iterations:** {len(animation_data['iterations'])}")
                st.write(f"**Maximum Flow:** {animation_data['final']['max_flow']}")
                st.write(f"**Minimum Cut Capacity:** {animation_data['final']['cut_capacity']}")
                st.write(f"**Min-Cut S:** {{{', '.join(animation_data['final']['S'])}}}")
                st.write(f"**Min-Cut T:** {{{', '.join(animation_data['final']['T'])}}}")
                
                st.divider()
                
                for i, iter_data in enumerate(animation_data['iterations'], 1):
                    path_str = " → ".join(
                        [iter_data['path'][0]['source']] +
                        [e['target'] for e in iter_data['path']]
                    )
                    st.markdown(f"""
                    **Iteration {i}:**
                    - Path: `{path_str}`
                    - Bottleneck: {iter_data['bottleneck']}
                    - Total Flow: {iter_data['total_flow']}
                    """)
        
        else:
            st.info("👈 Configure animation settings in the sidebar and click 'Start Animation'")
            
            # Show network structure
            with st.expander("🔍 Network Structure", expanded=True):
                st.write("**Edges:**")
                edge_data = []
                for (u, v), cap in network.edges.items():
                    edge_data.append({
                        "From": u,
                        "To": v,
                        "Capacity": cap
                    })
                st.table(edge_data)
    
    # Educational footer
    with st.expander("📚 About Ford-Fulkerson Algorithm"):
        st.markdown("""
        ### The Ford-Fulkerson Method
        
        Finds the **maximum flow** from source to sink in a flow network by:
        
        1. **Finding augmenting paths** in the residual network
        2. **Computing bottleneck** (minimum residual capacity on path)
        3. **Augmenting flow** by pushing bottleneck amount
        4. **Repeating** until no augmenting path exists
        
        ### Key Theorem
        
        **Max-Flow Min-Cut Theorem:** The maximum flow value equals the 
        minimum cut capacity. This visualization proves it by showing both!
        
        ### Path Finding Strategies
        
        - **BFS (Edmonds-Karp):** Finds shortest paths, runs in O(VE²)
        - **DFS:** May find longer paths, can be slower on some graphs
        
        ### Applications
        
        - Network routing and bandwidth allocation
        - Bipartite matching
        - Image segmentation
        - Supply chain optimization
        - Many more!
        """)


if __name__ == "__main__":
    main()
