import streamlit as st
import streamlit.components.v1 as components
import json
from collections import deque

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


def build_animation_data(network):
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
    
    while True:
        path, bottleneck = network.find_augmenting_path_bfs()
        if path is None:
            break
        
        iteration = {
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
    }


def create_network(preset):
    network = FlowNetwork()
    for node in preset['nodes']:
        network.add_node(node)
    for u, v, cap in preset['edges']:
        network.add_edge(u, v, cap)
    network.set_source_sink(preset['source'], preset['sink'])
    return network


# =============================================================================
# AUTO-PLAYING ANIMATION COMPONENT
# =============================================================================

def render_auto_animation(animation_data, step_duration=2000, height=700):
    """
    Render self-playing animation.
    
    step_duration: milliseconds to spend on each animation phase
    """
    
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
            
            /* Header */
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
            
            /* Graph */
            #cy {{
                flex: 1;
                background: radial-gradient(ellipse at center, #1a1a2e 0%, #0f0f1a 100%);
            }}
            
            /* Status Bar */
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
            
            /* Progress Bar */
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
            
            /* Path Display */
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
            
            /* Completion Screen */
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
            
            /* Particles */
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
            
            /* Legend */
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
            // ============== DATA ==============
            const DATA = {data_json};
            const STEP_DURATION = {step_duration};
            
            let cy = null;
            let currentIteration = 0;
            let isRunning = false;
            
            // ============== INITIALIZATION ==============
            function initCytoscape() {{
                const elements = [];
                
                // Nodes
                DATA.nodes.forEach(node => {{
                    let classes = '';
                    if (node === DATA.source) classes = 'source';
                    if (node === DATA.sink) classes = 'sink';
                    elements.push({{
                        data: {{ id: node, label: node }},
                        classes: classes
                    }});
                }});
                
                // Edges
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
            
            // ============== ANIMATION HELPERS ==============
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
            
            // ============== PARTICLE ANIMATION ==============
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
            
            // ============== MAIN ANIMATION LOOP ==============
            async function runAnimation() {{
                isRunning = true;
                currentIteration = 0;
                
                // Initial state
                updatePhase('🚀', 'Starting Algorithm', 'Initial flow is 0 on all edges');
                updateProgress(0);
                await delay(STEP_DURATION);
                
                // Process each iteration
                for (let i = 0; i < DATA.iterations.length; i++) {{
                    currentIteration = i + 1;
                    const iter = DATA.iterations[i];
                    
                    document.getElementById('iteration-value').textContent = currentIteration;
                    
                    const progress = ((i + 0.5) / DATA.iterations.length) * 100;
                    updateProgress(progress);
                    
                    // Phase 1: Finding path
                    updatePhase('🔍', `Iteration ${{currentIteration}}: Finding Augmenting Path`, 
                               'Searching for a path from source to sink in residual network...');
                    await delay(STEP_DURATION * 0.5);
                    
                    // Highlight path edges sequentially
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
                    
                    // Phase 2: Pushing flow
                    updatePhase('💧', `Pushing Flow: ${{iter.bottleneck}} units`, 
                               `Found path with bottleneck capacity ${{iter.bottleneck}}`);
                    
                    await animateParticles(iter.path);
                    await delay(STEP_DURATION * 0.3);
                    
                    // Phase 3: Updating flow values
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
                    
                    // Clear highlights
                    cy.elements().removeClass('highlighted active');
                    hidePath();
                    
                    updateProgress(((i + 1) / DATA.iterations.length) * 100);
                    await delay(STEP_DURATION * 0.3);
                }}
                
                // Final phase: Show min-cut
                updatePhase('✂️', 'No More Augmenting Paths!', 
                           'Computing minimum cut...');
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
                
                updatePhase('✅', 'Algorithm Complete!', 
                           `Max Flow = ${{DATA.final.max_flow}}, Min Cut = ${{DATA.final.cut_capacity}}`);
                updateProgress(100);
                
                await delay(STEP_DURATION);
                
                // Show completion screen
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
            
            // ============== START ==============
            initCytoscape();
            
            // Auto-start after a brief delay
            setTimeout(runAnimation, 1000);
        </script>
    </body>
    </html>
    """
    
    components.html(html, height=height)


# =============================================================================
# STREAMLIT APP
# =============================================================================

def main():
    st.set_page_config(
        page_title="Ford-Fulkerson Animation",
        page_icon="🌊",
        layout="wide"
    )
    
    # Minimal UI - just the animation
    st.markdown("""
        <style>
            .block-container {
                padding-top: 1rem;
                padding-bottom: 0rem;
            }
            header {visibility: hidden;}
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.title("🌊 Ford-Fulkerson")
        st.markdown("**Maximum Flow Algorithm**")
        
        st.divider()
        
        presets = get_presets()
        preset_name = st.selectbox(
            "Select Network:",
            list(presets.keys()),
            index=0
        )
        
        step_duration = st.slider(
            "Animation Speed (ms per phase):",
            min_value=500,
            max_value=4000,
            value=2000,
            step=250,
            help="Lower = faster animation"
        )
        
        st.divider()
        
        if st.button("🚀 Start Animation", type="primary", use_container_width=True):
            st.session_state.run_animation = True
            st.session_state.preset = preset_name
            st.session_state.speed = step_duration
            st.rerun()
        
        st.divider()
        
        with st.expander("ℹ️ About"):
            st.markdown("""
            **Ford-Fulkerson Method** finds the maximum flow 
            from source to sink in a flow network.
            
            **Key Concepts:**
            - 🟢 **Source**: Where flow originates
            - 🔴 **Sink**: Where flow terminates
            - 🔵 **Augmenting Path**: Path with available capacity
            - ⚪ **Bottleneck**: Minimum capacity on path
            - 🟡 **Min-Cut**: Partition proving optimality
            
            **Max-Flow Min-Cut Theorem:**  
            Maximum flow = Minimum cut capacity
            """)
    
    # Main content
    if 'run_animation' not in st.session_state:
        st.session_state.run_animation = True
        st.session_state.preset = list(presets.keys())[0]
        st.session_state.speed = 2000
    
    # Build and render animation
    network = create_network(presets[st.session_state.preset])
    animation_data = build_animation_data(network)
    
    render_auto_animation(
        animation_data,
        step_duration=st.session_state.speed,
        height=700
    )


if __name__ == "__main__":
    main()
