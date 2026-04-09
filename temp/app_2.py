import streamlit as st
import streamlit.components.v1 as components
import json
from collections import deque
import random

# =============================================================================
# FORD-FULKERSON ALGORITHM IMPLEMENTATION (Same as before)
# =============================================================================

class FlowNetwork:
    """Represents a flow network with capacities and flows."""
    
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
        if (u, v) in self.edges:
            return self.edges[(u, v)] - self.flow[(u, v)]
        elif (v, u) in self.edges:
            return self.flow[(v, u)]
        else:
            return 0
    
    def get_residual_edges(self):
        residual = []
        for u in self.nodes:
            for v in self.nodes:
                if u != v:
                    cf = self.get_residual_capacity(u, v)
                    if cf > 0:
                        is_forward = (u, v) in self.edges
                        residual.append({
                            'source': u,
                            'target': v,
                            'residual_capacity': cf,
                            'is_forward': is_forward
                        })
        return residual
    
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
                if v not in visited:
                    cf = self.get_residual_capacity(u, v)
                    if cf > 0:
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
            cf = self.get_residual_capacity(u, v)
            bottleneck = min(bottleneck, cf)
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
                if v not in visited:
                    cf = self.get_residual_capacity(u, v)
                    if cf > 0:
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
            cf = self.get_residual_capacity(u, v)
            bottleneck = min(bottleneck, cf)
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
        total = 0
        for v in self.nodes:
            if (self.source, v) in self.flow:
                total += self.flow[(self.source, v)]
        return total
    
    def get_min_cut(self):
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
        cut_capacity = 0
        cut_edges = []
        
        for u in S:
            for v in T:
                if (u, v) in self.edges:
                    cut_capacity += self.edges[(u, v)]
                    cut_edges.append((u, v))
        
        return S, T, cut_capacity, cut_edges
    
    def copy(self):
        new_network = FlowNetwork()
        new_network.nodes = self.nodes.copy()
        new_network.edges = self.edges.copy()
        new_network.flow = self.flow.copy()
        new_network.source = self.source
        new_network.sink = self.sink
        return new_network


# =============================================================================
# ANIMATION STATE BUILDER
# =============================================================================

def build_animation_timeline(network, strategy='bfs'):
    """
    Execute Ford-Fulkerson and build complete animation timeline.
    Returns a data structure with all animation frames.
    """
    
    network = network.copy()
    timeline = {
        'nodes': list(network.nodes),
        'source': network.source,
        'sink': network.sink,
        'edges': [],
        'iterations': [],
        'final_state': {}
    }
    
    # Build edge list
    for (u, v), cap in network.edges.items():
        timeline['edges'].append({
            'id': f"{u}-{v}",
            'source': u,
            'target': v,
            'capacity': cap
        })
    
    # Run algorithm and capture each iteration
    iteration_num = 0
    
    while True:
        # Find augmenting path
        if strategy == 'bfs':
            path, bottleneck = network.find_augmenting_path_bfs()
        else:
            path, bottleneck = network.find_augmenting_path_dfs()
        
        if path is None:
            break
        
        iteration_num += 1
        
        # Capture current state before augmentation
        iteration_data = {
            'iteration': iteration_num,
            'path': [{'source': u, 'target': v} for u, v in path],
            'bottleneck': bottleneck,
            'flow_before': {},
            'flow_after': {},
            'new_residual_edges': []
        }
        
        # Record flows before
        for (u, v) in network.edges:
            iteration_data['flow_before'][f"{u}-{v}"] = network.get_flow(u, v)
        
        # Augment flow
        network.augment_flow(path, bottleneck)
        
        # Record flows after
        for (u, v) in network.edges:
            iteration_data['flow_after'][f"{u}-{v}"] = network.get_flow(u, v)
        
        # Find new residual edges (reverse edges that appeared)
        for u, v in path:
            if (v, u) not in network.edges and network.get_flow(u, v) > 0:
                iteration_data['new_residual_edges'].append({
                    'source': v,
                    'target': u,
                    'capacity': network.get_flow(u, v)
                })
        
        timeline['iterations'].append(iteration_data)
    
    # Final state with min-cut
    S, T, cut_cap, cut_edges = network.get_min_cut()
    timeline['final_state'] = {
        'max_flow': network.get_max_flow_value(),
        'min_cut_S': list(S),
        'min_cut_T': list(T),
        'cut_capacity': cut_cap,
        'cut_edges': [{'source': u, 'target': v} for u, v in cut_edges]
    }
    
    return timeline


# =============================================================================
# PRESET NETWORKS
# =============================================================================

def get_preset_networks():
    presets = {}
    
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
    network = FlowNetwork()
    for node in preset_data['nodes']:
        network.add_node(node)
    for u, v, cap in preset_data['edges']:
        network.add_edge(u, v, cap)
    network.set_source_sink(preset_data['source'], preset_data['sink'])
    return network


# =============================================================================
# ANIMATED VISUALIZATION COMPONENT
# =============================================================================

def render_animated_network(timeline_data, animation_speed=1.0, show_residual=True, height=600):
    """
    Render fully animated Ford-Fulkerson visualization.
    
    Parameters:
    - timeline_data: Complete animation timeline from build_animation_timeline()
    - animation_speed: Speed multiplier (1.0 = normal, 2.0 = 2x speed)
    - show_residual: Whether to show residual edges
    - height: Component height
    """
    
    timeline_json = json.dumps(timeline_data)
    speed = animation_speed
    show_res = 'true' if show_residual else 'false'
    
    html_code = f"""
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
                background: #f8f9fa;
            }}
            #cy {{
                width: 100%;
                height: {height - 120}px;
                background: #ffffff;
                border-bottom: 1px solid #dee2e6;
            }}
            #controls {{
                padding: 15px;
                background: #ffffff;
                border-bottom: 1px solid #dee2e6;
                display: flex;
                align-items: center;
                gap: 15px;
                flex-wrap: wrap;
            }}
            #info {{
                padding: 15px;
                background: #e9ecef;
                font-size: 14px;
                line-height: 1.6;
            }}
            button {{
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
                font-size: 14px;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.2s;
            }}
            button:hover:not(:disabled) {{
                transform: translateY(-1px);
                box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            }}
            button:disabled {{
                opacity: 0.5;
                cursor: not-allowed;
            }}
            #playBtn {{
                background: #28a745;
                color: white;
            }}
            #pauseBtn {{
                background: #ffc107;
                color: #333;
            }}
            #resetBtn {{
                background: #6c757d;
                color: white;
            }}
            #stepBtn {{
                background: #007bff;
                color: white;
            }}
            input[type="range"] {{
                width: 150px;
            }}
            .stat {{
                display: inline-block;
                padding: 4px 12px;
                background: #fff;
                border-radius: 4px;
                margin-right: 10px;
                border: 1px solid #dee2e6;
            }}
            .stat strong {{
                color: #007bff;
            }}
            #pathDisplay {{
                background: #fff;
                padding: 8px 12px;
                border-radius: 4px;
                border: 2px solid #007bff;
                font-weight: 500;
                color: #007bff;
            }}
            .pulse {{
                animation: pulse 1s ease-in-out infinite;
            }}
            @keyframes pulse {{
                0%, 100% {{ opacity: 1; }}
                50% {{ opacity: 0.5; }}
            }}
            .particle {{
                position: absolute;
                width: 8px;
                height: 8px;
                background: #007bff;
                border-radius: 50%;
                pointer-events: none;
                z-index: 1000;
            }}
        </style>
    </head>
    <body>
        <div id="controls">
            <button id="playBtn">▶ Play</button>
            <button id="pauseBtn" disabled>⏸ Pause</button>
            <button id="stepBtn">⏭ Step</button>
            <button id="resetBtn">↻ Reset</button>
            <div style="display: flex; align-items: center; gap: 8px;">
                <label>Speed:</label>
                <input type="range" id="speedSlider" min="0.5" max="3" step="0.5" value="{speed}">
                <span id="speedDisplay">{speed}x</span>
            </div>
            <div style="margin-left: auto;">
                <span class="stat">Iteration: <strong id="iterationDisplay">0</strong></span>
                <span class="stat">Flow: <strong id="flowDisplay">0</strong></span>
            </div>
        </div>
        
        <div id="cy"></div>
        
        <div id="info">
            <div id="pathDisplay">Ready to start. Click Play or Step.</div>
        </div>
        
        <script>
            // ============== GLOBAL STATE ==============
            const TIMELINE = {timeline_json};
            const SHOW_RESIDUAL = {show_res};
            
            let currentIteration = 0;
            let isPlaying = false;
            let animationSpeed = {speed};
            let cy = null;
            let animationPhase = 'idle';  // idle, highlighting, flowing, updating
            
            // ============== INITIALIZE CYTOSCAPE ==============
            function initializeCytoscape() {{
                const elements = [];
                
                // Add nodes
                TIMELINE.nodes.forEach(node => {{
                    let nodeClass = '';
                    if (node === TIMELINE.source) nodeClass = 'source';
                    if (node === TIMELINE.sink) nodeClass = 'sink';
                    
                    elements.push({{
                        data: {{ id: node, label: node }},
                        classes: nodeClass
                    }});
                }});
                
                // Add edges
                TIMELINE.edges.forEach(edge => {{
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
                                'background-color': '#6c757d',
                                'label': 'data(label)',
                                'text-valign': 'center',
                                'text-halign': 'center',
                                'color': '#fff',
                                'font-size': '16px',
                                'font-weight': 'bold',
                                'width': '50px',
                                'height': '50px',
                                'border-width': 3,
                                'border-color': '#495057',
                                'transition-property': 'background-color, border-color',
                                'transition-duration': '0.3s'
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
                            selector: 'node.highlighted',
                            style: {{
                                'border-color': '#ffc107',
                                'border-width': 6
                            }}
                        }},
                        {{
                            selector: 'edge',
                            style: {{
                                'width': 4,
                                'line-color': '#adb5bd',
                                'target-arrow-color': '#adb5bd',
                                'target-arrow-shape': 'triangle',
                                'curve-style': 'bezier',
                                'label': 'data(label)',
                                'font-size': '13px',
                                'font-weight': 'bold',
                                'text-background-color': '#fff',
                                'text-background-opacity': 1,
                                'text-background-padding': '4px',
                                'text-margin-y': -12,
                                'color': '#212529',
                                'transition-property': 'line-color, width',
                                'transition-duration': '0.3s'
                            }}
                        }},
                        {{
                            selector: 'edge.highlighted',
                            style: {{
                                'line-color': '#007bff',
                                'target-arrow-color': '#007bff',
                                'width': 7,
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
                                'width': 3
                            }}
                        }}
                    ],
                    layout: {{
                        name: 'dagre',
                        rankDir: 'LR',
                        nodeSep: 100,
                        rankSep: 150,
                        padding: 40
                    }}
                }});
            }}
            
            // ============== ANIMATION FUNCTIONS ==============
            
            function delay(ms) {{
                return new Promise(resolve => setTimeout(resolve, ms / animationSpeed));
            }}
            
            async function animateIteration(iterData) {{
                const path = iterData.path;
                const bottleneck = iterData.bottleneck;
                
                // Phase 1: Highlight path edges sequentially
                updatePathDisplay(`Finding augmenting path...`);
                for (let i = 0; i < path.length; i++) {{
                    const edge = path[i];
                    const edgeId = `${{edge.source}}-${{edge.target}}`;
                    const cyEdge = cy.getElementById(edgeId);
                    
                    if (cyEdge.length > 0) {{
                        cyEdge.addClass('highlighted');
                        cy.getElementById(edge.source).addClass('highlighted');
                        cy.getElementById(edge.target).addClass('highlighted');
                        await delay(400);
                    }}
                }}
                
                // Phase 2: Show bottleneck
                updatePathDisplay(`Path found! Bottleneck = ${{bottleneck}}`);
                await delay(600);
                
                // Phase 3: Animate flow (create particles)
                updatePathDisplay(`Pushing ${{bottleneck}} units of flow...`);
                await animateFlowParticles(path, bottleneck);
                
                // Phase 4: Update flow values with animation
                updatePathDisplay(`Updating flow values...`);
                for (const [edgeId, newFlow] of Object.entries(iterData.flow_after)) {{
                    const cyEdge = cy.getElementById(edgeId);
                    if (cyEdge.length > 0) {{
                        const capacity = cyEdge.data('capacity');
                        const oldFlow = cyEdge.data('flow');
                        
                        // Animate flow number counting up
                        await animateFlowValue(cyEdge, oldFlow, newFlow, capacity);
                        
                        // Mark as saturated if needed
                        if (newFlow === capacity) {{
                            cyEdge.addClass('saturated');
                        }} else {{
                            cyEdge.removeClass('saturated');
                        }}
                    }}
                }}
                
                // Phase 5: Show residual edges appearing
                if (SHOW_RESIDUAL && iterData.new_residual_edges.length > 0) {{
                    updatePathDisplay(`Adding residual edges...`);
                    for (const resEdge of iterData.new_residual_edges) {{
                        const edgeId = `${{resEdge.source}}-${{resEdge.target}}`;
                        
                        // Check if edge already exists
                        if (cy.getElementById(edgeId).length === 0) {{
                            cy.add({{
                                data: {{
                                    id: edgeId,
                                    source: resEdge.source,
                                    target: resEdge.target,
                                    label: `${{resEdge.capacity}}`,
                                    capacity: resEdge.capacity,
                                    flow: 0
                                }},
                                classes: 'residual'
                            }});
                            
                            // Fade in animation
                            const newEdge = cy.getElementById(edgeId);
                            newEdge.style('opacity', 0);
                            newEdge.animate({{
                                style: {{ opacity: 1 }}
                            }}, {{
                                duration: 500 / animationSpeed
                            }});
                        }}
                    }}
                    await delay(600);
                }}
                
                // Phase 6: Clear highlights
                cy.elements().removeClass('highlighted');
                await delay(400);
                
                // Update display
                const totalFlow = Object.values(iterData.flow_after).reduce((a, b) => a + b, 0) / 2;
                document.getElementById('flowDisplay').textContent = Math.round(totalFlow);
            }}
            
            async function animateFlowValue(cyEdge, oldFlow, newFlow, capacity) {{
                const steps = 10;
                const stepDuration = 300 / steps;
                
                for (let i = 0; i <= steps; i++) {{
                    const currentFlow = oldFlow + (newFlow - oldFlow) * (i / steps);
                    cyEdge.data('label', `${{Math.round(currentFlow)}}/${{capacity}}`);
                    cyEdge.data('flow', currentFlow);
                    await delay(stepDuration);
                }}
            }}
            
            async function animateFlowParticles(path, bottleneck) {{
                const particlePromises = [];
                
                for (const edge of path) {{
                    const edgeId = `${{edge.source}}-${{edge.target}}`;
                    const cyEdge = cy.getElementById(edgeId);
                    
                    if (cyEdge.length > 0) {{
                        // Create 3 particles with staggered timing
                        for (let i = 0; i < 3; i++) {{
                            particlePromises.push(
                                delay(i * 150).then(() => createFlowParticle(cyEdge))
                            );
                        }}
                    }}
                }}
                
                await Promise.all(particlePromises);
                await delay(800);  // Wait for particles to complete
            }}
            
            function createFlowParticle(cyEdge) {{
                return new Promise(resolve => {{
                    const particle = document.createElement('div');
                    particle.className = 'particle';
                    document.body.appendChild(particle);
                    
                    const sourcePos = cyEdge.source().renderedPosition();
                    const targetPos = cyEdge.target().renderedPosition();
                    
                    particle.style.left = sourcePos.x + 'px';
                    particle.style.top = sourcePos.y + 'px';
                    
                    const duration = 800 / animationSpeed;
                    
                    particle.animate([
                        {{ left: sourcePos.x + 'px', top: sourcePos.y + 'px', opacity: 1 }},
                        {{ left: targetPos.x + 'px', top: targetPos.y + 'px', opacity: 0.3 }}
                    ], {{
                        duration: duration,
                        easing: 'ease-in-out'
                    }}).onfinish = () => {{
                        particle.remove();
                        resolve();
                    }};
                }});
            }}
            
            async function showFinalCut() {{
                updatePathDisplay(`Algorithm complete! Showing minimum cut...`);
                
                const S = TIMELINE.final_state.min_cut_S;
                const T = TIMELINE.final_state.min_cut_T;
                const cutEdges = TIMELINE.final_state.cut_edges;
                
                // Highlight S nodes
                S.forEach(nodeId => {{
                    cy.getElementById(nodeId).style({{
                        'border-color': '#ffc107',
                        'border-width': 6
                    }});
                }});
                
                // Highlight T nodes
                T.forEach(nodeId => {{
                    cy.getElementById(nodeId).style({{
                        'border-color': '#17a2b8',
                        'border-width': 6
                    }});
                }});
                
                // Highlight cut edges
                cutEdges.forEach(edge => {{
                    const edgeId = `${{edge.source}}-${{edge.target}}`;
                    cy.getElementById(edgeId).style({{
                        'line-color': '#ffc107',
                        'target-arrow-color': '#ffc107',
                        'width': 8
                    }});
                }});
                
                updatePathDisplay(
                    `✅ Maximum Flow = ${{TIMELINE.final_state.max_flow}} | ` +
                    `Minimum Cut Capacity = ${{TIMELINE.final_state.cut_capacity}} | ` +
                    `S = \\{${{S.join(', ')}}\\} | T = \\{${{T.join(', ')}}\\}`
                );
            }}
            
            // ============== PLAYBACK CONTROLS ==============
            
            async function playAnimation() {{
                if (currentIteration >= TIMELINE.iterations.length) {{
                    await showFinalCut();
                    isPlaying = false;
                    updateControls();
                    return;
                }}
                
                isPlaying = true;
                updateControls();
                
                while (isPlaying && currentIteration < TIMELINE.iterations.length) {{
                    await stepAnimation();
                }}
                
                if (currentIteration >= TIMELINE.iterations.length) {{
                    await showFinalCut();
                }}
                
                isPlaying = false;
                updateControls();
            }}
            
            async function stepAnimation() {{
                if (currentIteration >= TIMELINE.iterations.length) {{
                    await showFinalCut();
                    return;
                }}
                
                const iterData = TIMELINE.iterations[currentIteration];
                document.getElementById('iterationDisplay').textContent = iterData.iteration;
                
                await animateIteration(iterData);
                
                currentIteration++;
                updateControls();
            }}
            
            function pauseAnimation() {{
                isPlaying = false;
                updateControls();
            }}
            
            function resetAnimation() {{
                isPlaying = false;
                currentIteration = 0;
                
                // Re-initialize cytoscape
                if (cy) {{
                    cy.destroy();
                }}
                initializeCytoscape();
                
                document.getElementById('iterationDisplay').textContent = '0';
                document.getElementById('flowDisplay').textContent = '0';
                updatePathDisplay('Ready to start. Click Play or Step.');
                updateControls();
            }}
            
            function updateControls() {{
                document.getElementById('playBtn').disabled = isPlaying || currentIteration >= TIMELINE.iterations.length;
                document.getElementById('pauseBtn').disabled = !isPlaying;
                document.getElementById('stepBtn').disabled = isPlaying || currentIteration >= TIMELINE.iterations.length;
            }}
            
            function updatePathDisplay(text) {{
                document.getElementById('pathDisplay').innerHTML = text;
            }}
            
            // ============== EVENT LISTENERS ==============
            
            document.getElementById('playBtn').addEventListener('click', playAnimation);
            document.getElementById('pauseBtn').addEventListener('click', pauseAnimation);
            document.getElementById('stepBtn').addEventListener('click', stepAnimation);
            document.getElementById('resetBtn').addEventListener('click', resetAnimation);
            
            document.getElementById('speedSlider').addEventListener('input', (e) => {{
                animationSpeed = parseFloat(e.target.value);
                document.getElementById('speedDisplay').textContent = animationSpeed + 'x';
            }});
            
            // ============== INITIALIZE ==============
            initializeCytoscape();
            updateControls();
        </script>
    </body>
    </html>
    """
    
    components.html(html_code, height=height)


# =============================================================================
# STREAMLIT APP
# =============================================================================

def main():
    st.set_page_config(
        page_title="Ford-Fulkerson Animated",
        page_icon="🌊",
        layout="wide"
    )
    
    st.title("🌊 Ford-Fulkerson Maximum Flow - Fully Animated")
    st.markdown("""
    Watch the algorithm come to life with **smooth 60 FPS animations**! 
    See augmenting paths highlighted, flow particles moving through edges, 
    and values updating in real-time.
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Network Configuration")
        
        presets = get_preset_networks()
        preset_name = st.selectbox(
            "Choose a network:",
            list(presets.keys())
        )
        
        st.divider()
        
        st.header("🎬 Animation Settings")
        
        strategy = st.radio(
            "Path finding strategy:",
            ["BFS (Edmonds-Karp)", "DFS"],
            help="BFS finds shortest paths"
        )
        strategy_key = 'bfs' if 'BFS' in strategy else 'dfs'
        
        animation_speed = st.slider(
            "Animation speed:",
            min_value=0.5,
            max_value=3.0,
            value=1.0,
            step=0.5,
            help="1.0 = normal speed, 2.0 = 2x faster"
        )
        
        show_residual = st.checkbox(
            "Show residual edges",
            value=True,
            help="Display reverse edges as they appear"
        )
        
        st.divider()
        
        if st.button("🚀 Load & Animate", type="primary", use_container_width=True):
            st.session_state.loaded = True
            st.session_state.preset_name = preset_name
            st.session_state.strategy = strategy_key
            st.session_state.speed = animation_speed
            st.session_state.show_res = show_residual
    
    # Main content
    if 'loaded' not in st.session_state:
        st.info("👈 Select a network and click 'Load & Animate' to begin")
        
        # Preview
        st.subheader("Preview: Textbook Example")
        example = create_network_from_preset(presets["Textbook Example (Fig 26.1)"])
        timeline = build_animation_timeline(example, 'bfs')
        
        with st.expander("Network Structure", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Nodes", len(timeline['nodes']))
            with col2:
                st.metric("Edges", len(timeline['edges']))
            with col3:
                st.metric("Iterations", len(timeline['iterations']))
        
        return
    
    # Build timeline
    with st.spinner("Computing algorithm timeline..."):
        network = create_network_from_preset(presets[st.session_state.preset_name])
        timeline = build_animation_timeline(network, st.session_state.strategy)
    
    # Show stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Network", st.session_state.preset_name)
    with col2:
        st.metric("Iterations", len(timeline['iterations']))
    with col3:
        st.metric("Max Flow", timeline['final_state']['max_flow'])
    with col4:
        st.metric("Min Cut", timeline['final_state']['cut_capacity'])
    
    st.divider()
    
    # Render animation
    render_animated_network(
        timeline,
        animation_speed=st.session_state.speed,
        show_residual=st.session_state.show_res,
        height=650
    )
    
    # Show iteration details
    with st.expander("📋 Iteration Details", expanded=False):
        for iter_data in timeline['iterations']:
            path_str = " → ".join(
                [iter_data['path'][0]['source']] + 
                [e['target'] for e in iter_data['path']]
            )
            st.markdown(f"""
            **Iteration {iter_data['iteration']}:**
            - Path: `{path_str}`
            - Bottleneck: {iter_data['bottleneck']}
            """)
    
    # Educational content
    with st.expander("📚 How the Animation Works", expanded=False):
        st.markdown("""
        ### Animation Phases (per iteration):
        
        1. **Path Highlighting** (400ms per edge)
           - Edges light up sequentially from source to sink
           - Nodes along path also highlighted
        
        2. **Bottleneck Display** (600ms)
           - Shows the minimum residual capacity
        
        3. **Flow Particles** (800ms)
           - Blue dots animate along the path
           - Represents flow being pushed
        
        4. **Value Updates** (300ms)
           - Flow labels count up smoothly
           - Edges turn red when saturated
        
        5. **Residual Edges** (600ms)
           - Reverse edges fade in (dashed cyan)
           - Shows flow can be "undone"
        
        6. **Cleanup** (400ms)
           - Highlights removed, ready for next iteration
        
        ### Controls:
        - **Play**: Run all iterations automatically
        - **Pause**: Stop at current iteration
        - **Step**: Execute one iteration at a time
        - **Reset**: Start over from beginning
        - **Speed Slider**: Adjust animation speed (0.5x to 3x)
        
        ### Technical Details:
        - Uses **Cytoscape.js** for graph rendering
        - **requestAnimationFrame** for smooth 60 FPS
        - **CSS transitions** for color changes
        - **JavaScript Promises** for async sequencing
        - All computation done in **Python**, animation in **JavaScript**
        """)


if __name__ == "__main__":
    main()
