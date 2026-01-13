import pandas as pd
import json
import os
import networkx as nx

# ==========================================
# CONFIGURATION
# ==========================================
EDGES_FILE = '/project/home/p200253/hyperbolic/projections/files/graph/graph_edges_detailed.csv'
METADATA_FILE = '/project/home/p200253/hyperbolic/projections/files/combined_secondary_tertiary_skills.csv'
OUTPUT_HTML = '/project/home/p200253/hyperbolic/projections/files/graph/interactive_curriculum.html'

# ==========================================
# HTML TEMPLATE (D3.js v7)
# ==========================================
# This template is embedded directly to avoid external dependencies
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Interactive Skill Curriculum</title>
    <style>
        body { margin: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; overflow: hidden; background-color: #f8f9fa; }
        #controls {
            position: absolute; top: 10px; left: 10px; z-index: 10;
            background: rgba(255, 255, 255, 0.9); padding: 15px; border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1); border: 1px solid #ddd;
            max-width: 300px;
        }
        input[type="text"] { width: 100%; padding: 8px; margin-bottom: 10px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box;}
        .legend { display: flex; align-items: center; margin-bottom: 5px; font-size: 14px; }
        .dot { width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; display: inline-block; }
        #info-box {
            margin-top: 10px; padding-top: 10px; border-top: 1px solid #eee; font-size: 13px; color: #333;
            max-height: 400px; overflow-y: auto;
        }
        .node text { pointer-events: none; font-size: 10px; fill: #555; text-shadow: 1px 1px 0 #fff, -1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff; }
        .link { stroke-opacity: 0.3; stroke: #999; }
    </style>
    <script src="https://d3js.org/d3.v7.min.js"></script>
</head>
<body>

<div id="controls">
    <h3>Skill Curriculum Map</h3>
    <input type="text" id="search" placeholder="Search for a skill..." oninput="searchNode()">
    <div class="legend"><span class="dot" style="background:#1f77b4"></span>Secondary (Foundation)</div>
    <div class="legend"><span class="dot" style="background:#ff7f0e"></span>Tertiary (Specialized)</div>
    <div id="info-box">Hover over a node to see details.</div>
    <button onclick="resetZoom()" style="margin-top:10px; width:100%; padding:5px; cursor:pointer;">Reset View</button>
</div>

<svg id="viz"></svg>

<script>
    // --- DATA INJECTION POINT ---
    const graph = __GRAPH_DATA__;
    // ----------------------------

    const width = window.innerWidth;
    const height = window.innerHeight;

    const svg = d3.select("#viz")
        .attr("width", width)
        .attr("height", height)
        .call(d3.zoom().on("zoom", (event) => {
            g.attr("transform", event.transform);
        }))
        .append("g");

    const g = svg.append("g");

    // Force Simulation
    // We add a forceX to push Secondary left and Tertiary right slightly
    const simulation = d3.forceSimulation(graph.nodes)
        .force("link", d3.forceLink(graph.links).id(d => d.id).distance(50))
        .force("charge", d3.forceManyBody().strength(-30))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collide", d3.forceCollide().radius(8).iterations(2));
    
    // Add Arrows
    svg.append("defs").selectAll("marker")
        .data(["end"])
        .enter().append("marker")
        .attr("id", "arrow")
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", 15)
        .attr("refY", 0)
        .attr("markerWidth", 6)
        .attr("markerHeight", 6)
        .attr("orient", "auto")
        .append("path")
        .attr("d", "M0,-5L10,0L0,5")
        .attr("fill", "#999");

    const link = g.append("g")
        .attr("class", "links")
        .selectAll("line")
        .data(graph.links)
        .enter().append("line")
        .attr("class", "link")
        .attr("stroke-width", d => Math.sqrt(d.value))
        .attr("marker-end", "url(#arrow)");

    const node = g.append("g")
        .attr("class", "nodes")
        .selectAll("g")
        .data(graph.nodes)
        .enter().append("g")
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended));

    // Draw Circles
    node.append("circle")
        .attr("r", 5)
        .attr("fill", d => d.group === 'secondary' ? "#1f77b4" : "#ff7f0e")
        .attr("stroke", "#fff")
        .attr("stroke-width", 1.5)
        .on("mouseover", mouseOver)
        .on("mouseout", mouseOut)
        .on("click", showInfo);

    // Draw Labels (only visible on zoom or if high degree)
    const text = node.append("text")
        .attr("dx", 8)
        .attr("dy", ".35em")
        .text(d => d.name)
        .style("opacity", d => d.degree > 10 ? 1 : 0); // Hide low degree labels initially

    simulation.on("tick", () => {
        link
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

        node
            .attr("transform", d => `translate(${d.x},${d.y})`);
    });

    // --- INTERACTION FUNCTIONS ---

    function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }

    // Highlighting
    const linkedByIndex = {};
    graph.links.forEach(d => {
        linkedByIndex[`${d.source.id},${d.target.id}`] = 1;
    });

    function isConnected(a, b) {
        return linkedByIndex[`${a.id},${b.id}`] || linkedByIndex[`${b.id},${a.id}`] || a.id === b.id;
    }

    function mouseOver(event, d) {
        // Dim all
        node.style("opacity", 0.1);
        link.style("opacity", 0.05);
        text.style("opacity", 0);

        // Highlight neighbors
        const neighbors = new Set([d.id]);
        
        link.style("stroke", function(l) {
            if (l.source.id === d.id) { neighbors.add(l.target.id); return "green"; } // Outgoing
            if (l.target.id === d.id) { neighbors.add(l.source.id); return "red"; }   // Incoming
            return "#999";
        }).style("opacity", function(l) {
            return (l.source.id === d.id || l.target.id === d.id) ? 1 : 0.05;
        });

        node.style("opacity", function(n) {
            return (neighbors.has(n.id)) ? 1 : 0.1;
        });
        
        text.style("opacity", function(n) {
             return (neighbors.has(n.id)) ? 1 : 0;
        });

        showInfo(event, d);
    }

    function mouseOut() {
        node.style("opacity", 1);
        link.style("stroke", "#999").style("opacity", 0.3);
        text.style("opacity", d => d.degree > 10 ? 1 : 0);
    }

    function showInfo(event, d) {
        const info = document.getElementById('info-box');
        info.innerHTML = `
            <strong>${d.name}</strong><br>
            <span style="color:${d.group === 'secondary' ? '#1f77b4' : '#ff7f0e'}">
                ${d.group.toUpperCase()}
            </span><br><br>
            <em>${d.description || "No description available."}</em><br><br>
            Connections: ${d.degree}
        `;
    }

    function searchNode() {
        const term = document.getElementById('search').value.toLowerCase();
        if (!term) { mouseOut(); return; }

        const found = graph.nodes.find(n => n.name.toLowerCase().includes(term));
        if (found) {
            // Focus on node
            mouseOver(null, found);
            
            // Optional: Zoom to node (simple jump)
            // In a real app we'd animate the view transform
        }
    }
    
    function resetZoom() {
        // Reload page or reset transform (simplified)
        location.reload(); 
    }

</script>
</body>
</html>
"""

# ==========================================
# MAIN GENERATOR
# ==========================================
def main():
    print("Generating Interactive Graph...")
    
    # 1. Load Data
    edges_df = pd.read_csv(EDGES_FILE)
    meta_df = pd.read_csv(METADATA_FILE)
    
    # Create Mappings
    id_to_name = dict(zip(meta_df['skill_uid'], meta_df['name']))
    id_to_desc = dict(zip(meta_df['skill_uid'], meta_df['description']))
    id_to_group = dict(zip(meta_df['skill_uid'], meta_df['education_level']))
    
    # 2. Build Graph Structure for D3
    G = nx.from_pandas_edgelist(edges_df, 'source_id', 'target_id', create_using=nx.DiGraph())
    
    # Calculate degrees for sizing/filtering
    degrees = dict(G.degree())
    
    nodes_data = []
    # Ensure we include all nodes in metadata, or just those in the graph
    # Let's stick to those in the graph to avoid isolated dots
    for uid in G.nodes():
        nodes_data.append({
            "id": uid,
            "name": id_to_name.get(uid, uid),
            "group": id_to_group.get(uid, "unknown"),
            "description": str(id_to_desc.get(uid, "")).replace('"', "'"), # Escape quotes
            "degree": degrees.get(uid, 0)
        })
        
    links_data = []
    for _, row in edges_df.iterrows():
        links_data.append({
            "source": row['source_id'],
            "target": row['target_id'],
            "value": 1
        })
        
    graph_json = {
        "nodes": nodes_data,
        "links": links_data
    }
    
    # 3. Inject into HTML
    json_str = json.dumps(graph_json)
    final_html = HTML_TEMPLATE.replace('__GRAPH_DATA__', json_str)
    
    # 4. Save
    os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)
    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(final_html)
        
    print(f"✅ Success! Open this file in your browser:\n{OUTPUT_HTML}")

if __name__ == "__main__":
    main()