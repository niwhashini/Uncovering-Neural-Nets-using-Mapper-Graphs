const width = window.innerWidth;
const height = window.innerHeight;

const svg = d3.select("#graph")
  .append("svg")
  .attr("width", width)
  .attr("height", height);

const g = svg.append("g");

// Zoom + pan
svg.call(
  d3.zoom().on("zoom", (event) => {
    g.attr("transform", event.transform);
  })
);

//  LOAD JSON 
d3.json("covid_9states.json").then(data => {
    let showPie = false;
    let showLabels = true;   // labels ON by default
    const nodes = data.nodes;
    const links = data.links;

    const labelGroup = g.append("g")
                        .attr("class", "labels");
    // one label per dominant class
    const labels = [];

    Object.keys(data.state_names).forEach(classId => {
        // collect nodes with this class
        const classNodes = nodes.filter(n => n.dominant_class == classId);
        if (classNodes.length === 0) return;

        // create one label object for this class
        labels.push({
            classId: +classId,
            name: data.state_names[classId],
            nodes: classNodes,
            x: 0,
            y: 0
        });
    });

    const labelTexts = labelGroup.selectAll("text")
    .data(labels)
    .enter()
    .append("text")
    .attr("font-size", 10)
    .attr("fill", "#fff")
    .attr("text-anchor", "middle")
    .text(d => d.name);

    labelGroup.raise();


    const classCount = Object.keys(data.state_names).length;
    const classColor = d3.scaleOrdinal(d3.schemeCategory10)
                        .domain(d3.range(classCount));
    
    // CONFIRMED COLOR (continuous red scale)
    const confirmedExtent = d3.extent(nodes, d => d.avg_confirmed); 
    const confirmedColor = d3.scaleSequential()
                            .domain(confirmedExtent)
                            .interpolator(d3.interpolateReds);

    // declare simulation variable early so drag handlers can reference it
    let simulation;

    //  Draw links 
    const link = g.selectAll("line")
      .data(links)
      .enter()
      .append("line")
      .attr("stroke", "#aaa")
      .attr("stroke-width", 1);

    //  Draw nodes (circle view) 
    const circleGroup = g.append("g");
    const node = circleGroup.selectAll("circle")
      .data(nodes)
      .enter()
      .append("circle")
      .attr("r", d => {
            if (d.size) return Math.sqrt(d.size) + 3;
            if (d.members) return Math.sqrt(d.members.length) + 3;
            return 5;
        })
      .attr("fill", d => classColor(d.dominant_class));
    
    updateNodeColors("class");

    //  GROUP 2: Pie chart nodes 
    const pieGroup = g.append("g")
                    .attr("visibility", "hidden");  

    const pieGen = d3.pie()
        .value(([, count]) => count)
        .sort(null);

    const arc = d3.arc()
        .innerRadius(0)
        .outerRadius(6);   
    
    const pies = pieGroup.selectAll("g")
        .data(nodes)
        .enter()
        .append("g")
        .each(function(d) {

            // compute radius based on node size 
            const r = (d.size ? Math.sqrt(d.size) : Math.sqrt(d.members.length)) + 3;

            // arc generator specific to this node
            const arcGen = d3.arc()
                .innerRadius(0)
                .outerRadius(r);

            const counts = Object.entries(d.counts || {});
            const arcs = pieGen(counts);

            d3.select(this).selectAll("path")
            .data(arcs)
            .enter()
            .append("path")
            .attr("d", arcGen)
            .attr("fill", arcDatum => classColor(arcDatum.data[0]));
        })
        .call(
            d3.drag()
            .on("start", (event, d) => dragstarted(event, d, simulation))
            .on("drag", (event, d) => dragged(event, d))
            .on("end", (event, d) => dragended(event, d, simulation))
        );

    node.append("title")
    .text(d => `Node: ${d.id}
Dominant class: ${data.state_names[d.dominant_class] || d.dominant_class}
Purity: ${d.purity != null ? d.purity.toFixed(3) : "N/A"}
Average days: ${d.avg_days != null ? d.avg_days.toFixed(1) : "N/A"}
Average confirmed: ${d.avg_confirmed != null ? d.avg_confirmed.toFixed(1) : "N/A"}`);
    
    pies.append("title")
    .text(d => `Node: ${d.id}
Dominant class: ${data.state_names[d.dominant_class] || d.dominant_class}
Purity: ${d.purity != null ? d.purity.toFixed(3) : "N/A"}
Average days: ${d.avg_days != null ? d.avg_days.toFixed(1) : "N/A"}
Average confirmed: ${d.avg_confirmed != null ? d.avg_confirmed.toFixed(1) : "N/A"}
Members: ${d.members.length}`);


    console.log("Loaded:", nodes.length, "nodes,", links.length, "links");
    // -- update coloring scheme based on selection --
    function updateNodeColors(mode) {
    if (mode === "class") {
        node.attr("fill", d => classColor(d.dominant_class));
    } 
    else if (mode === "confirmed") {
        node.attr("fill", d => confirmedColor(d.avg_confirmed));
    }
}

    // Force simulation
    simulation = d3.forceSimulation(nodes)
        .force("link", d3.forceLink(links)
            .id(d => d.id)
            .distance(40)
        )
        .force("charge", d3.forceManyBody().strength(-50))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("radial", d3.forceRadial(200, width / 2, height / 2))
        .on("tick", ticked);

    node.call(
        d3.drag()
            .on("start", (event, d) => dragstarted(event, d, simulation))
            .on("drag", (event, d) => dragged(event, d))
            .on("end", (event, d) => dragended(event, d, simulation))
    );

    pies.call(
        d3.drag()
            .on("start", (event, d) => dragstarted(event, d, simulation))
            .on("drag", (event, d) => dragged(event, d))
            .on("end", (event, d) => dragended(event, d, simulation))
    );

    function ticked() {
        link
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

        node
            .attr("cx", d => d.x)
            .attr("cy", d => d.y);

        // update pies 
        pies
            .attr("transform", d => `translate(${d.x},${d.y})`);
        
             
        labelTexts
            .attr("x", function(d) {

                // find node furthest from center
                const furthest = d.nodes.reduce((a, b) => {
                    const ra = a.x*a.x + a.y*a.y;
                    const rb = b.x*b.x + b.y*b.y;
                    return rb > ra ? b : a;
                });

                // offset outward (radial push)
                const dx = furthest.x;
                const dy = furthest.y;
                const len = Math.sqrt(dx*dx + dy*dy);
                const ox = (dx / len) * 20;   // offset by 20px outward

                d.x = furthest.x + ox;
                return d.x;
            })
            .attr("y", function(d) {
                const furthest = d.nodes.reduce((a, b) => {
                    const ra = a.x*a.x + a.y*a.y;
                    const rb = b.x*b.x + b.y*b.y;
                    return rb > ra ? b : a;
                });

                const dx = furthest.x;
                const dy = furthest.y;
                const len = Math.sqrt(dx*dx + dy*dy);
                const oy = (dy / len) * 20;

                d.y = furthest.y + oy;
                return d.y;
            });
    }

    // drag helpers (use simulation arg where needed)
    function dragstarted(event, d, sim) {
        if (!event.active && sim) sim.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }
    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }
    function dragended(event, d, sim) {
        if (!event.active && sim) sim.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }

    // Toggle button
    document.getElementById("togglePie").onclick = () => {
        showPie = !showPie;

        if (showPie) {
            circleGroup.attr("visibility", "hidden");
            pieGroup.attr("visibility", "visible");
        } else {
            pieGroup.attr("visibility", "hidden");
            circleGroup.attr("visibility", "visible");
        }
    };
    document.getElementById("toggleLabels").onclick = () => {
        showLabels = !showLabels;

        if (showLabels) {
            labelGroup.attr("visibility", "visible");
        } else {
            labelGroup.attr("visibility", "hidden");
        }
    };
    document.getElementById("infoNodes").textContent = nodes.length;
    document.getElementById("infoLinks").textContent = links.length;
    document.getElementById("infoClasses").textContent = Object.keys(data.state_names).length;
    node.on("click", (event, d) => showSelectedNode(d));
    pies.on("click", (event, d) => showSelectedNode(d));

    function showSelectedNode(d) {
        document.getElementById("infoNodeId").textContent = d.id;
        document.getElementById("infoNodeSize").textContent = d.size?.toFixed(1) ?? "–";
        document.getElementById("infoNodePurity").textContent = d.purity?.toFixed(3) ?? "–";
        document.getElementById("infoNodeMisrate").textContent = d.misrate ?? "–";
        document.getElementById("infoNodeConfirmed").textContent = d.avg_confirmed = d.avg_confirmed.toFixed(1) ?? "–";

        const countsDiv = document.getElementById("infoNodeCounts");
        countsDiv.innerHTML = ""; // clear

        if (d.counts) {
            Object.entries(d.counts).forEach(([key, count]) => {
                const name = data.state_names[key];
                const pct = (count / d.members.length * 100).toFixed(1);

                const row = document.createElement("div");
                row.textContent = `${name}: ${count} (${pct}%)`;
                countsDiv.appendChild(row);
            });
        }
    }
    document.getElementById("colorMode").onchange = (e) => {
        const mode = e.target.value;
        updateNodeColors(mode);
    };

});
