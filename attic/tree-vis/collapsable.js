function format(node) {
    var moves = 0;
    var f = "<tt>";
    for (var y = 0; y < 8; ++y) {
        for (var x = 0; x < 8; ++x) {
            var c = node["board"][x][y];
            if (c==".") moves += 1;

            if (c==" ") {
                f += "&nbsp; ";
            }
            else {
                f += c + " ";
            }
        }
        if(y!=7) f += "<br />";
    }
    if ("nn" in node) f += "<br />nn: " + node["nn"];
    f += "<br />wins: " + node["wins"] + " (" + (node["wins"]/node["games"]).toFixed(2) + ")";
    f += "<br />games: " + node["games"];
    if ("criteria" in node) f += "<br />criteria: " + node["criteria"];
    // f += "<br />moves: " + moves;
    // if ("children" in node) f += "<br />children: " + node["children"].length;
    f += "</tt>"
    return f;
}

function expandTree(root) {
    var tree = {};
    tree["innerHTML"] = format(root);
    tree["collapsed"] = true;

    if ("children" in root) {
        tree["children"] = [];

        for (child of root["children"]) {
            tree.children.push(expandTree(child));
        }
    }

    return tree;
}



var chart_config = {
    chart: {
        container: "#collapsable-example",

        animateOnInit: false,

        node: {
            collapsable: true
        },
        animation: {
            nodeAnimation: "easeOutBounce",
            nodeSpeed: 700,
            connectorsAnimation: "bounce",
            connectorsSpeed: 700
        },
        connectors: {
            style: {
                stroke: "white",
            }
        }
    },
    nodeStructure: expandTree(DATA)
};
