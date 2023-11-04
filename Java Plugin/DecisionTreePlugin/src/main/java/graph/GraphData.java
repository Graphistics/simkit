package graph;

import java.util.ArrayList;

public class GraphData {
    public ArrayList<Nodelist> nodeList;
    public ArrayList<EdgeList> edgeList;

    public GraphData(ArrayList<Nodelist> nodeList, ArrayList<EdgeList> edgeList) {
        this.nodeList = nodeList;
        this.edgeList = edgeList;
    }


	public ArrayList<Nodelist> getNodeList() {
        return nodeList;
    }

    public void setNodeList(ArrayList<Nodelist> nodeList) {
        this.nodeList = nodeList;
    }

    public ArrayList<EdgeList> getEdgeList() {
        return edgeList;
    }

    public void setEdgeList(ArrayList<EdgeList> edgeList) {
        this.edgeList = edgeList;
    }
    
}

