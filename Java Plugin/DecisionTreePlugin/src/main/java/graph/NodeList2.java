package graph;

import java.util.Map;

public class NodeList2 {
    private String index;
    private Map<String, Object> properties;

    public NodeList2(String index2, Map<String, Object> properties) {
        this.index = index2;
        this.properties = properties;
    }

    public NodeList2() {
	}

	public String getIndex() {
        return index;
    }

    public void setIndex(String index) {
        this.index = index;
    }

    public Map<String, Object> getProperties() {
        return properties;
    }

    public void setProperties(Map<String, Object> properties) {
        this.properties = properties;
    }

    @Override
    public String toString() {
        return "NodeList{" +
                "index=" + index +
                ", properties=" + properties +
                '}';
    }
}
