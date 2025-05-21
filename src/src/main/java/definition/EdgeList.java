package definition;

public class EdgeList {
    private long source;
    private long target;

    private double weight;

    public Integer getIndex() {
        return index;
    }

    private Integer index;

    public EdgeList(long source, long target, double weight, Integer index) {
        this.source = source;
        this.target = target;
        this.weight = weight;
        this.index = index;
    }

    public long getSource() {
        return source;
    }

    public void setSource(long source) {
        this.source = source;
    }

    public long getTarget() {
        return target;
    }

    public void setTarget(long target) {
        this.target = target;
    }

    public double getWeight() {
        return weight;
    }

    public void setWeight(double weight) {
        this.weight = weight;
    }
    @Override
    public String toString() {
        return "EdgeList{" +
                "source=" + source +
                ", target=" + target +
                ", weight=" + weight +
                '}';
    }
}

