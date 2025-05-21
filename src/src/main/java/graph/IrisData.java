package graph;

public class IrisData {
    private int index;
    private double sepalLength;
    private double sepalWidth;
    private double petalLength;
    private double petalWidth;
    private String classType;

    public IrisData(int index, double sepalLength, double sepalWidth, double petalLength, double petalWidth, String classType) {
        this.index = index;
        this.sepalLength = sepalLength;
        this.sepalWidth = sepalWidth;
        this.petalLength = petalLength;
        this.petalWidth = petalWidth;
        this.classType = classType;
    }

    public int getIndex() {
        return index;
    }

    public double getSepalLength() {
        return sepalLength;
    }

    public double getSepalWidth() {
        return sepalWidth;
    }

    public double getPetalLength() {
        return petalLength;
    }

    public double getPetalWidth() {
        return petalWidth;
    }

    public String getClassType() {
        return classType;
    }

    @Override
    public String toString() {
        return "IrisData{" +
                "index=" + index +
                ", sepalLength=" + sepalLength +
                ", sepalWidth=" + sepalWidth +
                ", petalLength=" + petalLength +
                ", petalWidth=" + petalWidth +
                ", classType='" + classType + '\'' +
                '}';
    }
}
