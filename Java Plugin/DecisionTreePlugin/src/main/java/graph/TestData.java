package graph;

public class TestData {
    private String points;
    private double x_coordinate;
    private double y_coordinate;
    private double class_type;

    public TestData(double x_coordinate, double y_coordinate, double class_type, String points) {
        this.x_coordinate = x_coordinate;
        this.y_coordinate = y_coordinate;
        this.class_type = class_type;
        this.points = points;
    }

    public String getPoints() {
        return points;
    }

    public double getX_coordinate() {
        return x_coordinate;
    }

    public double getY_coordinate() {
        return y_coordinate;
    }

    public double getClass_type() {
        return class_type;
    }

    @Override
    public String toString() {
        return "TestData{" +
                "points=" + points +
                ", x_coordinate=" + x_coordinate +
                ", y_coordinate=" + y_coordinate +
                ", class='" + class_type + '\'' +
                '}';
    }
}
