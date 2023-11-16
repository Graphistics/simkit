package definition;

public class Nodelist {
    private String points;
    private Integer index;

    public double getX_cordinate() {
        return x_cordinate;
    }

    public double getY_cordinate() {
        return y_cordinate;
    }

    public double getClass_attrribute() {
        return class_attrribute;
    }

    private double x_cordinate;
    private double y_cordinate;
    private double class_attrribute;

    public Nodelist(String points, Integer index ,double x_cordinate,double y_cordinate,double class_attrribute ) {
        this.points = points;
        this.index = index;
        this.x_cordinate = x_cordinate;
        this.y_cordinate = y_cordinate;
        this.class_attrribute = class_attrribute;
    }

    public String getPoints() {
        return points;
    }

    public Integer getIndex() {
        return index;
    }
}
