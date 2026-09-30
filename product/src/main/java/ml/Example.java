package ml;

public class Example {
    private final double[] features;
    private final String label;

    public Example(double[] features, String label) {
        this.features = features;
        this.label = label;
    }

    public double[] getFeatures() { return features; }
    public String getLabel() { return label; }
    public int numFeatures() { return features.length; }
}