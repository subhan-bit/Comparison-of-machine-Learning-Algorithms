package ml;

public interface Classifier {
    void train(Dataset trainingData);
    String predict(Example example);
}