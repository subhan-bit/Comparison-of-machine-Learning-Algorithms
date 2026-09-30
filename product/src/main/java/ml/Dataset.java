package ml;

import java.util.ArrayList;
import java.util.List;

public class Dataset {
    private final List<Example> examples = new ArrayList<>();

    public void add(Example e) { examples.add(e); }
    public Example get(int i) { return examples.get(i); }
    public int size() { return examples.size(); }
    public List<Example> getExamples() { return examples; }
}