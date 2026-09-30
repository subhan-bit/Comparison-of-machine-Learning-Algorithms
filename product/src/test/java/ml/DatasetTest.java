package ml;

import static org.junit.jupiter.api.Assertions.assertEquals;
import org.junit.jupiter.api.Test;

class DatasetTest {
    @Test
    void addingExamplesIncreasesSize() {
        Dataset d = new Dataset();
        d.add(new Example(new double[]{1.0, 2.0}, "A"));
        d.add(new Example(new double[]{3.0, 4.0}, "B"));
        assertEquals(2, d.size());
        assertEquals("B", d.get(1).getLabel());
    }
}