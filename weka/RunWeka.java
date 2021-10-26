package dtree.team;

import weka.core.converters.CSVLoader;
import weka.core.Instances;
import java.io.File;

import org.ho.yaml.Yaml;

import weka.classifiers.trees.J48;
import weka.classifiers.trees.j48.ClassifierTree;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.core.SerializationHelper;
import weka.classifiers.Evaluation;
import java.util.ArrayList;
import java.util.Scanner;
import java.io.FileReader;
import java.io.FileWriter;
import java.text.SimpleDateFormat;  
import java.util.Date;
import java.util.Enumeration;

/**
 * Na podstawie:
 * http://aragorn.pb.bialystok.pl/~grekowj/iaid/Wyklady/3.Weka%20in%20Java_v2.pdf
 * 
 * Algorytm:
 * https://weka.sourceforge.io/doc.dev/weka/classifiers/trees/J48.html
 */
public class RunWeka
{
    public static void main(String[] args) throws Exception
    {
        ArrayList<String> datasets = new ArrayList<>();
        try (Scanner s = new Scanner(new FileReader("./../datasets.txt"))) {
            while (s.hasNext()) {
                datasets.add(s.nextLine());
            }
        }
        for (String dataset: datasets) {
            
            // Datasets paths
            String trainDataFilepath = "./../datasets/" + dataset + "_trte.data"; 
            String testDataFilepath = "./../datasets/" + dataset + "_clean.data";

            // Loading data
            CSVLoader loader = new CSVLoader();
            loader.setNoHeaderRowPresent(true);
            loader.setSource(new File(trainDataFilepath));
            Instances trainData = loader.getDataSet();
            trainData.setClassIndex(trainData.numAttributes() - 1);
            loader.setSource(new File(testDataFilepath));
            Instances testData = loader.getDataSet();
            testData.setClassIndex(testData.numAttributes() - 1);

            // Formatting data
            NumericToNominal filter = new NumericToNominal();
            filter.setInputFormat(trainData);
            trainData = Filter.useFilter(trainData, filter);
            testData = Filter.useFilter(testData, filter);

            // Training
            CustomJ48 model = new CustomJ48();
            long start = System.nanoTime();
            model.buildClassifier(trainData);
            long finish = System.nanoTime();
            double timeElapsed = (finish - start) / (double) 1000000000;

            // Evaluating
            Evaluation evaluation = new Evaluation(trainData);
            evaluation.evaluateModel(model, testData);

            // Saving the serialized tree
            SerializationHelper.write("./../out/c4.5/c4.5_" + dataset + "_tree.model", model);

            // Saving the results
            FileWriter writer = new FileWriter("./../out/c4.5/c4.5_" + dataset + "_results.yml");
            writer.write("method: C4.5\n");
            writer.write("database: " + dataset + "\n");
            writer.write("confusion matrix:\n");
            double[][] confusionMatrix = evaluation.confusionMatrix();
            for (int i = 0; i < confusionMatrix.length; ++i) {
                for (int j = 0; j < confusionMatrix[i].length; ++j) {
                    writer.write(j == 0 ? "- " : "  ");
                    writer.write("- " + confusionMatrix[i][j]);
                    writer.write("\n");
                }
            }
            writer.write("tree:\n");
            writer.write("  size: " + (int) model.getMeasure("measureTreeSize") + "\n");
            writer.write("  height: " + model.getMaxDepth(model.getTreeRoot()) + "\n");
            writer.write("  serialization: c4.5_" + dataset + "_tree.model\n");
            SimpleDateFormat formatter = new SimpleDateFormat("dd-MM-yyyy HH:mm:ss z");  
            Date date = new Date();
            writer.write("time of experiment: " + formatter.format(date) + "\n");
            writer.write("cpu time: " + timeElapsed + "\n");
            writer.close();
        }
    }
}

class CustomJ48 extends J48
{
    public ClassifierTree getTreeRoot()
    {
        return m_root;
    }

    public int getMaxDepth(ClassifierTree node)
    {
        if (node.isLeaf()) {
            return 1;
        }
        int maxDepth = -1;
        for (ClassifierTree child: node.getSons()) {
            int childMaxDepth = getMaxDepth(child);
            if (childMaxDepth > maxDepth) {
                maxDepth = childMaxDepth;
            }
        }
        return maxDepth + 1;
    }
}
