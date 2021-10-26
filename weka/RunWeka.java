package dtree.team;

import weka.core.converters.CSVLoader;
import weka.core.Instances;
import java.io.File;

import org.ho.yaml.Yaml;

import weka.classifiers.trees.J48;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.core.SerializationHelper;
import weka.classifiers.Evaluation;
import java.util.ArrayList;
import java.util.Scanner;
import java.io.FileReader;

/**
 * Na podstawie:
 * http://aragorn.pb.bialystok.pl/~grekowj/iaid/Wyklady/3.Weka%20in%20Java_v2.pdf
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
            J48 model = new J48(); 
            model.buildClassifier(trainData);

            // Evaluating
            Evaluation evaluation = new Evaluation(trainData);
            evaluation.evaluateModel(model, testData);

            // Saving the serialized tree
            SerializationHelper.write("./../out/c4.5/c4.5_" + dataset + "_tree.model", model);

            // Saving the results
            double treeSize = model.getMeasure("measureTreeSize");


            Results results = new Results();
            results.method = "C4.5";
            Yaml.dump(results, new File("./object.yml"));
        }
    }
}

class Results
{
    public String method;
    public String database;

    public Results() {}
}