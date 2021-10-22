package dtree.team;

import weka.core.converters.CSVLoader;
import weka.core.Instances;
import java.io.File;
import java.io.IOException;

public class RunWeka
{
    public static void main(String[] args) throws IOException
    {
        String dataset = "breast-cancer-wisconsin";
        String trainDataFilepath = "./../datasets/" + dataset + "_trte.data"; 
        String testDataFilepath = "./../datasets/" + dataset + "_clean.data";

        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(trainDataFilepath));
        Instances trainData = loader.getDataSet();

        System.out.println("Hello World!");
    }
}