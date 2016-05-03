package com.stelmod.recommendations;

import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.model.DataModel;

import java.io.File;
import java.net.URL;

public class AlgorithmEvaluation {
    public static void main(String[] args) throws Exception {
        URL inputStream = Thread.currentThread().getContextClassLoader().getResource("ml-small.csv");

        File inputFile = new File(inputStream.toURI());
        DataModel dataModel = new FileDataModel(inputFile);

        long startTime = System.currentTimeMillis();
        NN_Evaluator nn_evaluator = new NN_Evaluator(dataModel, new int[] {2, 8, 20, 50, 100, 300});
        nn_evaluator.evaluateModel();

        SVD_evaluator svd_evaluator = new SVD_evaluator(dataModel, new int[] {5, 10, 25});
        svd_evaluator.evaluateModel();
        System.out.println("Elapsed time: " + (System.currentTimeMillis() - startTime) / 1000 + " seconds");
    }
}
