package com.stelmod.recommendations;

import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.model.DataModel;

import java.io.File;
import java.net.URL;

public class App {
    public static void main(String[] args) throws Exception {
        URL inputStream = Thread.currentThread().getContextClassLoader().getResource("ml-small.csv");

        File inputFile = new File(inputStream.toURI());
        DataModel dataModel = new FileDataModel(inputFile);

        NN_Evaluator nn_evaluator = new NN_Evaluator(dataModel, new int[] {2, 4});
        nn_evaluator.evaluateModel();
    }
}