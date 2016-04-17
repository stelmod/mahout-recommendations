package com.stelmod.recommendations;

import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.RMSRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

import java.io.File;
import java.net.URL;

public class App {
    public static void main(String[] args) throws Exception {
        URL inputStream = Thread.currentThread().getContextClassLoader().getResource("ml-small.csv");

        File inputFile = new File(inputStream.toURI());
        DataModel dataModel = new FileDataModel(inputFile);

        RecommenderEvaluator evaluator = new RMSRecommenderEvaluator();
        RecommenderBuilder recommenderBuilder = dataModel1 -> {
            UserSimilarity similarity = new PearsonCorrelationSimilarity(dataModel1);
            UserNeighborhood neighborhood = new NearestNUserNeighborhood(2, similarity, dataModel1);
            return new GenericUserBasedRecommender(dataModel1, neighborhood, similarity);
        };

        double score = evaluator.evaluate(recommenderBuilder, null, dataModel, 0.7, 1.0);
        System.out.println(score);
    }
}
