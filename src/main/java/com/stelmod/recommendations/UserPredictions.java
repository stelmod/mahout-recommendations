package com.stelmod.recommendations;

import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

import java.io.File;
import java.net.URL;
import java.util.List;

public class UserPredictions {

    public static void main(String[] args) throws Exception {
        URL inputStream = Thread.currentThread().getContextClassLoader().getResource("ml-small.csv");

        File inputFile = new File(inputStream.toURI());
        DataModel dataModel = new FileDataModel(inputFile);

        UserSimilarity similarity = new PearsonCorrelationSimilarity(dataModel);
        RecommenderBuilder recommenderBuilder = new NN_Pearson_GenericRecommender(20, similarity);
        Recommender recommender = recommenderBuilder.buildRecommender(dataModel);
        List<RecommendedItem> recommendedItems = recommender.recommend(285, 10);
        recommendedItems.forEach(item -> System.out.println(item));
    }
}
