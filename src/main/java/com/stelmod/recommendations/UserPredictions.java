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

    public static final int USER_ID = 103;
    public static final int NUMBER_OF_RECOMMENDATIONS = 10;
    public static final int NEIGHBORHOOD_SIZE = 20;

    public static void main(String[] args) throws Exception {
        URL inputStream = Thread.currentThread().getContextClassLoader().getResource("ml-small.csv");

        File inputFile = new File(inputStream.toURI());
        DataModel dataModel = new FileDataModel(inputFile);

        UserSimilarity similarity = new PearsonCorrelationSimilarity(dataModel);
        RecommenderBuilder recommenderBuilder = new NN_Pearson_GenericRecommender(NEIGHBORHOOD_SIZE, similarity);
        Recommender recommender = recommenderBuilder.buildRecommender(dataModel);

        List<RecommendedItem> recommendedItems = recommender.recommend(USER_ID, NUMBER_OF_RECOMMENDATIONS);
        recommendedItems.forEach(item -> System.out.println(item));
    }
}
