package com.stelmod.recommendations;

import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

import java.io.File;
import java.net.URL;
import java.util.List;

public class App {
    public static void main(String[] args) throws Exception {
        URL inputStream = Thread.currentThread().getContextClassLoader().getResource("sample.csv");

        File inputFile = new File(inputStream.toURI());
        DataModel dataModel = new FileDataModel(inputFile);
        UserSimilarity similarity = new PearsonCorrelationSimilarity(dataModel);
        UserNeighborhood neighborhood = new NearestNUserNeighborhood(2, similarity, dataModel);
        Recommender recommender = new GenericUserBasedRecommender(dataModel, neighborhood, similarity);
        List<RecommendedItem> recommendedItems = recommender.recommend(1, 1);
        for (RecommendedItem recommendedItem : recommendedItems) {
            System.out.println(recommendedItem);
        }
    }
}
