package com.stelmod.recommendations;

import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.impl.eval.RMSRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.recommender.svd.SVDPlusPlusFactorizer;
import org.apache.mahout.cf.taste.impl.recommender.svd.SVDRecommender;
import org.apache.mahout.cf.taste.model.DataModel;

public class SVD_evaluator {
    private final DataModel dataModel;
    private final int[] featuresToEvaluate;

    public SVD_evaluator(DataModel dataModel, int[] features) {
        this.dataModel = dataModel;
        this.featuresToEvaluate = features;
    }

    public void evaluateModel() throws Exception {
        System.out.println("Matrix factorization recommendations");
        for (int features: featuresToEvaluate) {
            RMSRecommenderEvaluator evaluator = new RMSRecommenderEvaluator();
            RecommenderBuilder recommenderBuilder = dm -> new SVDRecommender(dm, new SVDPlusPlusFactorizer(dm, features, 20));
            double score = evaluator.evaluate(recommenderBuilder, null, dataModel, 0.8, 1.0);
            System.out.println("Features " + features + " RMSE: " + score);
        }

    }
}
