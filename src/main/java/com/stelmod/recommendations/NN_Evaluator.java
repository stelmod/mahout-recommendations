package com.stelmod.recommendations;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.RMSRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

public class NN_Evaluator {
    private final DataModel dataModel;
    private final int[] neighbourhoodSizes;

    public NN_Evaluator(DataModel dataModel, int[] neighbourhoodSizes) {
        this.dataModel = dataModel;
        this.neighbourhoodSizes = neighbourhoodSizes;
    }

    public void evaluateModel() throws Exception {
        UserSimilarity similarity = new PearsonCorrelationSimilarity(dataModel);

        for (int neighbourhoodSize : neighbourhoodSizes) {
            trainAndEvaluareError(dataModel, new RMSRecommenderEvaluator(), neighbourhoodSize, similarity);
        }
    }

    private double trainAndEvaluareError(DataModel dataModel, RecommenderEvaluator evaluator, int neighbourhoodSize, UserSimilarity similarity) throws TasteException {
        RecommenderBuilder recommenderBuilder = new NN_Pearson_GenericRecommender(neighbourhoodSize, similarity);
        double score = evaluator.evaluate(recommenderBuilder, null, dataModel, 0.8, 1.0);
        System.out.println("Neighbourhood size: " + neighbourhoodSize + ", RMSE: " + score);
        return score;
    }
}
