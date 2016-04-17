package com.stelmod.recommendations;


import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

public class NN_Pearson_GenericRecommender implements RecommenderBuilder {
    private final int neighbourhoodSize;
    private final UserSimilarity similarity;

    public NN_Pearson_GenericRecommender(int neighbourhoodSize, UserSimilarity similarity) {
        this.neighbourhoodSize = neighbourhoodSize;
        this.similarity = similarity;
    }

    @Override
    public Recommender buildRecommender(DataModel dataModel) throws TasteException {
        UserNeighborhood neighborhood = new NearestNUserNeighborhood(neighbourhoodSize, similarity, dataModel);
        return new GenericUserBasedRecommender(dataModel, neighborhood, similarity);
    }

}
