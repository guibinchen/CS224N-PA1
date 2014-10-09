package edu.stanford.nlp.mt.decoder.feat;

import java.util.List;

import edu.stanford.nlp.mt.util.FeatureValue;
import edu.stanford.nlp.mt.util.Featurizable;
import edu.stanford.nlp.mt.util.IString;
import edu.stanford.nlp.mt.decoder.feat.RuleFeaturizer;
import edu.stanford.nlp.util.Generics;

/**
 * A rule featurizer.
 */
public class MyFeaturizer implements RuleFeaturizer<IString, String> {

  private static final String FEATURE_NAME = "TGTD";

  @Override
  public void initialize() {
    // Do any setup here.
  }

  @Override
  public List<FeatureValue<String>> ruleFeaturize(
      Featurizable<IString, String> f) {

    List<FeatureValue<String>> features = Generics.newLinkedList();

    // Count the number of 'of'.
    int cnt = 0;
    for (IString istring : f.targetPhrase) {
      if (istring.toString().equals("of")) {
        cnt++;
      }
    }
    features.add(new FeatureValue<String>(
        String.format("%s:%d",FEATURE_NAME, cnt), 1.0));

    return features;
  }

  @Override
  public boolean isolationScoreOnly() {
    return false;
  }
}
