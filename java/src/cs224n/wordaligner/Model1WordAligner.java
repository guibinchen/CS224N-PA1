package cs224n.wordaligner;

import cs224n.util.Counter;
import cs224n.util.CounterMap;
import cs224n.util.Counters;

import java.util.List;

/**
 * Word aligner using IBM Model 1.
 */
public class Model1WordAligner implements WordAligner {
  // NULL in source language.
  // TODO: extract to abstract class
  private static final String NULL = "";
  private static final double EPSILON = 1e-4;
  // TODO: determine a proper value
  private static final int T = 10;
  private CounterMap<String, String> t;

  @Override
  public Alignment align(SentencePair sentencePair) {
    return null;
  }

  @Override
  public void train(List<SentencePair> trainingData) {
    CounterMap<String, String> t = getInitialProbabilities(trainingData);

    // Run EM algorithm
    for (int i = 0; i < T; i++) {
      // Set initial counts to 0 (implicitly)
      CounterMap<String, String> sourceTargetCounts = new CounterMap<>();

      // E-step: update counts based on probabilities
      for (SentencePair pair : trainingData) {
        List<String> targetWords = pair.getTargetWords();
        List<String> sourceWords = pair.getSourceWords();

        for (String target : targetWords) {
          // Cache $$sum_{j=1}^{l_k} t(f_i^{(k)}|e_j^{(k)})$$
          double sumT = 0.0;
          for (String source : sourceWords) {
            sumT += t.getCount(source, target);
          }
          sumT += t.getCount(NULL, target);

          for (String source : sourceWords) {
            // Increment probability count
            double deltaKIJ = t.getCount(source, target) / sumT;
            sourceTargetCounts.incrementCount(source, target, deltaKIJ);
          }
          double deltaKIJ = t.getCount(NULL, target) / sumT;
          sourceTargetCounts.incrementCount(NULL, target, deltaKIJ);
        }
      }

      // M-step: update probabilities based on updated counts
      CounterMap<String, String> tPrime = Counters.conditionalNormalize(sourceTargetCounts);

      if (hasConverged(t, tPrime)) break;

      t = tPrime;
    }

    this.t = t;
  }

  /**
   * Check if the probabilities have converged.
   * @param t - original probabilities
   * @param tPrime - updated probabilities
   * @return true if converged
   */
  private static boolean hasConverged(CounterMap<String, String> t, CounterMap<String, String> tPrime) {
    for (String source : t.keySet()) {
      Counter<String> probs = t.getCounter(source);
      Counter<String> probsPrime = tPrime.getCounter(source);

      for (String target : probs.keySet()) {
        double prob = probs.getCount(target);
        double probPrime = probsPrime.getCount(target);

        if (!(prob == probPrime || Math.abs(prob - probPrime) <= EPSILON)) {
          return false;
        }
      }
    }

    // Finally!
    return true;
  }

  /**
   * Returns initial uniform probabilities.
   * @param trainingData
   * @return
   */
  private CounterMap<String, String> getInitialProbabilities(List<SentencePair> trainingData) {
    // Initialize counts
    CounterMap<String, String> sourceTargetCounts = new CounterMap<>();
    for (SentencePair pair : trainingData) {
      List<String> targetWords = pair.getTargetWords();
      List<String> sourceWords = pair.getSourceWords();

      for (String target : targetWords) {
        for (String source : sourceWords) {
          // Set count to 1 to get uniform probability after normalization
          sourceTargetCounts.setCount(source, target, 1.0);
        }
        sourceTargetCounts.setCount(NULL, target, 1.0);
      }
    }

    // Normalize to get initial uniform probability for t(f|e)
    return Counters.conditionalNormalize(sourceTargetCounts);
  }
}
