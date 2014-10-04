package cs224n.wordaligner;

import cs224n.util.Counter;
import cs224n.util.CounterMap;
import cs224n.util.Counters;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

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
    CounterMap<String, String> t = null;
    // Use in the first iteration to save space
    double initProb = getInitialProbability(trainingData);

    // Run EM algorithm
    for (int i = 0; i < T; i++) {
      // Set initial counts to 0 (implicitly)
      CounterMap<String, String> sourceTargetCounts = new CounterMap<>();

      // E-step: update counts based on probabilities
      // for k = 1..n
      // Note: Not using index explicitly as List may not have random access.
      for (SentencePair pair : trainingData) {
        // source is "French"
        List<String> sourceWords = pair.getSourceWords();
        // target is "English"
        List<String> targetWords = pair.getTargetWords();

        // for i = 1..m_k
        for (String source : sourceWords) {
          // Cache $$sum_{j=0}^{l_k} t(f_i^{(k)}|e_j^{(k)})$$
          double sumT = 0.0;
          for (String target : targetWords) {
            sumT += i == 0 ? initProb : t.getCount(target, source);
          }
          sumT += i == 0 ? initProb : t.getCount(NULL, source);

          // for j = 1..l_k
          for (String target : targetWords) {
            // Increment probability count
            double deltaKIJ = (i == 0 ? initProb : t.getCount(target, source)) / sumT;
            sourceTargetCounts.incrementCount(target, source, deltaKIJ);
          }
          double deltaKIJ = (i == 0 ? initProb : t.getCount(NULL, source)) / sumT;
          sourceTargetCounts.incrementCount(NULL, source, deltaKIJ);
        }
      }

      // M-step: update probabilities based on updated counts
      CounterMap<String, String> tPrime = Counters.conditionalNormalize(sourceTargetCounts);

      if (i > 0 && hasConverged(t, tPrime)) break;

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
   * Returns initial uniform probability.
   * @param trainingData
   * @return
   */
  private double getInitialProbability(List<SentencePair> trainingData) {
    // Gather all source words
    Set<String> sourceWordSet = new HashSet<>();

    for (SentencePair pair : trainingData) {
      List<String> sourceWords = pair.getSourceWords();
      sourceWordSet.addAll(sourceWords);
    }

    return 1.0 / sourceWordSet.size();
  }
}
