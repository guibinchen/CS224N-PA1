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
  private static final double EPSILON = 1e-4;
  // TODO: determine a proper value
  private static final int T = 50;
  private CounterMap<String, String> t;

  /**
   * Get T(f|e) which model 2 will use it to do initialization.
   * @return t
   */
  public CounterMap<String, String> getT() {
    return t;
  }

  @Override
  public Alignment align(SentencePair sentencePair) {
    Alignment alignment = new Alignment();

    List<String> sourceWords = sentencePair.getSourceWords();
    List<String> targetWords = sentencePair.getTargetWords();
    int numSourceWords = sourceWords.size();
    int numTargetWords = targetWords.size();

    // Find best alignment for each source word
    // In Model 1, q(j|i,l,m) is a constant, so only need to consider t(e|f).
    for (int tgtIndex = 0; tgtIndex < numTargetWords; tgtIndex++) {
      String target = targetWords.get(tgtIndex);
      // Match with NULL_WORD.
      double bestScore = t.getCount(NULL_WORD, target);
      int bestIndex = numSourceWords;

      // Match with source text.
      for (int srcIndex = 0; srcIndex < numSourceWords; srcIndex++) {
        String source = sourceWords.get(srcIndex);
        double score = t.getCount(source, target);

        if (score > bestScore) {
          bestScore = score;
          bestIndex = srcIndex;
        }
      }

      alignment.addPredictedAlignment(tgtIndex, bestIndex);
    }

    return alignment;
  }

  @Override
  public void train(List<SentencePair> trainingData) {
    CounterMap<String, String> t = null;
    // Use in the first iteration to save space
    double initProb = getInitialProbability(trainingData);

    // Run EM algorithm
    for (int i = 0; i < T; i++) {
      System.out.println("Iteration " + i);

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
        for (String target : targetWords) {
          // Cache $$sum_{j=0}^{l_k} t(e_i^{(k)}|f_j^{(k)})$$
          double sumT = 0.0;
          for (String source : sourceWords) {
            sumT += i == 0 ? initProb : t.getCount(source, target);
          }
          sumT += i == 0 ? initProb : t.getCount(NULL_WORD, target);

          // for j = 1..l_k
          for (String source : sourceWords) {
            // Increment probability count
            double deltaKIJ = (i == 0 ? initProb : t.getCount(source, target)) / sumT;
            sourceTargetCounts.incrementCount(source, target, deltaKIJ);
          }
          double deltaKIJ = (i == 0 ? initProb : t.getCount(NULL_WORD, target)) / sumT;
          sourceTargetCounts.incrementCount(NULL_WORD, target, deltaKIJ);
        }
      }

      // M-step: update probabilities based on updated counts
      CounterMap<String, String> tPrime = Counters.conditionalNormalize(sourceTargetCounts);

      // Check convergence every 5 iterations
      if ((i + 1) % 5 == 0 && hasConverged(t, tPrime)) {
        System.out.println("Converged at iteration " + i);
        break;
      }

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
    // Gather all target words
    Set<String> targetWordSet = new HashSet<>();

    for (SentencePair pair : trainingData) {
      List<String> targetWords = pair.getTargetWords();
      targetWordSet.addAll(targetWords);
    }

    return 1.0 / targetWordSet.size();
  }
}
