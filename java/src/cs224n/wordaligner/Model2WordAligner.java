package cs224n.wordaligner;

import cs224n.util.Counter;
import cs224n.util.CounterMap;
import cs224n.util.Counters;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Word aligner using IBM Model 2.
 */
public class Model2WordAligner implements WordAligner {

  // Maximum number of iterations.
  private static final int T = 100;
  private static final double EPSILON = 1e-4;

  // <Source, Target> => Count.
  private CounterMap<String, String> t;
  // <<targetIndex, sourceLength, targetLength>, sourceIndex> => Count.
  private CounterMap<String, Integer> q;


  @Override
  public Alignment align(SentencePair sentencePair) {
    Alignment alignment = new Alignment();

    List<String> sourceWords = sentencePair.getSourceWords();
    List<String> targetWords = sentencePair.getTargetWords();
    int numSourceWords = sourceWords.size();
    int numTargetWords = targetWords.size();

    // Find best alignment for each source word
    for (int tgtIndex = 0; tgtIndex < numTargetWords; tgtIndex++) {
      String target = targetWords.get(tgtIndex);
      String index = tgtIndex + "," + numSourceWords + "," + numTargetWords;

      // Match with NULL_WORD.
      double bestScore = q.getCount(index, numSourceWords) * t.getCount(NULL_WORD, target);
      int bestIndex = numSourceWords;

      // Match with source text.
      for (int srcIndex = 0; srcIndex < numSourceWords; srcIndex++) {
        String source = sourceWords.get(srcIndex);
        double score = q.getCount(index, srcIndex) * t.getCount(source, target);

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
    // Initialize t and q.
    initialize(trainingData);

    // Maximum T iterations.
    for (int iter = 0; iter < T; iter++) {
      CounterMap<String, String> sourceTargetCounts = new CounterMap<>();
      CounterMap<String, Integer> alignmentCounts = new CounterMap<>();

      // For each sentence in the training data.
      for (SentencePair sentencePair : trainingData) {
        List<String> sourceWords = sentencePair.sourceWords;
        List<String> targetWords = sentencePair.targetWords;
        int numSourceWords = sourceWords.size();
        int numTargetWords = targetWords.size();
        String suffix = "," + numSourceWords + "," + numTargetWords;

        for (int i = 0; i < numTargetWords; i++) {
          double sumQT = 0;
          String target = targetWords.get(i);

          for (int j = 0; j <= numSourceWords; j++) {
            String source = j == numSourceWords ? NULL_WORD : sourceWords.get(j);
            String index = i + suffix;
            sumQT += q.getCount(index, j) * t.getCount(source, target);
          }

          for (int j = 0; j <= numSourceWords; j++) {
            String source = j == numSourceWords ? NULL_WORD : sourceWords.get(j);
            String index = i + suffix;

            double deltaKIJ = alignmentCounts.getCount(index, j) *
                sourceTargetCounts.getCount(source, target) / sumQT;

            sourceTargetCounts.incrementCount(target, source, deltaKIJ);
            alignmentCounts.incrementCount(index, j, deltaKIJ);
          }
        }

        CounterMap<String, String> tPrime = Counters.conditionalNormalize(sourceTargetCounts);
        CounterMap<String, Integer> qPrime = Counters.conditionalNormalize(alignmentCounts);

        if ((iter + 1) % 5 == 0 && hasConverged(t, tPrime)) {
          System.out.println("Converged at iteration: " + iter);
          break;
        }

        t = tPrime;
        q = qPrime;
      }
    }
  }

  // Judge if EM has converged.
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


  // Initialize model 2 parameter using model 1.
  private void initialize(List<SentencePair> trainingData) {
    // Use model 1 to train t.
    Model1WordAligner model1 = new Model1WordAligner();
    model1.train(trainingData);
    t = model1.getT();
    q = new CounterMap<>();

    // Randomly initialize q.
    for (SentencePair pair : trainingData) {
      int m = pair.getSourceWords().size();
      int n = pair.getTargetWords().size();
      for (int i = 0; i < n; i++) { // Target.
        for (int j = 0; j <= m; j++) { // Source.
          q.setCount(i + "," + n + "," + m, j, i + Math.random());
        }
      }
    }

    // Normalize q.
    Counters.conditionalNormalize(q);
  }
}
