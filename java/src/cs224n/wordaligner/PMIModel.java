package cs224n.wordaligner;

import cs224n.util.*;
import org.omg.CORBA.CODESET_INCOMPATIBLE;

import java.util.List;

/**
 * Simple word alignment PMI model.
 */
public class PMIModel implements WordAligner {

  private static final long serialVersionUID = 1315751943476440515L;

  // From the training data.
  private Counter<String> sourceCounts;
  private Counter<String> targetCounts;
  private CounterMap<String,String> sourceTargetCounts;

  public Alignment align(SentencePair sentencePair) {
    // Predict alignments based on the counts you collected with train().
    Alignment alignment = new Alignment();
    List<String> sourceWords = sentencePair.getSourceWords();
    List<String> targetWords = sentencePair.getTargetWords();
    int numSourceWords = sourceWords.size();
    int numTargetWords = targetWords.size();

    for (int tgtIndex = 0; tgtIndex < numTargetWords; tgtIndex++) {
      String tgt = targetWords.get(tgtIndex);
      // Match with NULL.
      double bestScore = calculateScore(NULL_WORD, tgt);
      int bestIndex = numSourceWords;
      // Match with source text.
      for (int srcIndex = 0; srcIndex < numSourceWords; srcIndex++) {
        String src = sourceWords.get(srcIndex);
        double score = calculateScore(src, tgt);
        if (score > bestScore) {
          bestScore = score;
          bestIndex = srcIndex;
        }
      }
      alignment.addPredictedAlignment(tgtIndex, bestIndex);
    }
    return alignment;
  }

  public void train(List<SentencePair> trainingPairs) {
    sourceTargetCounts = new CounterMap<String,String>();
    sourceCounts = new Counter<String>();
    targetCounts = new Counter<String>();
    for(SentencePair pair : trainingPairs){
      List<String> targetWords = pair.getTargetWords();
      List<String> sourceWords = pair.getSourceWords();

      // Calculate P(f_j, e_i).
      for(String target : targetWords){
        for(String source : sourceWords){
          sourceTargetCounts.incrementCount(source, target, 1.0);
        }
        sourceTargetCounts.incrementCount(NULL_WORD, target, 1.0);
      }

      // Count the occurrences of e_i.
      for (String source : sourceWords) {
        sourceCounts.incrementCount(source, 1.0);
      }
      sourceCounts.incrementCount(NULL_WORD, 1.0);

      // Count the occurrences of f_j.
      for (String target : targetWords) {
        targetCounts.incrementCount(target, 1.0);
      }
    }
  }

  // Calculate score based on the formula: p(f_j, e_i) / P(f_j) / P(e_i).
  private double calculateScore(String src, String tgt) {
    return sourceTargetCounts.getCount(src, tgt) /
        sourceCounts.getCount(src) / targetCounts.getCount(tgt);
  }
}
