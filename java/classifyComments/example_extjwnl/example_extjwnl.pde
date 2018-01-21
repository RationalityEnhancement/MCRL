import net.sf.extjwnl.JWNLException;
import net.sf.extjwnl.data.IndexWord;
import net.sf.extjwnl.data.POS;
import net.sf.extjwnl.data.PointerType;
import net.sf.extjwnl.data.PointerUtils;
import net.sf.extjwnl.data.list.PointerTargetNodeList;
import net.sf.extjwnl.data.list.PointerTargetTree;
import net.sf.extjwnl.data.relationship.AsymmetricRelationship;
import net.sf.extjwnl.data.relationship.Relationship;
import net.sf.extjwnl.data.relationship.RelationshipFinder;
import net.sf.extjwnl.data.relationship.RelationshipList;
import net.sf.extjwnl.dictionary.Dictionary;
import net.sf.extjwnl.data.Synset;
import net.sf.extjwnl.data.list.PointerTargetNode;
import net.sf.extjwnl.data.Word;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import java.util.List;

void setup() {
    try {
       Dictionary dictionary = null;
       //FileInputStream inputStream = new FileInputStream("/Users/luckyfish/Desktop/processing/libraries/jwnl/config/file_properties.xml");
       //dictionary = Dictionary.getInstance(inputStream);
       dictionary = Dictionary.getDefaultResourceInstance();
  
      if (null != dictionary) {
          new Examples(dictionary).go();
      }
    
    } catch (Exception ex) {
      ex.printStackTrace();
      System.exit(-1);
    }    
}

public class Examples {
    private IndexWord wtry;
    private IndexWord wguess;
    private IndexWord wgut;
    private IndexWord wb;
    private IndexWord wr;
    private final static String MORPH_PHRASE = "running-away";
    private final Dictionary dictionary;

    public Examples(Dictionary dictionary) throws JWNLException {
        this.dictionary = dictionary;
    }

    public void go() throws JWNLException, CloneNotSupportedException {
       // demonstrateMorphologicalAnalysis(MORPH_PHRASE);
       // synonyms(dictionary.getIndexWord(POS.VERB, "try")); // returns nothing
       
        //synonyms_tree(dictionary.getIndexWord(POS.VERB, "try"));
        
        println ("Trying:      -----------------------");
        IndexWord w = dictionary.lookupIndexWord(POS.VERB, "tried");
        synonyms_tree(w);
        
        println ("Black:      -----------------------");
        w = dictionary.lookupIndexWord(POS.NOUN, "black");
        synonyms_tree(w);
        
        w = dictionary.lookupIndexWord(POS.ADJECTIVE, "black");
        synonyms_tree(w);
        
        //demonstrateTreeOperation(dictionary.getIndexWord(POS.VERB, "trying"));
        
        //demonstrateAsymmetricRelationshipOperation(dictionary.getIndexWord(POS.NOUN, "guess"), 
                                             //       dictionary.lookupIndexWord(POS.NOUN, "gut"));
       // demonstrateSymmetricRelationshipOperation(FUNNY, DROLL);
    }

    private void synonyms(IndexWord word) throws JWNLException {
        List<Synset> synset=word.getSenses();
        int nums = word.sortSenses();
        
        println("-------- " +  nums + " senses, ", synset.size());
        int i = 0;
        for (  Synset syn : synset) {
          PointerTargetNodeList s = PointerUtils.getSynonyms(syn);
          System.out.println("Synonyms of \"" + word.getLemma() + "\":");
          s.print();
          i++;
          if (i >= nums) break;
        }
    }

    private void synonyms_tree(IndexWord word) throws JWNLException {
      List<Synset> synset=word.getSenses();
      
      int nums = word.sortSenses();
      println("-------- " +  nums + " senses, ", synset.size());
      int i = 0;
        
      for (  Synset syn : synset) {
        PointerTargetTree s = PointerUtils.getSynonymTree(syn, 2 /*depth*/);
        System.out.println("Tree Synonyms of \"" + word.getLemma() + "\":");
        println("sense " + i);
        
        List<PointerTargetNodeList>  l = s.toList();
        int t = 0;
        for (PointerTargetNodeList nl : l) {
          println("tree element", t); t++;
          int wi = 0;
          for (PointerTargetNode n : nl) {
            Synset ns = n.getSynset();
            if (ns!=null) {
              List<Word> ws = ns.getWords();
              for (Word ww : ws) {
                // ww.getUseCount() is the frequency of occurance as reported by weordent engine
                println(wi , ":", ww.getLemma(), "use:", ww.getUseCount(), " ptrs:",  ww.getPointers().size()); wi++;
              }
            }
            //println(wi , ":", n.toString() ); wi++;
          }
        }
        
        //s.print();
        i++;
        if (i >= nums) break;
      }
    }

    private void demonstrateAsymmetricRelationshipOperation(IndexWord start, IndexWord end) throws JWNLException, CloneNotSupportedException {
        // Try to find a relationship between the first sense of <var>start</var> and the first sense of <var>end</var>
        RelationshipList list = RelationshipFinder.findRelationships(start.getSenses().get(0), end.getSenses().get(0), PointerType.HYPERNYM);
        System.out.println("Hypernym relationship between \"" + start.getLemma() + "\" and \"" + end.getLemma() + "\":");
        for (Object aList : list) {
            ((Relationship) aList).getNodeList().print();
        }
        System.out.println("Common Parent Index: " + ((AsymmetricRelationship) list.get(0)).getCommonParentIndex());
        System.out.println("Depth: " + list.get(0).getDepth());
    }

    private void demonstrateSymmetricRelationshipOperation(IndexWord start, IndexWord end) throws JWNLException, CloneNotSupportedException {
        // find all synonyms that <var>start</var> and <var>end</var> have in common
        RelationshipList list = RelationshipFinder.findRelationships(start.getSenses().get(0), end.getSenses().get(0), PointerType.SIMILAR_TO);
        System.out.println("Synonym relationship between \"" + start.getLemma() + "\" and \"" + end.getLemma() + "\":");
        for (Object aList : list) {
            ((Relationship) aList).getNodeList().print();
        }
        System.out.println("Depth: " + list.get(0).getDepth());
}
}