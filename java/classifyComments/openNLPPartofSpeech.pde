import opennlp.tools.cmdline.*;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.dictionary.*;
import opennlp.tools.doccat.*;
import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSSample;
import opennlp.tools.postag.POSTaggerME;
import opennlp.tools.cmdline.postag.POSModelLoader;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.tokenize.WhitespaceTokenizer;
import java.io.StringReader;

import java.io.FileInputStream;
import java.io.FileNotFoundException;

POSModel model = null;

/*
POS tags
        CC Coordinating conjunction
        CD Cardinal number
        DT Determiner
        EX Existential there
        FW Foreign word
        IN Preposition or subordinating conjunction
        JJ Adjective
        JJR Adjective, comparative
        JJS Adjective, superlative
        LS List item marker
        MD Modal
        NN Noun, singular or mass
        NNS Noun, plural
        NNP Proper noun, singular
        NNPS Proper noun, plural
        PDT Predeterminer
        POS Possessive ending
        PRP Personal pronoun
        PRP$ Possessive pronoun
        RB Adverb
        RBR Adverb, comparative
        RBS Adverb, superlative
        RP Particle
        SYM Symbol
        TO to
        UH Interjection
        VB Verb, base form
        VBD Verb, past tense
        VBG Verb, gerund or present participle
        VBN Verb, past participle
        VBP Verb, non­3rd person singular present
        VBZ Verb, 3rd person singular present
        WDT Wh­determiner
        WP Wh­pronoun
        WP$ Possessive wh­pronoun
        WRB Wh­adverb
*/

ArrayList<WRD> tokeniseSentence(String input) {
  ArrayList<WRD> words = new ArrayList<WRD>();
  
  try {
      
    PerformanceMonitor perfMon = new PerformanceMonitor(System.err, "sent");
    POSTaggerME tagger = new POSTaggerME(model);
   
    //String input = "Tried to reveal the most squares per move.";
    ObjectStream<String> lineStream = new PlainTextByLineStream(new StringReader(input));
   
    perfMon.start();
    String line;
    while ((line = lineStream.read()) != null) {
   
      String whitespaceTokenizerLine[] = WhitespaceTokenizer.INSTANCE.tokenize(line);
      String[] tags = tagger.tag(whitespaceTokenizerLine);
      
      // these are the tags that we care about, will discard everything else
      /*
          NN Noun, singular or mass
          NNS Noun, plural
          NNP Proper noun, singular
          NNPS Proper noun, plural
      */
      
      /*
          VB Verb, base form
          VBD Verb, past tense
          VBG Verb, gerund or present participle
          VBN Verb, past participle
          VBP Verb, non­3rd person singular present
          VBZ Verb, 3rd person singular present
      */
      
      /*
          RB Adverb
          RBR Adverb, comparative
          RBS Adverb, superlative
          WRB Wh­adverb
      */
      
      /*
          JJ Adjective
          JJR Adjective, comparative
          JJS Adjective, superlative
      */
      // Tried to reveal the most squares per move
      // NNP TO VB DT RBS NNS IN NN
      
      // becomes: Tried reveal most squares move
      //POSSample sample = new POSSample(whitespaceTokenizerLine, tags);
      //System.out.println(sample.toString());
      
      // keep the wanted tags and leave out the unwanted tags
      int i = 0;
      String rg = "[,.!?]";
     
      for (String tag : tags) {
         whitespaceTokenizerLine[i] =  whitespaceTokenizerLine[i].trim().replaceAll(rg, "").toLowerCase();
        if (tag.equals("NN") || tag.equals("NNP")) {
          words.add(new WRD(whitespaceTokenizerLine[i], 0, POS.NOUN));
        } else if (tag.equals("NNS") || tag.equals("NNPS")) {
          words.add(new WRD(rita.singularize(whitespaceTokenizerLine[i]), 0, POS.NOUN));
        } else if (tag.equals("VB") ) {
          words.add(new WRD(whitespaceTokenizerLine[i], 0, POS.VERB));
        } else if (tag.indexOf("VB") != -1 ) {
          IndexWord idw = dictionary.lookupIndexWord(POS.VERB, whitespaceTokenizerLine[i]);
          if (idw == null) {
            println("ERROR JWNL not found: ", whitespaceTokenizerLine[i], tag);
          } else {
            if (whitespaceTokenizerLine[i].equals("felt")) {
              words.add(new WRD("feel", 0, POS.VERB));
            } else {
              println("Verb correction: ",  whitespaceTokenizerLine[i], "to", idw.getLemma());
              words.add(new WRD(idw.getLemma(), 0, POS.VERB));
            }
          }
        }
        else if (tag.indexOf("RB") != -1) {
          words.add(new WRD(whitespaceTokenizerLine[i], 0, POS.ADVERB));
        } else if (tag.indexOf("JJ") != -1) {
          words.add(new WRD(whitespaceTokenizerLine[i], 0, POS.ADVERB));
        }
        i++;
      }
   
      perfMon.incrementCounter();
    }
    perfMon.stopAndPrintFinalResult();
    }
    catch (IOException e) {
      e.printStackTrace();
    }
    catch (JWNLException e) {
      e.printStackTrace();
    }
    catch (TerminateToolException e) {
      e.printStackTrace();
    }
    finally {
      
    }
  
    return words;
}