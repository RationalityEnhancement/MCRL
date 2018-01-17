//import opennlp.tools.chunker.*;
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

void setup() {
  InputStream modelIn = null;
  try {
    // this simply splits the data to words, as in splitTokens()
    
    // the bin files are downloaded from
    // http://opennlp.sourceforge.net/models-1.5/
    // and encode various pre-trined textprocessing models
    
    //modelIn = new FileInputStream("Desktop/processing/openNLPDemo/en-token.bin");
    //TokenizerModel model = new TokenizerModel(modelIn);
    
    POSTag();
  }
  catch (IOException e) {
    e.printStackTrace();
  }
  catch (TerminateToolException e) {
    e.printStackTrace();
  }
  finally {
    if (modelIn != null) {
      try {
        modelIn.close();
      }
      catch (IOException e) {
      }
    }
  }
  noLoop();
}

void draw() {}

public static void POSTag() throws IOException {
  POSModel model = new POSModelLoader().load(new File("Desktop/processing/openNLPDemo/en-pos-maxent.bin"));
  PerformanceMonitor perfMon = new PerformanceMonitor(System.err, "sent");
  POSTaggerME tagger = new POSTaggerME(model);
 
  String input = "Tried to reveal the most squares per move.";
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
    POSSample sample = new POSSample(whitespaceTokenizerLine, tags); //<>//
    System.out.println(sample.toString());
 
    perfMon.incrementCounter();
  }
  perfMon.stopAndPrintFinalResult();
}