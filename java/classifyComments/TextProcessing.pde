// extract bag of words and their frequencies from the comment classification file 
// will replace plural with singular


void extractWords() {
  synsDic = new StringDict();
  
  String lines[] = loadStrings(commentsFileName);
  println(lines.length, "comments");
  
  PrintWriter wp = createWriter(commentsPreprocessed);
  StringList words = new StringList();
  
  for (int l = 0; l < lines.length; l++) {
    
     String[] line = splitTokens(lines[l], "\t");     
     String preprocessedLine = "";
     if (line.length < indexOfCommentColumn) {
       println("Incomplete line: ", line.length, line);
       continue;
     }
     String sentence = line[indexOfCommentColumn].trim();
     sentence = sentence.replaceAll("&quot", "");
     
     String rg = "[(){};<>%'`]";
     sentence =  sentence.trim().replaceAll(rg, "").toLowerCase();
     
     // extract the words and their part of speech tags
     ArrayList<WRD> ww = tokeniseSentence(sentence);
     
     for (int w = 0; w < ww.size(); w++) { 
       
       // the word
       String s =  ww.get(w).w;
       
       // noun, verb, adjective or adverb
       net.sf.extjwnl.data.POS pos = ww.get(w).pos;

       if ( s.length() > 2 && !IgnoreList(s) ) {

           if (!noReplaceList(s) && replaceByCommonSynonyms) {
             // stemming; the first element is the most common form of this word, which we will retain
             String[] replacementStr = getSynonymStr(s, pos);
             if (replacementStr != null) {
               if (replacementStr.length > 0) {
                 String repString = "";
                 int countrep = 0;
                 for (String r : replacementStr){
                   if (repString.length() > 0) repString +=",";
                   repString += r; countrep++;
                   if (countrep > 3) break;
                 }
                 
                 if (countrep > 1 ) synsDic.set(replacementStr[0], repString);
                 if (!s.equals(replacementStr[0])) println(s, "=>" , replacementStr[0] , " String: ", repString);
                 s = replacementStr[0];
               }
             }
           }
         words.append(s);
         if (preprocessedLine.length() > 0) preprocessedLine+= " ";
         preprocessedLine += s;
       }
      }
      
      wp.println(line[indexOfSubjectColumn] + "\t" + line[indexOfLabelColumn] + "\t" + preprocessedLine);
      wp.flush();
      println(line[indexOfSubjectColumn] + "\t" + line[indexOfLabelColumn] + "\t" + preprocessedLine);
  }
  
  wp.close();
  println(words.size(), "words");
  
  // now extract the unique words and count how often they occur
  IntDict unique = new IntDict();
  
  for (int i = 0; i< words.size(); i++) {
    if (unique.hasKey(words.get(i))) {
      unique.set(words.get(i), unique.get(words.get(i))+1);
    } else {
      unique.set(words.get(i),1);
    }
  }
  
  println(unique.size(), "unique words");
  unique.sortValuesReverse();
  String[] keys = unique.keyArray();
  PrintWriter wr = createWriter(wordfreqfilename);
  IntDict word_counts_yes_label = new IntDict();
  IntDict word_counts_no_label = new IntDict();

  for (int i = 0; i<unique.size(); i++) {
    wr.println(keys[i] + "\t" + unique.get(keys[i]));
    word_counts_yes_label.set(keys[i],0);
    word_counts_no_label.set(keys[i],0);
  }
  wr.flush();
  wr.close();
  
  // count how many times each word occurs with a yes and a no label
  PrintWriter wr_labels = createWriter("wordFreqWithLables.txt");
  
  String[] comments = loadStrings(commentsPreprocessed);
  println("initialsied label count", word_counts_yes_label.size(), word_counts_no_label.size());
  
  for (int j = 0; j< comments.length; j++) {
    String line = comments[j];
    String[] words1 = splitTokens(line, "\t");
    if (words1.length <= commentsPreprocessedCommentIndex) continue;
    int label = parseInt(words1[commentsPreprocessedCommentIndex]);
    
    
    words1 = splitTokens(words1[commentsPreprocessedCommentIndex], " ");
    
    for (int i = 0; i< words1.length; i++) {
      if (label == YesLabel) {
        if (word_counts_yes_label.hasKey(words1[i])) {
          word_counts_yes_label.set(words1[i], word_counts_yes_label.get(words1[i])+1);
        } else {
          println("word not found,y: ", words1[i] );
        }
      } else {
        if (word_counts_no_label.hasKey(words1[i])) {
          word_counts_no_label.set(words1[i], word_counts_no_label.get(words1[i])+1);
        } else {
          println("word not found,n: ", words1[i] );
        }
      }
    }
  }
  
  for (int i = 0; i<unique.size(); i++) {
    wr_labels.println(keys[i] + "\t" + word_counts_yes_label.get(keys[i]) + "\t" + word_counts_no_label.get(keys[i]));
  }
  wr_labels.flush();
  wr_labels.close();
  
  println("Equivalence list:", synsDic.size());
  synsDic.sortKeys();
  for (int i = 0; i< synsDic.size(); i++) {
    println(synsDic.keyArray()[i], ":", synsDic.valueArray()[i] );
  }
}

String[] getSynonymStr(String strW, POS pos) {
  try {
    
    if (dictionary == null) {
      println("ERR: dictionary not initialised");
      return null;
    }
    
   IndexWord idw = dictionary.lookupIndexWord(pos, strW);
   ArrayList<WRD> d1 = synonyms_tree(idw);
   
   // now select only the unique ones and discard low frequency
   IntDict ret = new IntDict();
   replaceOrAdd(ret, d1);
   ret.sortValuesReverse();
   return ret.keyArray();
   
  } catch (Exception ex) {
      println("Exception in extJWNL");
      ex.printStackTrace();
  }  
   
  return null;
}


String[] getSynonymStr(String strW) {
  try {
    
    if (dictionary == null) {
      println("ERR: dictionary not initialised");
      return null;
    }
    
  // println("synonyms for ", strW, " as verb" ); 
   IndexWord idw = dictionary.lookupIndexWord(POS.VERB, strW);
   ArrayList<WRD> d1 = synonyms_tree(idw);
 //  println("synonyms for ", strW, " as noun" ); 
   idw = dictionary.lookupIndexWord(POS.NOUN, strW);
   ArrayList<WRD> d2 = synonyms_tree(idw);
 //  println("synonyms for ", strW, " as adjective" ); 
   idw = dictionary.lookupIndexWord(POS.ADJECTIVE, strW);
   ArrayList<WRD> d3 = synonyms_tree(idw);
 //  println("synonyms for ", strW, " as adverb" ); 
   idw = dictionary.lookupIndexWord(POS.ADVERB, strW);
   ArrayList<WRD> d4 = synonyms_tree(idw);
   
   // now select only the unique ones and discard low frequency
   IntDict ret = new IntDict();
   replaceOrAdd(ret, d1);
   replaceOrAdd(ret, d2);
   replaceOrAdd(ret, d3);
   replaceOrAdd(ret, d4);
   ret.sortValuesReverse();
  // println();
   return ret.keyArray();
   
  } catch (Exception ex) {
      println("Exception in extJWNL");
      ex.printStackTrace();
  }  
   
  return null;
}

ArrayList<WRD> synonyms_tree(IndexWord word) throws JWNLException {
  if (word == null) {
    //println("synonyms_tree bad input");
    return null;
  }
  
  try {
      ArrayList<WRD> allWords = new ArrayList<WRD>();
      
      List<Synset> synset=word.getSenses();
      int nums = word.sortSenses();
      int i = 0;
        
      // for each sense of the word
      for (  Synset syn : synset) {
        
        // get the ynonyms of the sense
        PointerTargetTree s = PointerUtils.getSynonymTree(syn, 2 /*depth*/);        
        List<PointerTargetNodeList>  l = s.toList();
        
        for (PointerTargetNodeList nl : l) {
          for (PointerTargetNode n : nl) {
            Synset ns = n.getSynset();
            if (ns!=null) {
              List<Word> ws = ns.getWords();
              for (Word ww : ws) {
                // ww.getUseCount() is the frequency of occurance as reported by weordent engine
                if (ww.getLemma().indexOf(" ") == -1 &&  ww.getUseCount() > MINIMAL_FREQUENCY_SYNONYM && ww.getLemma().length() > 2) {
                  if (containsNoOrLessFrequent(allWords, ww.getLemma(), ww.getUseCount())) {
                    allWords.add(new WRD(ww.getLemma(), ww.getUseCount(), ww.getPOS()));
                  //  println(ww.getLemma(), ww.getUseCount());
                  }
                }
              }
            }
          }
        }
        i++;
        if (i >= nums) break;
      }
      
      return allWords;
  } catch (Exception ex) {
      println("Exception in extJWNL, synonyms_tree");
      ex.printStackTrace();
  }
  
  return null;
}

void replaceOrAdd(IntDict retWords, ArrayList<WRD> moreWords) {
  if (moreWords == null) return;
  
  for (WRD w : moreWords) {
    if (retWords.hasKey(w.w)) {
      if ( retWords.get(w.w) < w.freq) {
          retWords.set(w.w, w.freq);
      }
    } else {
      if (w.freq > 0) retWords.add(w.w, w.freq);
     // print ("[" +w.w, w.freq + "], ");
    }
  }
  return;
}

boolean containsNoOrLessFrequent(ArrayList<WRD> allWords, String wrd, int freq) {
  for (WRD w : allWords) {
    if (w.w.equals(wrd)) {
      if ( w.freq > freq) return false;
      return true;
    }
  }
  return true;
}

boolean noReplaceList(String w) {
  for (int i = 0; i < doNotReplaceList.length; i++) {
    if (doNotReplaceList[i].equals(w.trim())) return true;
  }
  return false;
  
}

boolean IgnoreList(String w) {
  for (int i = 0; i < ignoreList.length; i++) {
    if (ignoreList[i].equals(w.trim())) return true;
  }
  return false;
}

// --------------------------------------------------------------------


class WRD {
  String w;
  int freq;
  net.sf.extjwnl.data.POS pos;
  
  WRD() {
    w = ""; freq = 0; pos = POS.ADVERB;
  }
  
  WRD(String ws, int f, POS p) {
    w = ws; freq = f; pos = p;
  }
}

// -------------------------------------------------------------------

// text preprocessing, merging words if they have common synonyms
// this does not work so well, and in practice the common synonym is usually taken in a wrong context
void commonSynonyms() {
  
  IntList freq = new IntList();   // how many times has this word occured  
  StringList wordtags = new StringList(); // the words
  String[] s = loadStrings(wordfreqfilename);   // read the list of words and their frequencies as they appear before pre-processing
  
  for (int i = 0; i < s.length; i++) {
    String[] tok = splitTokens(s[i], "\t");
    wordtags.append(tok[0]);
    freq.append(Integer.parseInt(tok[1]));
  }
 
  for (int i = 0; i < s.length; i++) {
    
    if (freq.get(i) == 0) { 
              String[] tok = splitTokens(s[i], "\t");
              println("Error in " + wordfreqfilename + " line " + i + ": " + s[i], tok[0], tok[1]); 
              continue; 
    }
    
    println("Analysing synonyms ", wordtags.get(i));
    String[] tok = splitTokens(wordtags.get(i), ",");
    int mergeWith = -1;
    
    for(int t = 0; t < tok.length; t++) {
      
      // can this word absorb any of the other words in the file?
      String w = tok[t].trim();
      
      // the synonyms of this word
      if (!synsDic.hasKey(w)) continue;
      String[] syns = splitTokens(synsDic.get(w), ",");
        
      // now go through the original list of words and frequencies again
      for (int j = i; j < s.length; j++) {
        
        // ignore self
        if (i==j) continue;
        
        if (freq.get(j) > 0) {
          
          String[] jtok = splitTokens(wordtags.get(j), ",");
          for(int k = 0; k < jtok.length; k++) {
            for (int l = 0; l < syns.length; l++) {
              if (syns[l].equals(jtok[k].trim())) {
                // merge entry j into entry i
                println("\tmatched ", w+"->"+syns[l], " with ", jtok[k]);
                mergeWith = j; break;
              }
            }
            if (mergeWith !=-1) break;
          }
        }
        
        if (mergeWith !=-1) break;
      }
      
      if (mergeWith !=-1) break;
    }
    
    if (mergeWith !=-1) {
      int f = freq.get(mergeWith);
      freq.set(mergeWith, 0);
      
       wordtags.set(i, wordtags.get(i) + ", " + wordtags.get(mergeWith));
       freq.set(i, freq.get(i)+f);
    }
  }
  
  PrintWriter wr = createWriter(wordEquivalencefilename);
  for (int j = 0; j < s.length; j++) {
        if (freq.get(j) > 0) {
          wr.println(wordtags.get(j) + "\t" + freq.get(j));
        }
  }
  wr.flush();
  wr.close();
}

// get an equivalent synonym from the dictionary
String getEquivalent(String w /*the word for which a synonym is sought*/, StringList dict /*the dictionary*/){
  for (int j = 0; j < dict.size(); j++) {
        String[] tok = splitTokens(dict.get(j), ",");
        for(int t = 0; t < tok.length; t++) {
          String tt=tok[t].trim();
          if(tt.equals(w)) return tok[0].trim();
        }
  }
  return "";
}