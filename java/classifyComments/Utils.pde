String posName(String pos) {
   if (pos.equals(RiWordNet.NOUN)) return "noun";
   if (pos.equals(RiWordNet.VERB)) return "verb";
   if (pos.equals(RiWordNet.ADJ)) return "adjective";
   if (pos.equals(RiWordNet.ADV)) return "adverb";
   return "";
}

// remove punctuation or special chars from the tree
String preprocess(String w) {
  String REGEXP = "[(){},.;!?<>%'`]";
  return w.trim().toLowerCase().replaceAll(REGEXP, "");
}

// returns only the unique elements of the list
IntList unique(IntList s) {
  IntList r = new IntList();
  
  for (int p=0; p<s.size(); p++) {
    if (!r.hasValue(s.get(p))) {
      r.append(s.get(p));
    } 
  }
  
  return r;
}