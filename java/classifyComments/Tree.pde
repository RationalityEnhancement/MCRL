// ------------------- Tree Attribute --------------
class Attribute {
  int num=0;
  String name="";
  float gain=0;
  float chi_square = 0;
  float dof = 0;
  
  Attribute() {
  }
}

// ------------------- Decision Tree  --------------
class Tree {
  
  // trees growing from his tree
  Tree leafsTrue = null;
  Tree leafsFalse = null;
  
  StringList examplesYes = null;
  StringList examplesNo = null;
  
  //String attribute; // word
  Attribute attribute;
  int label; 
  
  Tree() {
    attribute=null;
    label=0; // 0 means not a leaf
    x = 0;
    y = 0;
  }
  
  // when the tree is drawn these are the coordinates of the tree node
  int x;
  int y;
}

// ------------------- Tree training ------------------ 

float log2(float x) {
  return log(x)/log(2);
}

float B(float q) {
  if (q==0) return 0;
  if (q==1) return 0;
  return -(q*log2(q) + (1-q)*log2(1-q));
}


// Imporant! Beta can only tell the difference between two classes of labels, "yes" or "no"
// So if the classification has more labels than this you will need to train a tree for each label
 
float getBeta(IntList labels, IntList u) {
  
  int[] counts = numberOfTimesEachLabel(labels, u);
        
  // here we only care whether the label is equal to YesLabel
  float kt=0, kf = 0;
  for (int k=0; k<u.size(); k++) {
    if (u.get(k)==YesLabel) {
      kt+=counts[k];
    } else {
      kf+=counts[k];
    }
  }
  
  float b = B(kt/(kt+kf));
  return b;
}

void divideExamplesByAttr(StringList examples, IntList labels, String att,
      StringList s_present, IntList i_p, StringList s_absent, IntList i_a) {
  
  // divide the examples by this attribute
   for (int e=0; e<examples.size(); e++) {
    if (examples.get(e).indexOf(att) != -1) {
      s_present.append(examples.get(e));
      i_p.append(labels.get(e));
    } else {
      s_absent.append(examples.get(e));
      i_a.append(labels.get(e));
    }
  }
}

Attribute maxAttribute(StringList wordAttributes, StringList examples, IntList labels) {
  
    IntList u = unique(labels); 
    int[] counts = numberOfTimesEachLabel(labels, u);
    int p = 0, n=0;
    
    for (int k=0; k<u.size(); k++) {
      if (u.get(k)==YesLabel) {
        p+=counts[k];
      } else {
        n+=counts[k];
      }
    }
    
    int maxa = -1;
    float maxgain =-100;
    String maxname = "";
    for(int a = 0; a<wordAttributes.size(); a++ ){
      
        // number of each label examples
        String att = wordAttributes.get(a);
       
        StringList s_present = new StringList(); // examples in which this attribute is present
        IntList i_p = new IntList();
        
        StringList s_absent = new StringList(); // examples in which this attribute is absent
        IntList i_a = new IntList();
        
        divideExamplesByAttr( examples, labels, att, s_present, i_p, s_absent,i_a);
        
        // some such word that is either present or absent in all of the examples
        if (s_present.size()==0 || s_absent.size() == 0) continue;
        
        float b1 = getBeta(i_a, u); // might be 0, if no YesLabel occurs in the attribute-absent examples
        float b2 = getBeta(i_p, u); // might be 0, if no YesLabel occurs in the attribute-present examples
        float h1 = i_a.size()*b1;
        float h2 = i_p.size()*b2;
        float re = ( h1 +  h2)/labels.size();
        float gain = getBeta(labels, u) - re;
        if (gain > maxgain) {
          maxa=a;
          maxgain=gain;
          maxname = att;
        } else if (gain<0) {
          println(gain);
          continue;
        }
    }
    
    Attribute A = new Attribute();
    A.gain=maxgain;
    A.num=maxa;
    A.dof = examples.size()-1;
    
    StringList s_present = new StringList(); // examples in which this attribute is present
    IntList i_p = new IntList();
    StringList s_absent = new StringList(); // examples in which this attribute is absent
    IntList i_a = new IntList();
    
    divideExamplesByAttr( examples, labels, maxname, s_present, i_p, s_absent,i_a);
    
    float p_present_expected = float(p*i_p.size())/A.dof;
    float n_present_expected = float(n*i_p.size())/A.dof;
    float p_absent_expected = float(p*i_a.size())/A.dof;
    float n_absent_expected = float(n*i_a.size())/A.dof;
    
    int p_present_actual = 0;
    int n_present_actual = 0;
    int p_absent_actual = 0;
    int n_absent_actual = 0;
     
    for (int i=0; i< i_p.size(); i++) {
      if (i_p.get(i)==YesLabel) {
        p_present_actual++;
      } else {
        n_present_actual++;
      }
    }
    
    for (int i=0; i< i_a.size(); i++) {
      if (i_a.get(i)==YesLabel) {
        p_absent_actual++;
      } else {
        n_absent_actual++;
      }
    }
    
    A.chi_square =  sq(p_present_actual-p_present_expected)/p_present_expected + sq(n_present_actual-n_present_expected)/n_present_expected +
                    sq(p_absent_actual-p_absent_expected)/p_absent_expected  + sq(n_absent_actual-n_absent_expected)/n_absent_expected;
    
    if (maxa != -1) {
      A.name=wordAttributes.get(maxa);
    }
    
    return A;
}

Tree trainTheTree(StringList examples, IntList labels, StringList Parentexamples, IntList ParentLabels, StringList wordAttributes) {
  Tree t = new Tree();
  
  // if no more examples left return the majority label of the parent
  if (examples.size() < 1) {
    t.label = majorityLabel(ParentLabels);
    return t;
  } 
  
  StringList yes = new StringList();
  StringList no = new StringList();
  
  for (int e=0; e<examples.size(); e++) {
    if (labels.get(e) == YesLabel) {
      yes.append(examples.get(e));
    } else {
      no.append(examples.get(e));
    }
  }
  
  t.examplesYes = yes;
  t.examplesNo = no;
    
  // do all examples have the same label?
  boolean same = true;
  int label = labels.get(0);
  
  if (PluralityForYesLabelOnly) {
    label = (labels.get(0)==YesLabel) ? YesLabel : 0;
  }
  
  for (int p=1; p<labels.size(); p++) {
    
    if (!PluralityForYesLabelOnly) {
      if (labels.get(p)!=YesLabel) {
        same = false;
        break;
      }
    } else {
      int l = (labels.get(p)==YesLabel) ? YesLabel : 0;
      if (l!=label) {
        same = false;
        break;
      }
    }
  }

  if (same) {
    t.label = label;
    return t;
  } else {
    
    // if attributes are empty then return the majority label
    if (wordAttributes.size()<1) {
      t.label = majorityLabel(labels);
      return t;
    } else {
      Attribute A = maxAttribute(wordAttributes, examples, labels);
      if (A.num == -1) {
        println("can not select an attribute ");
        t.label = majorityLabel(labels);
        return t;
      }
      
      int att = A.num;
      String maxa = wordAttributes.get(att);
      wordAttributes.remove(att);
      
      StringList s_present = new StringList(); // examples in which this attribute is present
      IntList i_p = new IntList();
      StringList s_absent = new StringList(); // examples in which this attribute is absent
      IntList i_a = new IntList();
    
      divideExamplesByAttr( examples, labels, maxa, s_present, i_p, s_absent,i_a);
      
      t.attribute = A;//maxa;
      println("attribute ", A.name, "(", A.chi_square, A.dof, ")");
      
      //t.gain = A.gain;
      t.leafsTrue  = trainTheTree(s_present, i_p, examples, labels, wordAttributes);
      t.leafsFalse = trainTheTree(s_absent,  i_a, examples, labels, wordAttributes);
    }
  }

  return t;
}

int majorityLabel(IntList ParentLabels ) {
  
    IntList u = unique(ParentLabels); 
    int[] counts = numberOfTimesEachLabel(ParentLabels, u);
      
    if (!PluralityForYesLabelOnly) {
      int parent = 0;
      
      // which labels does the node have?
      for (int j=1; j<u.size(); j++) {
        if (counts[parent] < counts[j]) parent=j;
      }
      
      return u.get(parent);
    } else {
      
      // we only want to return 0 or 1
      // return 1 if the majority label is teh YesLabel  and 0 if it is something else
      int countyes = 0;
      int countno = 0;
      
      for (int j=0; j<u.size(); j++) {
        if ( u.get(j) == YesLabel ) {
          countyes += counts[j];
        } else {
          countno += counts[j];
        }
      }
      return (countyes > countno) ? YesLabel : 0;
    }
}

// conts the number of times each integer label occurs in ParentLabels
int[] numberOfTimesEachLabel(IntList ParentLabels, IntList u) {

    // count how many of each label
    int[] counts = new int[u.size()];
    for (int j=0; j<u.size(); j++) counts[j]=0;
    
    for (int p=0; p<ParentLabels.size(); p++) {
      for (int j=0; j<u.size(); j++) {
        if (ParentLabels.get(p) == u.get(j)) {
          counts[j]++;
        }
      }
    }
    
    return counts;
}