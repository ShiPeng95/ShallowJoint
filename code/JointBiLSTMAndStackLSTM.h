#ifndef CPYPDICT_H_
#define CPYPDICT_H_

#include <string>
#include <iostream>
#include <cassert>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <unordered_map>
#include <functional>
#include <vector>
#include <map>
#include <string>

namespace cpyp {

class Corpus {

public:
  std::map<int,std::vector<unsigned>> correct_act_sent;
  std::map<int,std::vector<unsigned>> sentences;
  std::map<int,std::vector<unsigned>> sentencesPos;
  std::map<int,std::vector<unsigned>> sentencesDep;
  std::map<int,std::vector<unsigned>> sentencesPredDep;
  std::map<int,std::vector<unsigned>> sentencesSrl;
  std::map<int,std::vector<unsigned>> sentencesPredSrl;
  std::map<int,std::vector<unsigned>> sentencesFather;
  std::map<int,std::vector<std::vector<unsigned>>> sentencesChar;

  std::map<int,std::vector<unsigned>> correct_act_sentDev;
  std::map<int,std::vector<unsigned>> sentencesDev;
  std::map<int,std::vector<unsigned>> sentencesPosDev;
  std::map<int,std::vector<unsigned>> sentencesDepDev;
  std::map<int,std::vector<unsigned>> sentencesPredDepDev;
  std::map<int,std::vector<unsigned>> sentencesSrlDev;
  std::map<int,std::vector<unsigned>> sentencesPredSrlDev;
  std::map<int,std::vector<unsigned>> sentencesFatherDev;
  std::map<int,std::vector<std::vector<unsigned>>> sentencesCharDev;
  std::map<int,std::vector<std::string>> sentencesStrDev;

  std::map<int,std::vector<unsigned>> correct_act_sentTest;
  std::map<int,std::vector<unsigned>> sentencesTest;
  std::map<int,std::vector<unsigned>> sentencesPosTest;
  std::map<int,std::vector<unsigned>> sentencesDepTest;
  std::map<int,std::vector<unsigned>> sentencesPredDepTest;
  std::map<int,std::vector<unsigned>> sentencesSrlTest;
  std::map<int,std::vector<unsigned>> sentencesPredSrlTest;
  std::map<int,std::vector<unsigned>> sentencesFatherTest;
  std::map<int,std::vector<std::vector<unsigned>>> sentencesCharTest;
  std::map<int,std::vector<std::string>> sentencesStrTest;

  unsigned nsentencesTrain;
  unsigned nsentencesDev;
  unsigned nsentencesTest;

  unsigned nwords;
  unsigned nwordsDev;
  unsigned nwordsTest;
  unsigned nchars;
  unsigned ncharsDev;
  unsigned ncharsTest;
  unsigned nactions;
  unsigned npos;
  unsigned ndeps;
  unsigned nsrls;

  unsigned max;
  unsigned maxChars;
  unsigned maxPos;
  unsigned maxDep;
  unsigned maxSrl;


  std::map<std::string, unsigned> wordsToInt;
  std::map<unsigned, std::string> intToWords;

  std::vector<std::string> actions;

  std::map<std::string, unsigned> posToInt;
  std::map<unsigned, std::string> intToPos;

  std::map<std::string, unsigned> charsToInt;
  std::map<unsigned, std::string> intToChars;

  std::map<std::string, unsigned> depToInt;
  std::map<unsigned, std::string> intToDep;

  std::map<std::string, unsigned> srlToInt;
  std::map<unsigned, std::string> intToSrl;

  static constexpr const char* UNK = "UNK";
  static constexpr const char* BAD0 = "<BAD0>";
  static constexpr const char* CUNK = "CUNK";
  static constexpr const char* CBAD0 = "<CBAD0>";
  //static constexpr const char* BAD0 = "<BAD0>";
  //Because if the key not exist, it will return 0
  

public:
  Corpus() {

  }

  inline unsigned UTF8Len(unsigned char x) {
    if (x < 0x80) return 1;
    else if ((x >> 5) == 0x06) return 2;
    else if ((x >> 4) == 0x0e) return 3;
    else if ((x >> 3) == 0x1e) return 4;
    else if ((x >> 2) == 0x3e) return 5;
    else if ((x >> 1) == 0x7e) return 6;
    else return 0;
  }

  inline void load_correct_actions(std::string file){
    std::cerr << "Reading file from " << file << "\n";
    std::ifstream actionsFile(file);
    std::string lineS;

    int count=-1;
    int sentence=-1;
    bool initial=false;
    bool first=true;
    wordsToInt[Corpus::BAD0] = 0;
    intToWords[0] = Corpus::BAD0;
    wordsToInt[Corpus::UNK] = 1; // unknown word
    intToWords[1] = Corpus::UNK;

    charsToInt[Corpus::CBAD0] = 0;
    intToChars[0] = Corpus::CBAD0;
    charsToInt[Corpus::CUNK] = 1; // unknown char
    intToChars[1] = Corpus::CUNK;

    assert(max == 0);
    assert(maxChars == 0);
    assert(maxPos == 0);
    assert(maxDep == 0);
    assert(maxSrl == 0);

    max = 2;
    maxChars = 2;
    maxPos = 1;
    maxDep = 1;
    maxSrl = 1;

    std::vector<unsigned> current_sent;
    std::vector<std::vector<unsigned>> current_sent_char;
    std::vector<unsigned> current_sent_pos;
    std::vector<unsigned> current_sent_dep;
    std::vector<unsigned> current_sent_srl;
    while (getline(actionsFile, lineS)) {
      /*
      std::cerr << countLine << "\n";
      if (lineS.empty()) {
        std::cerr << "This line is empty\n";
        abort();
      }
      */
      if (lineS.empty()) {
        count = 0;
        if (!first) { // add the sentence information here just once
          sentences[sentence] = current_sent;
          sentencesChar[sentence] = current_sent_char;
          sentencesPos[sentence] = current_sent_pos;
          sentencesDep[sentence] = current_sent_dep;
          sentencesSrl[sentence] = current_sent_srl;
        }

        sentence ++;
        nsentencesTrain = sentence;
        initial = true;
        current_sent.clear();
        current_sent_char.clear();
        current_sent_pos.clear();
        current_sent_dep.clear();
        current_sent_srl.clear();
      } else if (count == 0) {
        first = false;
        count = 1;

        if (initial){
          // the initial ine in each sentence may look like:
          // [][verb-pos-dep-srl, ROOT-ROOT-ROOT-ROOT]
          // first, get rid of the square brackets.
          lineS = lineS.substr(3, lineS.size() - 4);
          // read the initial line, token by token "the-det," "cat-noun," ...
          std::istringstream iss(lineS);

          do {
            std::string token;
            iss >> token;

            std::string word;
            std::string pos;
            std::string dep;
            std::string srl;

            if (token.size() == 0) {continue;}
            // split the string at '-' in to word , POS, dep, SRL tag.
            if (token[token.size() - 1] == ',') { 
              token = token.substr(0, token.size() - 1);
            }
            size_t wordIndex = token.find('-');
            word = token.substr(0, wordIndex);
            token = token.substr(wordIndex+1);

            size_t posIndex = token.find('-');
            pos = token.substr(0, posIndex);
            token = token.substr(posIndex+1);

            size_t depIndex = token.find('-');
            dep = token.substr(0, depIndex);
            token = token.substr(depIndex+1);

            srl = token.substr(0);

            // check the token here
            //std::cerr << word << pos << dep << srl << "\n";

            // new word
            if (wordsToInt[word] == 0) {
              wordsToInt[word] = max;
              intToWords[max] = word;
              nwords = max; // the number of words is include UNK but not BAD0
              max ++;

              unsigned j = 0;
              while (j < word.length()) {
                std::string wj = "";
                for (unsigned h = j; h < j + UTF8Len(word[j]); h++) {
                  wj += word[h];
                }
                if (charsToInt[wj] == 0) {
                  charsToInt[wj] = maxChars;
                  intToChars[maxChars] = wj;
                  nchars = maxChars;
                  maxChars ++;
                }
                j += UTF8Len(word[j]);
              }
            } // end of new words

            //add char
            std::vector<unsigned> current_word_char;
            unsigned j = 0;
            while (j < word.length()) {
              std::string wj = "";
              for (unsigned h = j; h < j + UTF8Len(word[j]); h++) {
                wj += word[h];
              }
              current_word_char.push_back(charsToInt[wj]);
              j += UTF8Len(word[j]);
            }

            //new POS
            if (posToInt[pos] == 0) {
              posToInt[pos] = maxPos;
              intToPos[maxPos] = pos;
              npos = maxPos;
              maxPos ++;
            }

            //new Dep
            if (depToInt[dep] == 0) {
              depToInt[dep] = maxDep;
              intToDep[maxDep] = dep;
              ndeps = maxDep;
              maxDep ++;
            }

            //new SRL
            if (srlToInt[srl] == 0) {
              srlToInt[srl] = maxSrl;
              intToSrl[maxSrl] = srl;
              nsrls = maxSrl;
              maxSrl ++;
            }

            current_sent.push_back(wordsToInt[word]);
            current_sent_char.push_back(current_word_char);
            current_sent_pos.push_back(posToInt[pos]);
            current_sent_dep.push_back(depToInt[dep]);
            current_sent_srl.push_back(srlToInt[srl]);


          } while (iss);// end of reading tokens
        } // end of initial
        initial = false;

      } // end of count == 0
      else if (count == 1) {
        int i = 0;
        bool found = false;
        for (auto a : actions) {
          if (a == lineS) {
            std::vector<unsigned> a = correct_act_sent[sentence];
            a.push_back(i);
            correct_act_sent[sentence] = a;
            found = true;
          }
          i++;
        }
        if (!found) {
          actions.push_back(lineS);
          std::vector<unsigned> a = correct_act_sent[sentence];
          a.push_back(actions.size()-1);
          correct_act_sent[sentence] = a;
        }
        count = 0;
      } // end of count == 1
    } // end while

    // Add the last sentence
    if (current_sent.size() > 0) {
      sentences[sentence] = current_sent;
      sentencesChar[sentence] = current_sent_char;
      sentencesPos[sentence] = current_sent_pos;
      sentencesDep[sentence] = current_sent_dep;
      sentencesSrl[sentence] = current_sent_srl;
      sentence++;
      nsentencesTrain = sentence;
    }

    actionsFile.close();

    nactions = actions.size();

    // Check the information of corpus here
    std::cerr << "nwords: " << nwords-2 << "\n";
    std::cerr << "nchars: " << nchars-2 << "\n";
    std::cerr << "npos: " << npos-1 << "\n";
    std::cerr << "ndeps: " << ndeps-1 << "\n";
    std::cerr << "nsrls: " << nsrls-1 << "\n";
    std::cerr << "nactions: " << nactions << "\n";

    for (auto a : actions) {
      std::cerr << a << "\n";
    }

    for (unsigned i=0; i<npos; i++) {
      std::cerr << i << " "<<intToPos[i] << "\n";
    }

    for (unsigned i=0; i<ndeps; i++) {
      std::cerr << i<<" "<<intToDep[i] << "\n";
    }

    for (unsigned i=0; i<nsrls; i++) {
      std::cerr << i << " " << intToSrl[i] << "\n";
    }
    std::cerr << "There are " << nsentencesTrain << " in training data" << "\n";



  } // end reading from training corpus

  inline void load_correct_actionsDev(std::string file) {
    std::ifstream actionsFile(file);
    std::string lineS;

    assert(maxPos > 1);
    assert(max > 3);
    assert(maxChars > 3);
    assert(maxDep > 1);
    assert(maxSrl > 1);

    int count = -1;
    int sentence = -1;
    bool initial = false;
    bool first = true;

    std::vector<unsigned> current_sent;
    std::vector<std::vector<unsigned> > current_sent_char;
    std::vector<unsigned> current_sent_pos;
    std::vector<unsigned> current_sent_dep;
    std::vector<unsigned> current_sent_srl;
    std::vector<std::string> current_sent_str;

    while (getline(actionsFile, lineS)) {
      if (lineS.empty()) {
        // an empty line marks the end of a sentence
        count = 0;
        if (!first) {
          sentencesDev[sentence] = current_sent;
          sentencesCharDev[sentence] = current_sent_char;
          sentencesPosDev[sentence] = current_sent_pos;
          sentencesStrDev[sentence] = current_sent_str;
          sentencesDepDev[sentence] = current_sent_dep;
          sentencesSrlDev[sentence] = current_sent_srl;
        } // end first

        sentence++;
        nsentencesDev = sentence;

        initial = true;
        current_sent.clear();
        current_sent_char.clear();
        current_sent_pos.clear();
        current_sent_dep.clear();
        current_sent_srl.clear();
        current_sent_str.clear();
      } else if (count == 0) {
        first = false;
        count = 1;
        if (initial) {
          lineS = lineS.substr(3, lineS.size() - 4);
          std::istringstream iss(lineS);
          do {
            std::string token;
            iss >> token;
            if (token.size()  == 0) { continue; }
            if (token[token.size() - 1] == ',') { 
              token = token.substr(0, token.size() - 1);
            }

            //split token at '-' into word pos dep srl

            std::string word;
            std::string pos;
            std::string dep;
            std::string srl;

            size_t wordIndex = token.find('-');
            word = token.substr(0, wordIndex);
            token = token.substr(wordIndex+1);

            size_t posIndex = token.find('-');
            pos = token.substr(0, posIndex);
            token = token.substr(posIndex+1);

            size_t depIndex = token.find('-');
            dep = token.substr(0, depIndex);
            token = token.substr(depIndex+1);

            srl = token.substr(0);

            // check the token here
            //std::cerr << word << pos << dep << srl << "\n";
            // new POS, we not allow new POS tag
            if (posToInt[pos] == 0) {
              std::cerr << "Unknown POS" << pos << "\n";
            }
            // new DEP, we not allow new DEP tag
            if (depToInt[dep] == 0) {
              std::cerr << "Unknown DEP" << dep << "\n";
            }
            // new SRL, we not allow new SRL tag
            if (srlToInt[srl] == 0) {
              std::cerr << "Unknown SRL" << srl << "\n";
            }
           

            std::vector<unsigned> current_word_char;
            unsigned j = 0;
            while(j < word.length()) {
              std::string wj = "";
              for (unsigned h = j; h < j + UTF8Len(word[j]); h++) {
                wj += word[h];
              }
              if (charsToInt[wj] == 0) {
                wj = Corpus::CUNK;
              }
              current_word_char.push_back(charsToInt[wj]);
              j += UTF8Len(word[j]);
            }

           
	    // new word, we set as UNK, but save the format
            current_sent_str.push_back("");
            if (wordsToInt[word] == 0) {
              current_sent_str[current_sent_str.size()-1] = word;
              word = Corpus::UNK;
            }
	    


            



            current_sent.push_back(wordsToInt[word]);
            current_sent_char.push_back(current_word_char);
            current_sent_pos.push_back(posToInt[pos]);
            current_sent_dep.push_back(depToInt[dep]);
            current_sent_srl.push_back(srlToInt[srl]);
          } while(iss);

        }

        initial = false;
      }
      else if (count == 1) {
        auto actionIter = std::find(actions.begin(), actions.end(), lineS);
        if (actionIter != actions.end()) {
          unsigned actionIndex = std::distance(actions.begin(), actionIter);
          correct_act_sentDev[sentence].push_back(actionIndex);
        } else {
          std::cerr << "Unknown Actions : " << lineS << "\n";
        }
        count = 0;
      }
    } // end of while loop that read corpus
    
    // Add the last sentence
    if (current_sent.size() > 0) {
      sentencesDev[sentence] = current_sent;
      sentencesCharDev[sentence] = current_sent_char;
      sentencesPosDev[sentence] = current_sent_pos;
      sentencesDepDev[sentence] = current_sent_dep;
      sentencesSrlDev[sentence] = current_sent_srl;
      sentencesStrDev[sentence] = current_sent_str;
      sentence++;
      nsentencesDev = sentence;
    }

    std::cerr << "There are " << nsentencesDev << " in dev data" << "\n";
    actionsFile.close();

  } // end of inline : reading corpus from dev

  inline void load_correct_actionsTest(std::string file) {
    std::ifstream actionsFile(file);
    std::string lineS;

    assert(maxPos > 1);
    assert(max > 3);
    assert(maxChars > 3);
    assert(maxDep > 1);
    assert(maxSrl > 1);

    int count = -1;
    int sentence = -1;
    bool initial = false;
    bool first = true;

    std::vector<unsigned> current_sent;
    std::vector<std::vector<unsigned> > current_sent_char;
    std::vector<unsigned> current_sent_pos;
    std::vector<unsigned> current_sent_dep;
    std::vector<unsigned> current_sent_srl;
    std::vector<std::string> current_sent_str;

    while (getline(actionsFile, lineS)) {
      if (lineS.empty()) {
        // an empty line marks the end of a sentence
        count = 0;
        if (!first) {
          sentencesTest[sentence] = current_sent;
          sentencesCharTest[sentence] = current_sent_char;
          sentencesPosTest[sentence] = current_sent_pos;
          sentencesStrTest[sentence] = current_sent_str;
          sentencesDepTest[sentence] = current_sent_dep;
          sentencesSrlTest[sentence] = current_sent_srl;
        } // end first

        sentence++;
        nsentencesTest = sentence;

        initial = true;
        current_sent.clear();
        current_sent_char.clear();
        current_sent_pos.clear();
        current_sent_dep.clear();
        current_sent_srl.clear();
        current_sent_str.clear();
      } else if (count == 0) {
        first = false;
        count = 1;
        if (initial) {
          lineS = lineS.substr(3, lineS.size() - 4);
          std::istringstream iss(lineS);
          do {
            std::string token;
            iss >> token;
            if (token.size()  == 0) { continue; }
            if (token[token.size() - 1] == ',') { 
              token = token.substr(0, token.size() - 1);
            }

            //split token at '-' into word pos dep srl

            std::string word;
            std::string pos;
            std::string dep;
            std::string srl;

            size_t wordIndex = token.find('-');
            word = token.substr(0, wordIndex);
            token = token.substr(wordIndex+1);

            size_t posIndex = token.find('-');
            pos = token.substr(0, posIndex);
            token = token.substr(posIndex+1);

            size_t depIndex = token.find('-');
            dep = token.substr(0, depIndex);
            token = token.substr(depIndex+1);

            srl = token.substr(0);

            // check the token here
            //std::cerr << word << pos << dep << srl << "\n";
            // new POS, we not allow new POS tag
            if (posToInt[pos] == 0) {
              std::cerr << "Unknown POS" << pos << "\n";
            }
            // new DEP, we not allow new DEP tag
            if (depToInt[dep] == 0) {
              std::cerr << "Unknown DEP" << dep << "\n";
            }
            // new SRL, we not allow new SRL tag
            if (srlToInt[srl] == 0) {
              std::cerr << "Unknown SRL" << srl << "\n";
            }
	         

            std::vector<unsigned> current_word_char;
            unsigned j = 0;
            while(j < word.length()) {
              std::string wj = "";
              for (unsigned h = j; h < j + UTF8Len(word[j]); h++) {
                wj += word[h];
              }
              if (charsToInt[wj] == 0) {
                wj = Corpus::CUNK;
              }
              current_word_char.push_back(charsToInt[wj]);
              j += UTF8Len(word[j]);
            }
	    
            // new word, we set as UNK, but save the format
            current_sent_str.push_back("");
            if (wordsToInt[word] == 0) {
              current_sent_str[current_sent_str.size()-1] = word;
              word = Corpus::UNK;
            }
	    


            



            current_sent.push_back(wordsToInt[word]);
            current_sent_char.push_back(current_word_char);
            current_sent_pos.push_back(posToInt[pos]);
            current_sent_dep.push_back(depToInt[dep]);
            current_sent_srl.push_back(srlToInt[srl]);
          } while(iss);
        }
        
        initial = false;
      }
      else if (count == 1) {
        auto actionIter = std::find(actions.begin(), actions.end(), lineS);
        if (actionIter != actions.end()) {
          unsigned actionIndex = std::distance(actions.begin(), actionIter);
          correct_act_sentTest[sentence].push_back(actionIndex);
        } else {
          std::cerr << "Unknown Actions : " << lineS << "\n";
        }
        count = 0;
      }
    } // end of while loop that read corpus
    
    // Add the last sentence
    if (current_sent.size() > 0) {
      sentencesTest[sentence] = current_sent;
      sentencesCharTest[sentence] = current_sent_char;
      sentencesPosTest[sentence] = current_sent_pos;
      sentencesDepTest[sentence] = current_sent_dep;
      sentencesSrlTest[sentence] = current_sent_srl;
      sentencesStrTest[sentence] = current_sent_str;
      sentence++;
      nsentencesTest = sentence;
    }

    std::cerr << "There are " << nsentencesTest << " in test data" << "\n";
    actionsFile.close();

  } // end of inline : reading corpus from test

  inline unsigned get_or_add_word(const std::string& word) {
    unsigned& id = wordsToInt[word];
    if (id == 0) {
      id = max;
      ++max;
      intToWords[id] = word;
      nwords = max;
    }
    return id;
  }





}; // end of class Corpus


} // end of namespace

#endif
