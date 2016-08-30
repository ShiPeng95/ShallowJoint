#include <cstdlib>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <vector>
#include <limits>
#include <cmath>
#include <chrono>
#include <ctime>

#include <unordered_map>
#include <unordered_set>

#include <execinfo.h>
#include <unistd.h>
#include <signal.h>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/program_options.hpp>

#include "cnn/training.h"
#include "cnn/cnn.h"
#include "cnn/expr.h"
#include "cnn/nodes.h"
#include "cnn/lstm.h"
#include "cnn/rnn.h"

#include "JointBiLSTMAndStackLSTM.h"

cpyp::Corpus corpus;
volatile bool requested_stop = false;
unsigned BiLSTMLAYERS;// = 1;
unsigned STACKLSTMLAYERS;// = 1;
unsigned WORD_DIM;// = 64;
unsigned CHAR_DIM;// = 50;
unsigned CHAR_HIDDEN_DIM;// = 50;
unsigned POS_DIM;// = 30;
unsigned DEP_DIM;// = 30;
unsigned REL_DIM;// = 30;
unsigned SRL_DIM;// = 30;
unsigned B_LSTM_INPUT_DIM;// = 128;
unsigned S_LSTM_INPUT_DIM;// = 128;
unsigned B_HIDDEN_DIM;// = 128;
unsigned S_HIDDEN_DIM;// = 128;
unsigned TAG_HIDDEN_DIM;// = 32;
unsigned ACTION_DIM;// = 30;
unsigned PRETRAINED_DIM;// = 50;




bool USE_WORD = false;
bool USE_CHAR = false;
bool USE_POS = false;
bool USE_DEP = false;
bool USE_SRL = false;
bool USE_PRETRAIN = false;
bool USE_COMBINATION = false;
bool USE_DROPOUT = false;
bool SHARE_CHAR_LSTM = false;
bool PARSER = false;
bool LABELER = false;
bool PARSER_TIME = false;
bool LABELER_TIME = false;
bool LOOP_TRAIN = false;
bool JOINT_TRAIN_SRL_DEP = false;
bool JOINT_TRAIN_DEP_SRL = false;
//bool USE_DEP_INLOOP = false;


unsigned VOCAB_SIZE = 0;
unsigned CHAR_SIZE = 0;
unsigned POS_SIZE = 0;
unsigned DEP_SIZE = 0;
unsigned SRL_SIZE = 0;
unsigned ACTION_SIZE = 0;
unsigned RELATION_SIZE = 0;

using namespace cnn::expr;
using namespace cnn;
using namespace std;
namespace po = boost::program_options;

vector<unsigned> possible_actions;
vector<unsigned> possible_label;

unordered_map<unsigned, vector<float>> pretrained;

void InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()
    ("training_data,T", po::value<string>(), "List of Transitions - Training corpus")
    ("dev_data,d", po::value<string>(), "Development corpus")
    ("test_data,p", po::value<string>(), "Test corpus")
    ("test,e", "Should run evaluate every refresh?")
    ("unk_strategy,o", po::value<unsigned>()->default_value(1), "Unknown word strategy: 1 = singletons become UNK with probability unk_prob")
    ("unk_prob,u", po::value<double>()->default_value(0.2), "Probably with which to replace singletons with UNK in training data")
    ("model,m", po::value<string>(), "Load saved model from this file")
    ("use_word,W", "Use word")
    ("use_char,S", "Use char")
    ("use_pos,P", "make POS tags visible to parser")
    ("use_dep,D", "Use dep")
    ("use_srl,L", "Use srl")
    ("use_combination,C", "Use combination")
    ("use_dropout,r", "Use dropout")
    ("labeler,a", "Labeler Task?")
    ("parser,b", "Parser Task?")
    ("loop_train,l","Loop train?")
    ("joint_train_srl_dep,j", "Joint Train From SRL to DEP?")
    ("joint_train_dep_srl,k", "Joint Train From DEP to SRL?")
    ("sampling,s", "Schedul sampling?")
    ("bilstm_layers", po::value<unsigned>()->default_value(1), "number of Bi-LSTM layers")
    ("stack_lstm_layers", po::value<unsigned>()->default_value(1), "number of LSTM layers")
    ("action_dim", po::value<unsigned>()->default_value(30), "action embedding size")
    ("word_dim", po::value<unsigned>()->default_value(64), "input embedding size")
    ("char_dim", po::value<unsigned>()->default_value(32), "input char dim")
    ("char_hidden_dim", po::value<unsigned>()->default_value(64), "hidden char dim")
    ("pretrained_dim", po::value<unsigned>()->default_value(50), "pretrained input dimension")
    ("pos_dim", po::value<unsigned>()->default_value(30), "POS dimension")
    ("dep_dim", po::value<unsigned>()->default_value(30), "DEP dimension")
    ("rel_dim", po::value<unsigned>()->default_value(30), "relation dimension")
    ("srl_dim", po::value<unsigned>()->default_value(30), "SRL dimension")
    ("b_lstm_input_dim", po::value<unsigned>()->default_value(128), "Bi LSTM input dim")
    ("s_lstm_input_dim", po::value<unsigned>()->default_value(128), "Stack LSTM input dimension")
    ("b_hidden_dim", po::value<unsigned>()->default_value(128), "The hidden dim of Bi LSTM")
    ("s_hidden_dim", po::value<unsigned>()->default_value(128), "The hidden dim of Stack LSTM")
    ("tag_hidden_dim", po::value<unsigned>()->default_value(64), "tag hidden dim")
    ("train,t", "Should training be run?")
    ("words,w", po::value<string>(), "Pretrained word embeddings")
    ("help,h", "Help");
  po::options_description dcmdline_options;
  dcmdline_options.add(opts);
  po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
  if (conf->count("help")) {
    cerr << dcmdline_options << endl;
    exit(1);
  }
  if (conf->count("training_data") == 0) {
    cerr << "Please specify --traing_data (-T): this is required to determine the vocabulary mapping, even if the parser is used in prediction mode.\n";
    exit(1);
  }
}

struct JointModel {
  /* Sharing in Bi-LSTM and Stack-LSTM */
  LookupParameters* p_w;
  LookupParameters* p_t;
  LookupParameters* p_ch;
  LookupParameters* p_pos;
  LookupParameters* p_dep;
  LookupParameters* p_srl;

  /* Parameter in Bi-LSTM */
  // LSTM
  LSTMBuilder l2rbuilder;
  LSTMBuilder r2lbuilder;
  LSTMBuilder c_l2rbuilder;
  LSTMBuilder c_r2lbuilder;

  // Parameters in combining input to Bi-LSTM input
  Parameters* p_w2bil;
  Parameters* p_cf2bil;
  Parameters* p_cb2bil;
  Parameters* p_p2bil;
  Parameters* p_d2bil;
  Parameters* p_t2bil;
  Parameters* p_bilbias; 

  // Parameters in hidden layer
  Parameters* p_l2th;
  Parameters* p_r2th;
  Parameters* p_thbias;

  // Parameters in output layer
  Parameters* p_th2t;
  Parameters* p_tbias;

  // Guard
  Parameters* p_word_start;
  Parameters* p_word_end;


  /* Parameter in Stack-LSTM */
  // LSTM
  LSTMBuilder stack_lstm;
  LSTMBuilder buffer_lstm;
  LSTMBuilder action_lstm;
  LSTMBuilder cs_l2rbuilder;
  LSTMBuilder cs_r2lbuilder;

  // Lookuptable
  LookupParameters* p_a; // input action embeddings
  LookupParameters* p_r; // relation embeddings

  // Parameters in combining Stack LSTM input
  Parameters* p_w2sl; // word to LSTM input
  Parameters* p_cf2sl;
  Parameters* p_cb2sl;
  Parameters* p_p2sl; // POS to LSTM input
  Parameters* p_t2sl; // pretrained word embeddings to LSTM input
  Parameters* p_srl2sl; // SRL label to LSTM input
  Parameters* p_slbias; // LSTM input bias

  // Parameters in predicting the actions
  // input -> parser state
  Parameters* p_pbias; // parser state bias
  Parameters* p_A; // action lstm to parser state
  Parameters* p_B; // buffer lstm to parser state
  Parameters* p_S; // stack lstm to parser state
  // parser state -> action
  Parameters* p_p2a;   // parser state to action
  Parameters* p_abias;  // action bias

  // Parameters in combining two elements in stack
  Parameters* p_H; // head matrix for composition function
  Parameters* p_D; // dependency matrix for composition function
  Parameters* p_R; // relation matrix for composition function
  Parameters* p_cbias; // composition function bias

  // guard
  Parameters* p_action_start;  // action bias
  Parameters* p_buffer_guard;  // end of buffer
  Parameters* p_stack_guard;  // end of stack
  Parameters* p_s_word_start;
  Parameters* p_s_word_end;

  explicit JointModel(Model& model, const unordered_map<unsigned, vector<float>>& pretrained) :
    stack_lstm(STACKLSTMLAYERS, S_LSTM_INPUT_DIM, S_HIDDEN_DIM, &model),
    buffer_lstm(STACKLSTMLAYERS, S_LSTM_INPUT_DIM, S_HIDDEN_DIM, &model),
    action_lstm(STACKLSTMLAYERS, ACTION_DIM, S_HIDDEN_DIM, &model),
    l2rbuilder(BiLSTMLAYERS ,B_LSTM_INPUT_DIM, B_HIDDEN_DIM, &model),
    r2lbuilder(BiLSTMLAYERS, B_LSTM_INPUT_DIM, B_HIDDEN_DIM, &model),
    c_l2rbuilder(BiLSTMLAYERS, CHAR_DIM, CHAR_HIDDEN_DIM, &model),
    c_r2lbuilder(BiLSTMLAYERS, CHAR_DIM, CHAR_HIDDEN_DIM, &model),
    cs_l2rbuilder(BiLSTMLAYERS, CHAR_DIM, CHAR_HIDDEN_DIM, &model),
    cs_r2lbuilder(BiLSTMLAYERS, CHAR_DIM, CHAR_HIDDEN_DIM, &model) {
      /* The Parameter for Lookup table and LSTM_INPUT */
      if (USE_WORD) {
        p_w = model.add_lookup_parameters(VOCAB_SIZE, {WORD_DIM});
        if (USE_COMBINATION) {
          if (LABELER) p_w2bil = model.add_parameters({B_LSTM_INPUT_DIM, WORD_DIM});
          if (PARSER) p_w2sl = model.add_parameters({S_LSTM_INPUT_DIM, WORD_DIM});
        }
        
      }
        
      if (USE_CHAR) {
        p_ch = model.add_lookup_parameters(CHAR_SIZE, {CHAR_DIM});
        if (LABELER) {
          p_word_start = model.add_parameters({CHAR_DIM});
          p_word_end = model.add_parameters({CHAR_DIM});
        }
        if (PARSER) {
          p_s_word_start = model.add_parameters({CHAR_DIM});
          p_s_word_end = model.add_parameters({CHAR_DIM});
        }
        
        if (USE_COMBINATION) {
          if (LABELER) {
            p_cf2bil = model.add_parameters({B_LSTM_INPUT_DIM, CHAR_HIDDEN_DIM});
            p_cb2bil = model.add_parameters({B_LSTM_INPUT_DIM, CHAR_HIDDEN_DIM});
          }
          if (PARSER) {
            p_cf2sl = model.add_parameters({S_LSTM_INPUT_DIM, CHAR_HIDDEN_DIM});
            p_cb2sl = model.add_parameters({S_LSTM_INPUT_DIM, CHAR_HIDDEN_DIM});
          }
        }
      }

      if (USE_POS) {
        p_pos = model.add_lookup_parameters(POS_SIZE, {POS_DIM});
        if (USE_COMBINATION) {
          if (LABELER) p_p2bil = model.add_parameters({B_LSTM_INPUT_DIM, POS_DIM});
          if (PARSER) p_p2sl = model.add_parameters({S_LSTM_INPUT_DIM, POS_DIM});
        }
      }

      if (USE_DEP) {
        p_dep = model.add_lookup_parameters(DEP_SIZE, {DEP_DIM});
        if (USE_COMBINATION) {
          if (LABELER) p_d2bil = model.add_parameters({B_LSTM_INPUT_DIM, DEP_DIM});
        }
      }

      if (USE_SRL) {
        p_srl = model.add_lookup_parameters(SRL_SIZE, {SRL_DIM});
        if (USE_COMBINATION) {
          if (PARSER) p_srl2sl = model.add_parameters({S_LSTM_INPUT_DIM, SRL_DIM});
        }
      }

      if (USE_COMBINATION) {
        if (LABELER) p_bilbias = model.add_parameters({B_LSTM_INPUT_DIM});
        if (PARSER) p_slbias = model.add_parameters({S_LSTM_INPUT_DIM});
      }

      if (pretrained.size() > 0) {
        p_t = model.add_lookup_parameters(VOCAB_SIZE, {PRETRAINED_DIM});
        for (auto it : pretrained) 
          p_t -> Initialize(it.first, it.second);
        if (LABELER) p_t2bil = model.add_parameters({B_LSTM_INPUT_DIM, PRETRAINED_DIM});
        if (PARSER) p_t2sl = model.add_parameters({S_LSTM_INPUT_DIM, PRETRAINED_DIM});
      } else {
        p_t = nullptr;
        p_t2bil = nullptr;
        p_t2sl = nullptr;
      }

      /* The Parameter For Labeler Task */
      if (LABELER) {
        p_l2th = model.add_parameters({TAG_HIDDEN_DIM, B_HIDDEN_DIM});
        p_r2th = model.add_parameters({TAG_HIDDEN_DIM, B_HIDDEN_DIM});
        p_thbias = model.add_parameters({TAG_HIDDEN_DIM});

        p_th2t = model.add_parameters({SRL_SIZE, TAG_HIDDEN_DIM});
        p_tbias = model.add_parameters({SRL_SIZE});
      }

      /* The Parameter For Parser Task */
      if (PARSER) {
        // Look up table
        p_a = model.add_lookup_parameters(ACTION_SIZE,{ACTION_DIM});
        p_r = model.add_lookup_parameters(ACTION_SIZE,{REL_DIM});

        // input -> parser state
        p_pbias = model.add_parameters({S_HIDDEN_DIM});
        p_A = model.add_parameters({S_HIDDEN_DIM, S_HIDDEN_DIM});
        p_B = model.add_parameters({S_HIDDEN_DIM, S_HIDDEN_DIM});
        p_S = model.add_parameters({S_HIDDEN_DIM, S_HIDDEN_DIM});

        // parser state -> action
        p_p2a = model.add_parameters({ACTION_SIZE, S_HIDDEN_DIM});
        p_abias = model.add_parameters({ACTION_SIZE});

        // stack elements -> composition
        p_cbias = model.add_parameters({S_LSTM_INPUT_DIM});
        p_H = model.add_parameters({S_LSTM_INPUT_DIM, S_LSTM_INPUT_DIM});
        p_D = model.add_parameters({S_LSTM_INPUT_DIM, S_LSTM_INPUT_DIM});
        p_R = model.add_parameters({S_LSTM_INPUT_DIM, REL_DIM});

        // guard
        p_action_start = model.add_parameters({ACTION_DIM});
        p_buffer_guard = model.add_parameters({S_LSTM_INPUT_DIM});
        p_stack_guard = model.add_parameters({S_LSTM_INPUT_DIM});
      }
  } // end build model


  static bool IsActionForbidden(const string& a, unsigned bsize, unsigned ssize, const vector<int>& stacki) {
    if (a[1]=='W' && ssize<3) return true;
    if (a[1]=='W') {
          int top=stacki[stacki.size()-1];
          int sec=stacki[stacki.size()-2];
          if (sec>top) return true;
    }

    bool is_shift = (a[0] == 'S' && a[1]=='H');
    bool is_reduce = !is_shift;
    if (is_shift && bsize == 1) return true;
    if (is_reduce && ssize < 3) return true;
    if (bsize == 2 && // ROOT is the only thing remaining on buffer
        ssize > 2 && // there is more than a single element on the stack
        is_shift) return true;
    // only attach left to ROOT
    if (bsize == 1 && ssize == 3 && a[0] == 'R') return true;
    return false;
  }

  // take a vector of actions and return a parse tree (labeling of every
  // word position with its head's position)
  static map<int,int> compute_heads(unsigned sent_len, const vector<unsigned>& actions, const vector<string>& setOfActions, map<int,string>* pr = nullptr) {
    map<int,int> heads;
    map<int,string> r;
    map<int,string>& rels = (pr ? *pr : r);
    for(unsigned i=0;i<sent_len;i++) { heads[i]=-1; rels[i]="ERROR"; }
    vector<int> bufferi(sent_len + 1, 0), stacki(1, -999);
    for (unsigned i = 0; i < sent_len; ++i)
      bufferi[sent_len - i] = i;
    bufferi[0] = -999;
    for (auto action: actions) { // loop over transitions for sentence
      const string& actionString=setOfActions[action];
      const char ac = actionString[0];
      const char ac2 = actionString[1];
      if (ac =='S' && ac2=='H') {  // SHIFT
        assert(bufferi.size() > 1); // dummy symbol means > 1 (not >= 1)
        stacki.push_back(bufferi.back());
        bufferi.pop_back();
      }  else { // LEFT or RIGHT
        assert(stacki.size() > 2); // dummy symbol means > 2 (not >= 2)
        assert(ac == 'L' || ac == 'R');
        unsigned depi = 0, headi = 0;
        (ac == 'R' ? depi : headi) = stacki.back();
        stacki.pop_back();
        (ac == 'R' ? headi : depi) = stacki.back();
        stacki.pop_back();
        stacki.push_back(headi);
        heads[depi] = headi;
        rels[depi] = actionString;
      }
    }
    assert(bufferi.size() == 1);
    //assert(stacki.size() == 2);
    return heads;
  }
  //labeler(&hg, sentence, tsentence, sentenceChar, sentencePos, sentenceDep, sentenceSrl, corpus.intToWords, &right, pred_srl);
  vector<unsigned> labeler(ComputationGraph* hg,
    const vector<unsigned>& raw_sent,
    const vector<unsigned>& sent,
    const vector<vector<unsigned>>& sentChar,
    const vector<unsigned>& sentPos,
    const vector<unsigned>& sentDep,
    const vector<unsigned>& sentSrl,
    const map<unsigned, std::string> & intToWords,
    const vector<unsigned>& correct_actions,
    const vector<string>& setOfActions,
    double *right,
    vector<unsigned>& pred_srl,
    bool training
    //bool use_dep_inloop = false
    ) {
    vector<Expression> log_probs;
    pred_srl.resize(sent.size());
    const bool build_training_graph = training;
    l2rbuilder.new_graph(*hg);
    l2rbuilder.start_new_sequence();
    r2lbuilder.new_graph(*hg);
    r2lbuilder.start_new_sequence();

    Expression w2bil;
    Expression p2bil;
    Expression cf2bil;
    Expression cb2bil;
    Expression d2bil;
    Expression bilbias;
    if (USE_COMBINATION) {
      if (USE_WORD) w2bil = parameter(*hg, p_w2bil);
      if (USE_CHAR) {
        cf2bil = parameter(*hg, p_cf2bil);
        cb2bil = parameter(*hg, p_cb2bil);
      }
      if (USE_POS) p2bil = parameter(*hg, p_p2bil);
      if (USE_DEP) d2bil = parameter(*hg, p_d2bil);
      bilbias = parameter(*hg, p_bilbias);
    }

    Expression i_l2th = parameter(*hg, p_l2th);
    Expression i_r2th = parameter(*hg, p_r2th);
    Expression i_thbias = parameter(*hg, p_thbias);
    Expression i_th2t = parameter(*hg, p_th2t);
    Expression i_tbias = parameter(*hg, p_tbias);

    Expression word_start;
    if (USE_CHAR) word_start = parameter(*hg, p_word_start);
    Expression word_end;
    if (USE_CHAR) word_end = parameter(*hg, p_word_end);

    vector<Expression> errs;

    vector<Expression> i_words(sent.size());
    vector<Expression> fwds(sent.size());
    vector<Expression> revs(sent.size());

    if (USE_CHAR) {
      c_l2rbuilder.new_graph(*hg);
      c_r2lbuilder.new_graph(*hg);
    }

    // read the sentence from left to right
    for (unsigned t=0; t<sent.size(); ++t) {
      vector<Expression> args;
      if (USE_COMBINATION) args.push_back(bilbias);

      
      if (USE_WORD) {
        assert(sent[t] < VOCAB_SIZE);
        Expression word_t = lookup(*hg, p_w, sent[t]);
        if (USE_DROPOUT)
          if (build_training_graph)
            word_t = noise(word_t, 0.2);
        if (USE_COMBINATION) {
          args.push_back(w2bil);
          args.push_back(word_t);
        } else args.push_back(word_t);
        
      }  // end of add word

      if (USE_POS) {
        Expression pos_t = lookup(*hg, p_pos, sentPos[t]);
        if (USE_COMBINATION) {
          args.push_back(p2bil);
          args.push_back(pos_t);
        } else args.push_back(pos_t);
      } // end of add pos

      if (USE_DEP) {
        Expression dep_t = lookup(*hg, p_dep, sentDep[t]);
        if (USE_COMBINATION) {
          args.push_back(d2bil);
          args.push_back(dep_t);
        } else args.push_back(dep_t);
      } // end of add dep

      if (USE_CHAR) {
        c_l2rbuilder.start_new_sequence();
        c_l2rbuilder.add_input(word_start);
        vector<Expression> char_reps(sentChar[t].size());
        int char_index = 0;
        for (auto char_of_word: sentChar[t]) {
          char_reps[char_index] = lookup(*hg, p_ch, char_of_word);
          c_l2rbuilder.add_input(char_reps[char_index]);
          char_index ++;
        }
        c_l2rbuilder.add_input(word_end);

        c_r2lbuilder.start_new_sequence();
        c_r2lbuilder.add_input(word_end);
        for (int j=sentChar[t].size()-1; j>=0; --j)
          c_r2lbuilder.add_input(char_reps[j]);
        c_r2lbuilder.add_input(word_start);

        if (USE_COMBINATION) {
          args.push_back(cf2bil);
          args.push_back(c_l2rbuilder.back());
          args.push_back(cb2bil);
          args.push_back(c_r2lbuilder.back());
        } else {
          args.push_back(c_l2rbuilder.back());
          args.push_back(c_r2lbuilder.back());
        }
      } // end of add char
    
      if (USE_COMBINATION) i_words[t] = rectify(affine_transform(args));
      else i_words[t] = concatenate(args);

      fwds[t] = l2rbuilder.add_input(i_words[t]);

    } // process each token

    // read sequence from right to left
    for (unsigned t=0; t<sent.size(); ++t) 
      revs[sent.size()-t-1] = r2lbuilder.add_input(i_words[sent.size()-t-1]);

    // predict the srl here
    for (unsigned t=0; t<sent.size(); ++t) {
      Expression i_th = rectify(affine_transform({i_thbias, i_l2th, fwds[t], i_r2th, revs[t]}));
      Expression i_t2 = affine_transform({i_tbias, i_th2t, i_th});
      Expression i_t = log_softmax(i_t2);
      // find the best tag
      vector<float> dist = as_vector(hg->incremental_forward());
      double best = -9e99;
      unsigned besti = 0;
      assert(dist.size() == SRL_SIZE);
      for (unsigned i=1; i<dist.size(); ++i) {
        if (dist[i] > best) {
          besti = i;
          best = dist[i];
        }
      }
      pred_srl[t] = besti;
      if (besti == sentSrl[t]) (*right)++;
      Expression i_err = -pick(i_t, sentSrl[t]);
      log_probs.push_back(i_err);
    }

    /* We continue to do parser work here */
    vector<unsigned> results;
    if (JOINT_TRAIN_SRL_DEP) {

      stack_lstm.new_graph(*hg);
      buffer_lstm.new_graph(*hg);
      action_lstm.new_graph(*hg);


      // we do not share char lstm here , leave for further work
      if (USE_CHAR) {
        cs_l2rbuilder.new_graph(*hg);
        cs_r2lbuilder.new_graph(*hg);
      }

      stack_lstm.start_new_sequence();
      buffer_lstm.start_new_sequence();
      action_lstm.start_new_sequence();


      // Input phase
      Expression slbias = parameter(*hg, p_slbias);
      Expression w2sl;
      if (USE_WORD) {
        w2sl = parameter(*hg, p_w2sl);
      }

      Expression cf2sl;
      Expression cb2sl;
      Expression word_start;
      Expression word_end;
      if (USE_CHAR) {
        if (SHARE_CHAR_LSTM) {
          // leave for further code
          word_start = parameter(*hg, p_word_start);
          word_end = parameter(*hg, p_word_end);
        } else {
          word_start = parameter(*hg, p_s_word_start);
          word_end = parameter(*hg, p_s_word_end);
        }
        cf2sl = parameter(*hg, p_cf2sl);
        cb2sl = parameter(*hg, p_cb2sl);
      }

      Expression p2sl;
      if (USE_POS) {
        p2sl = parameter(*hg, p_p2sl);
      }

      Expression t2sl;
      if (USE_PRETRAIN) {
        t2sl = parameter(*hg, p_t2sl);
      }

      Expression srl2sl;
      if (USE_SRL) {
        srl2sl = parameter(*hg, p_srl2sl);
      }

      // Composition parameters
      Expression H = parameter(*hg, p_H);
      Expression D = parameter(*hg, p_D);
      Expression R = parameter(*hg, p_R);
      Expression cbias = parameter(*hg, p_cbias);
      // input -> parser state
      Expression pbias = parameter(*hg, p_pbias);
      Expression S = parameter(*hg, p_S);
      Expression B = parameter(*hg, p_B);
      Expression A = parameter(*hg, p_A);

      // parser state -> action
      Expression p2a = parameter(*hg, p_p2a);
      Expression abias = parameter(*hg, p_abias);
      Expression action_start = parameter(*hg, p_action_start);
      action_lstm.add_input(action_start);

      vector<Expression> buffer(sent.size() + 1);
      vector<int> bufferi(sent.size() + 1);

      for (unsigned i = 0; i < sent.size(); ++i) {
        vector<Expression> args = {slbias};
        // add word
        assert(sent[i] < VOCAB_SIZE);
        if (USE_WORD) {
          Expression w = lookup(*hg, p_w, sent[i]);
          if (USE_DROPOUT)
            if (build_training_graph)
              w = noise(w, 0.2);
          args.push_back(w2sl);
          args.push_back(w);
        }
        // add pos
        if (USE_POS) {
          Expression p = lookup(*hg, p_pos, sentPos[i]);
          args.push_back(p2sl);
          args.push_back(p);
        }
        // add pretrain
        if (p_t && pretrained.count(raw_sent[i])) {
          Expression t = const_lookup(*hg, p_t, raw_sent[i]);
          args.push_back(t2sl);
          args.push_back(t);
        }
        // add srl
        if (USE_SRL) {
          Expression srl = lookup(*hg, p_srl, pred_srl[i]);
          args.push_back(srl2sl);
          args.push_back(srl);
        }
        // add char
        if (USE_CHAR) {
          // add char here
          {
            cs_l2rbuilder.start_new_sequence();
            cs_l2rbuilder.add_input(word_start);
            vector<Expression> char_reps(sentChar[i].size());
            int char_index = 0;
            for (auto char_of_word : sentChar[i]) {
              char_reps[char_index] = lookup(*hg, p_ch, char_of_word);
              cs_l2rbuilder.add_input(char_reps[char_index]);
              char_index ++;
            }
            cs_l2rbuilder.add_input(word_end);

            cs_r2lbuilder.start_new_sequence();
            cs_r2lbuilder.add_input(word_end);
            for (int j=sentChar[i].size()-1; j>=0; --j) {
              cs_r2lbuilder.add_input(char_reps[j]);
            }
            cs_r2lbuilder.add_input(word_start);
          }
          args.push_back(cf2sl);
          args.push_back(cs_l2rbuilder.back());
          args.push_back(cb2sl);
          args.push_back(cs_r2lbuilder.back());
        }

        buffer[sent.size() - i] = rectify(affine_transform(args));
        bufferi[sent.size() - i] = i;

      } // end of adding buffer elements
      
      buffer[0] = parameter(*hg, p_buffer_guard);
      bufferi[0] = -999;
      for (auto & b : buffer)
        buffer_lstm.add_input(b);

      vector<Expression> stack;
      vector<int> stacki;
      stack.push_back(parameter(*hg, p_stack_guard));
      stacki.push_back(-999);
      stack_lstm.add_input(stack.back());

      //vector<Expression> log_probs;
      string rootword;
      unsigned action_count = 0;
      while (stack.size() > 2 || buffer.size() > 1) {
        // get list of possible actions for the current parser state
        vector<unsigned> current_valid_actions;
        for (auto a: possible_actions) {
          if (IsActionForbidden(setOfActions[a], buffer.size(), stack.size(), stacki))
            continue;
          current_valid_actions.push_back(a);
        }
     

        // p_t = pbias + S * slstm + B * blstm + A * alsmt
        Expression p_t = affine_transform({pbias, S, stack_lstm.back(), B, buffer_lstm.back(), A, action_lstm.back()});
        Expression nlp_t = rectify(p_t);
        // r_t = abias + p2a * nlp
        Expression r_t = affine_transform({abias, p2a, nlp_t});

        Expression adiste = log_softmax(r_t, current_valid_actions);
        vector<float> adist = as_vector(hg->incremental_forward());
        double best_score = adist[current_valid_actions[0]];
        unsigned best_a = current_valid_actions[0];
        for (unsigned i = 1; i < current_valid_actions.size(); ++i) {
          if (adist[current_valid_actions[i]] > best_score) {
            best_score = adist[current_valid_actions[i]];
            best_a = current_valid_actions[i];
          }
        }
        unsigned action = best_a;
        if (build_training_graph) {  // if we have reference actions (for training) use the reference action
          action = correct_actions[action_count];
          if (best_a == action) { (*right)++; }
        }
        ++action_count;
        log_probs.push_back(-pick(adiste, action));
        results.push_back(action);

        Expression actione = lookup(*hg, p_a, action);
        action_lstm.add_input(actione);

        Expression relation = lookup(*hg, p_r, action);

        // do action
        const string& actionString=setOfActions[action];
        const char ac = actionString[0];
        const char ac2 = actionString[1];

        if (ac =='S' && ac2=='H') {  // SHIFT
          assert(buffer.size() > 1); // dummy symbol means > 1 (not >= 1)
          stack.push_back(buffer.back());
          stack_lstm.add_input(buffer.back());
          buffer.pop_back();
          buffer_lstm.rewind_one_step();
          stacki.push_back(bufferi.back());
          bufferi.pop_back();
        } else { // LEFT or RIGHT
          assert(stack.size() > 2); // dummy symbol means > 2 (not >= 2)
          assert(ac == 'L' || ac == 'R');
          Expression dep, head;
          unsigned depi = 0, headi = 0;
          (ac == 'R' ? dep : head) = stack.back();
          (ac == 'R' ? depi : headi) = stacki.back();
          stack.pop_back();
          stacki.pop_back();
          (ac == 'R' ? head : dep) = stack.back();
          (ac == 'R' ? headi : depi) = stacki.back();
          stack.pop_back();
          stacki.pop_back();
          if (headi == sent.size() - 1) rootword = intToWords.find(sent[depi])->second;
          // composed = cbias + H * head + D * dep + R * relation
          Expression composed = affine_transform({cbias, H, head, D, dep, R, relation});
          Expression nlcomposed = tanh(composed);
          stack_lstm.rewind_one_step();
          stack_lstm.rewind_one_step();
          stack_lstm.add_input(nlcomposed);
          stack.push_back(nlcomposed);
          stacki.push_back(headi);
        }
      } // end of while
      assert(stack.size() == 2); // guard symbol, root
      assert(stacki.size() == 2);
      assert(buffer.size() == 1); // guard symbol
      assert(bufferi.size() == 1);
      Expression tot_neglogprob = sum(log_probs);
      assert(tot_neglogprob.pg != nullptr);
      return results;
    }

    Expression sum_errs = sum(log_probs);
    return results;


  } // end of labeler


  //===========================================================================================================


  //jointModel.parser(&hg, sentence, tsentence, sentenceChar, sentencePos, sentenceDep, 
  //sentenceSrl, corpus.intToWords, &right, pred_dep, actions, corpus.actions);
  vector<unsigned> parser(ComputationGraph* hg,
      const vector<unsigned>& raw_sent,
      const vector<unsigned>& sent,
      const vector<vector<unsigned>>& sentChar,
      const vector<unsigned>& sentPos,
      const vector<unsigned>& sentDep,
      const vector<unsigned>& sentSrl,
      const map<unsigned, std::string>& intToWords,
      double *right,
      vector<unsigned>& pred_dep,
      vector<unsigned>& pred_srl,
      const vector<unsigned>& correct_actions,
      const vector<string>& setOfActions,
      bool training
    ) {

    vector<Expression> log_probs;
    vector<unsigned> results;
    const bool build_training_graph = training;

    stack_lstm.new_graph(*hg);
    buffer_lstm.new_graph(*hg);
    action_lstm.new_graph(*hg);


    // we do not share char lstm here , leave for further work
    if (USE_CHAR) {
      cs_l2rbuilder.new_graph(*hg);
      cs_r2lbuilder.new_graph(*hg);
    }

    stack_lstm.start_new_sequence();
    buffer_lstm.start_new_sequence();
    action_lstm.start_new_sequence();


    // Input phase
    Expression slbias = parameter(*hg, p_slbias);
    Expression w2sl;
    if (USE_WORD) {
      w2sl = parameter(*hg, p_w2sl);
    }

    Expression cf2sl;
    Expression cb2sl;
    Expression word_start;
    Expression word_end;
    if (USE_CHAR) {
      if (SHARE_CHAR_LSTM) {
        // leave for further code
        word_start = parameter(*hg, p_word_start);
        word_end = parameter(*hg, p_word_end);
      } else {
        word_start = parameter(*hg, p_s_word_start);
        word_end = parameter(*hg, p_s_word_end);
      }
      cf2sl = parameter(*hg, p_cf2sl);
      cb2sl = parameter(*hg, p_cb2sl);
    }

    Expression p2sl;
    if (USE_POS) {
      p2sl = parameter(*hg, p_p2sl);
    }

    Expression t2sl;
    if (USE_PRETRAIN) {
      t2sl = parameter(*hg, p_t2sl);
    }

    Expression srl2sl;
    if (USE_SRL) {
      srl2sl = parameter(*hg, p_srl2sl);
    }

    // Composition parameters
    Expression H = parameter(*hg, p_H);
    Expression D = parameter(*hg, p_D);
    Expression R = parameter(*hg, p_R);
    Expression cbias = parameter(*hg, p_cbias);
    // input -> parser state
    Expression pbias = parameter(*hg, p_pbias);
    Expression S = parameter(*hg, p_S);
    Expression B = parameter(*hg, p_B);
    Expression A = parameter(*hg, p_A);

    // parser state -> action
    Expression p2a = parameter(*hg, p_p2a);
    Expression abias = parameter(*hg, p_abias);
    Expression action_start = parameter(*hg, p_action_start);
    action_lstm.add_input(action_start);

    vector<Expression> buffer(sent.size() + 1);
    vector<int> bufferi(sent.size() + 1);

    for (unsigned i = 0; i < sent.size(); ++i) {
      vector<Expression> args = {slbias};
      // add word
      assert(sent[i] < VOCAB_SIZE);
      if (USE_WORD) {
        Expression w = lookup(*hg, p_w, sent[i]);
        if (USE_DROPOUT)
          if (build_training_graph)
            w = noise(w, 0.2);
        args.push_back(w2sl);
        args.push_back(w);
      }
      // add pos
      if (USE_POS) {
        Expression p = lookup(*hg, p_pos, sentPos[i]);
        args.push_back(p2sl);
        args.push_back(p);
      }
      // add pretrain
      if (p_t && pretrained.count(raw_sent[i])) {
        Expression t = const_lookup(*hg, p_t, raw_sent[i]);
        args.push_back(t2sl);
        args.push_back(t);
      }
      // add srl
      if (USE_SRL) {
        Expression srl = lookup(*hg, p_srl, sentSrl[i]);
        args.push_back(srl2sl);
        args.push_back(srl);
      }
      // add char
      if (USE_CHAR) {
        // add char here
        {
          cs_l2rbuilder.start_new_sequence();
          cs_l2rbuilder.add_input(word_start);
          vector<Expression> char_reps(sentChar[i].size());
          int char_index = 0;
          for (auto char_of_word : sentChar[i]) {
            char_reps[char_index] = lookup(*hg, p_ch, char_of_word);
            cs_l2rbuilder.add_input(char_reps[char_index]);
            char_index ++;
          }
          cs_l2rbuilder.add_input(word_end);

          cs_r2lbuilder.start_new_sequence();
          cs_r2lbuilder.add_input(word_end);
          for (int j=sentChar[i].size()-1; j>=0; --j) {
            cs_r2lbuilder.add_input(char_reps[j]);
          }
          cs_r2lbuilder.add_input(word_start);
        }
        args.push_back(cf2sl);
        args.push_back(cs_l2rbuilder.back());
        args.push_back(cb2sl);
        args.push_back(cs_r2lbuilder.back());
      }

      buffer[sent.size() - i] = rectify(affine_transform(args));
      bufferi[sent.size() - i] = i;

    } // end of adding buffer elements
    
    buffer[0] = parameter(*hg, p_buffer_guard);
    bufferi[0] = -999;
    for (auto & b : buffer)
      buffer_lstm.add_input(b);

    vector<Expression> stack;
    vector<int> stacki;
    stack.push_back(parameter(*hg, p_stack_guard));
    stacki.push_back(-999);
    stack_lstm.add_input(stack.back());

    
    string rootword;
    unsigned action_count = 0;
    while (stack.size() > 2 || buffer.size() > 1) {
      // get list of possible actions for the current parser state
      vector<unsigned> current_valid_actions;
      for (auto a: possible_actions) {
        if (IsActionForbidden(setOfActions[a], buffer.size(), stack.size(), stacki))
          continue;
        current_valid_actions.push_back(a);
      }
   

      // p_t = pbias + S * slstm + B * blstm + A * alsmt
      Expression p_t = affine_transform({pbias, S, stack_lstm.back(), B, buffer_lstm.back(), A, action_lstm.back()});
      Expression nlp_t = rectify(p_t);
      // r_t = abias + p2a * nlp
      Expression r_t = affine_transform({abias, p2a, nlp_t});

      Expression adiste = log_softmax(r_t, current_valid_actions);
      vector<float> adist = as_vector(hg->incremental_forward());
      double best_score = adist[current_valid_actions[0]];
      unsigned best_a = current_valid_actions[0];
      for (unsigned i = 1; i < current_valid_actions.size(); ++i) {
        if (adist[current_valid_actions[i]] > best_score) {
          best_score = adist[current_valid_actions[i]];
          best_a = current_valid_actions[i];
        }
      }
      unsigned action = best_a;
      if (build_training_graph) {  // if we have reference actions (for training) use the reference action
        action = correct_actions[action_count];
        if (best_a == action) { (*right)++; }
      }
      ++action_count;
      log_probs.push_back(-pick(adiste, action));
      results.push_back(action);
      //results.push_back(best_a); // use auto to train
      Expression actione = lookup(*hg, p_a, action);
      action_lstm.add_input(actione);

      Expression relation = lookup(*hg, p_r, action);

      // do action
      const string& actionString=setOfActions[action];
      const char ac = actionString[0];
      const char ac2 = actionString[1];

      if (ac =='S' && ac2=='H') {  // SHIFT
        assert(buffer.size() > 1); // dummy symbol means > 1 (not >= 1)
        stack.push_back(buffer.back());
        stack_lstm.add_input(buffer.back());
        buffer.pop_back();
        buffer_lstm.rewind_one_step();
        stacki.push_back(bufferi.back());
        bufferi.pop_back();
      } else { // LEFT or RIGHT
        assert(stack.size() > 2); // dummy symbol means > 2 (not >= 2)
        assert(ac == 'L' || ac == 'R');
        Expression dep, head;
        unsigned depi = 0, headi = 0;
        (ac == 'R' ? dep : head) = stack.back();
        (ac == 'R' ? depi : headi) = stacki.back();
        stack.pop_back();
        stacki.pop_back();
        (ac == 'R' ? head : dep) = stack.back();
        (ac == 'R' ? headi : depi) = stacki.back();
        stack.pop_back();
        stacki.pop_back();
        if (headi == sent.size() - 1) rootword = intToWords.find(sent[depi])->second;
        // composed = cbias + H * head + D * dep + R * relation
        Expression composed = affine_transform({cbias, H, head, D, dep, R, relation});
        Expression nlcomposed = tanh(composed);
        stack_lstm.rewind_one_step();
        stack_lstm.rewind_one_step();
        stack_lstm.add_input(nlcomposed);
        stack.push_back(nlcomposed);
        stacki.push_back(headi);
      }
    } // end of while
    assert(stack.size() == 2); // guard symbol, root
    assert(stacki.size() == 2);
    assert(buffer.size() == 1); // guard symbol
    assert(bufferi.size() == 1);
    //Expression tot_neglogprob = -sum(log_probs);
    //assert(tot_neglogprob.pg != nullptr);

    // pick up the dep label here
    map<int,std::string> rel_hyp;
    map<int,int> hyp = compute_heads(sent.size(), results, corpus.actions, &rel_hyp);
    for (unsigned i=0; i<sent.size()-1; i++) {
      auto hirel = rel_hyp.find(i);
      assert(hirel != rel_hyp.end());
      std::string dep = hirel->second;
      size_t dash = dep.find('(');
      std::string dep_label = dep.substr(dash+1, dep.size()-dash-2);
      //cerr << dep_label << " ";
      pred_dep.push_back(corpus.depToInt[dep_label]);
    }
    //cerr << endl;
    pred_dep.push_back(corpus.depToInt["ROOT"]);
    /*
    for (auto dep: pred_dep) 
      cerr << corpus.intToDep[dep] << " ";
    cerr << endl;
    */

    
    if (JOINT_TRAIN_DEP_SRL) {
      pred_srl.resize(sent.size());
      
      l2rbuilder.new_graph(*hg);
      l2rbuilder.start_new_sequence();
      r2lbuilder.new_graph(*hg);
      r2lbuilder.start_new_sequence();

      Expression w2bil;
      Expression p2bil;
      Expression cf2bil;
      Expression cb2bil;
      Expression d2bil;
      Expression bilbias;
      if (USE_COMBINATION) {
        if (USE_WORD) w2bil = parameter(*hg, p_w2bil);
        if (USE_CHAR) {
          cf2bil = parameter(*hg, p_cf2bil);
          cb2bil = parameter(*hg, p_cb2bil);
        }
        if (USE_POS) p2bil = parameter(*hg, p_p2bil);
        if (USE_DEP) d2bil = parameter(*hg, p_d2bil);
        bilbias = parameter(*hg, p_bilbias);
      }

      Expression i_l2th = parameter(*hg, p_l2th);
      Expression i_r2th = parameter(*hg, p_r2th);
      Expression i_thbias = parameter(*hg, p_thbias);
      Expression i_th2t = parameter(*hg, p_th2t);
      Expression i_tbias = parameter(*hg, p_tbias);

      Expression word_start = parameter(*hg, p_word_start);
      Expression word_end = parameter(*hg, p_word_end);

      vector<Expression> errs;

      vector<Expression> i_words(sent.size());
      vector<Expression> fwds(sent.size());
      vector<Expression> revs(sent.size());

      if (USE_CHAR) {
        c_l2rbuilder.new_graph(*hg);
        c_r2lbuilder.new_graph(*hg);
      }

      // read the sentence from left to right
      for (unsigned t=0; t<sent.size(); ++t) {
        vector<Expression> args;
        if (USE_COMBINATION) args.push_back(bilbias);

        
        if (USE_WORD) {
          assert(sent[t] < VOCAB_SIZE);
          Expression word_t = lookup(*hg, p_w, sent[t]);
          if (USE_DROPOUT)
            if (build_training_graph)
              word_t = noise(word_t, 0.2);
          if (USE_COMBINATION) {
            args.push_back(w2bil);
            args.push_back(word_t);
          } else args.push_back(word_t);
          
        }  // end of add word

        if (USE_POS) {
          Expression pos_t = lookup(*hg, p_pos, sentPos[t]);
          if (USE_COMBINATION) {
            args.push_back(p2bil);
            args.push_back(pos_t);
          } else args.push_back(pos_t);
        } // end of add pos

        if (USE_DEP) {
          Expression dep_t = lookup(*hg, p_dep, pred_dep[t]);
          if (USE_COMBINATION) {
            args.push_back(d2bil);
	    if (build_training_graph)
		dep_t = noise(dep_t, 0.2);
            args.push_back(dep_t);
          } else args.push_back(dep_t);
        } // end of add dep

        if (USE_CHAR) {
          c_l2rbuilder.start_new_sequence();
          c_l2rbuilder.add_input(word_start);
          vector<Expression> char_reps(sentChar[t].size());
          int char_index = 0;
          for (auto char_of_word: sentChar[t]) {
            char_reps[char_index] = lookup(*hg, p_ch, char_of_word);
            c_l2rbuilder.add_input(char_reps[char_index]);
            char_index ++;
          }
          c_l2rbuilder.add_input(word_end);

          c_r2lbuilder.start_new_sequence();
          c_r2lbuilder.add_input(word_end);
          for (int j=sentChar[t].size()-1; j>=0; --j)
            c_r2lbuilder.add_input(char_reps[j]);
          c_r2lbuilder.add_input(word_start);

          if (USE_COMBINATION) {
            args.push_back(cf2bil);
            args.push_back(c_l2rbuilder.back());
            args.push_back(cb2bil);
            args.push_back(c_r2lbuilder.back());
          } else {
            args.push_back(c_l2rbuilder.back());
            args.push_back(c_r2lbuilder.back());
          }
        } // end of add char
      
        if (USE_COMBINATION) i_words[t] = rectify(affine_transform(args));
        else i_words[t] = concatenate(args);

        fwds[t] = l2rbuilder.add_input(i_words[t]);

      } // process each token

      // read sequence from right to left
      for (unsigned t=0; t<sent.size(); ++t) 
        revs[sent.size()-t-1] = r2lbuilder.add_input(i_words[sent.size()-t-1]);

      // predict the srl here
      for (unsigned t=0; t<sent.size(); ++t) {
        Expression i_th = rectify(affine_transform({i_thbias, i_l2th, fwds[t], i_r2th, revs[t]}));
        Expression i_t = affine_transform({i_tbias, i_th2t, i_th});
        //Expression i_t = log_softmax(i_t2);
        // find the best tag
        vector<float> dist = as_vector(hg->incremental_forward());
        double best = -9e99;
        unsigned besti = 0;
        assert(dist.size() == SRL_SIZE);
        for (unsigned i=1; i<dist.size(); ++i) {
          if (dist[i] > best) {
            besti = i;
            best = dist[i];
          }
        }
        pred_srl[t] = besti;
        if (besti == sentSrl[t]) (*right)++;
        Expression i_err = pickneglogsoftmax(i_t, sentSrl[t]);
        log_probs.push_back(i_err);
      }
    }

    Expression tot_neglogprob = sum(log_probs);
    //assert(tot_neglogprob.pg != nullptr);

    return results;
  } // end of parser





}; // end struct

void signal_callback_handler(int /* signum */) {
  if (requested_stop) {
    cerr << "\nReceived SIGINT again, quitting.\n";
    _exit(1);
  }
  cerr << "\nReceived SIGINT terminating optimization early...\n";
  requested_stop = true;
}

unsigned compute_correct(const map<int,int>& ref, const map<int,int>& hyp, unsigned len) {
  unsigned res = 0;
  for (unsigned i = 0; i < len; ++i) {
    auto ri = ref.find(i);
    auto hi = hyp.find(i);
    assert(ri != ref.end());
    assert(hi != hyp.end());
    if (ri->second == hi->second) ++res;
  }
  return res;
}

unsigned compute_correct_las(const map<int,std::string>& rel_ref, const map<int,std::string> rel_hyp, const map<int,int>& ref, const map<int,int>& hyp, unsigned len) {
  unsigned res = 0;
  for (unsigned i=0; i<len; i++) {
    auto ri = ref.find(i);
    auto hi = hyp.find(i);
    assert(ri != ref.end());
    assert(hi != hyp.end());
    if (ri->second == hi->second) {
      auto rirel = rel_ref.find(i);
      auto hirel = rel_hyp.find(i);
      assert(rirel != rel_ref.end());
      assert(hirel != rel_hyp.end());
      if (rirel->second == hirel->second)
        res ++;
    }
  }
  return res;
}

unsigned compute_correct_for_label(const vector<unsigned>& truth, const vector<unsigned>& pred, unsigned len) {
  unsigned res = 0;
  for (unsigned i=0; i<len; i++)
    if (truth[i] == pred[i] && truth[i] != corpus.srlToInt["FAKE"]) ++res;
  return res;
}

unsigned compute_length_for_label_retrieved(const vector<unsigned>& truth, const vector<unsigned>& pred, unsigned len) {
  unsigned length = 0;
  for (unsigned i = 0; i < len; ++i)
    if (pred[i] != corpus.srlToInt["FAKE"]) ++length;
  return length;
}

unsigned compute_length_for_label_relevant(const vector<unsigned>& truth, const vector<unsigned>& pred, unsigned len) {
  unsigned length = 0;
  for (unsigned i = 0; i < len; ++i) {
    if (truth[i] != corpus.srlToInt["FAKE"]) ++length;
  }
  return length;
}

int main(int argc, char** argv) {
  // We set rand seed == 1 here for comparison
  //cnn::Initialize(argc, argv , 2842580401);
  cnn::Initialize(argc,argv,1);
  cerr << "COMMAND:";
  for (unsigned i = 0; i < static_cast<unsigned>(argc); ++i) cerr << ' ' << argv[i];
  cerr << endl;

  unsigned status_every_i_iterations = 100;

  po::variables_map conf;
  InitCommandLine(argc, argv, &conf);

  USE_WORD = conf.count("use_word");
  if (USE_WORD) 
    cerr << "USE WORD" << endl;
  USE_CHAR = conf.count("use_char");
  if (USE_CHAR)
    cerr << "USE CHAR" << endl;
  USE_POS = conf.count("use_pos");
  if (USE_POS)
    cerr << "USE POS" << endl;
  USE_DEP = conf.count("use_dep");
  if (USE_DEP)
    cerr << "USE DEP" << endl;
  USE_SRL = conf.count("use_srl");
  if (USE_SRL)
    cerr << "USE SRL" << endl;
  USE_COMBINATION = conf.count("use_combination");
  if (USE_COMBINATION)
    cerr << "USE COMBINATION" << endl;
  USE_DROPOUT = conf.count("use_dropout");
  if (USE_DROPOUT)
    cerr << "USE DROPOUT" << endl;
  LOOP_TRAIN = conf.count("loop_train");
  if (LOOP_TRAIN)
    cerr << "LOOP_TRAIN" << endl;
  JOINT_TRAIN_SRL_DEP = conf.count("joint_train_srl_dep");
  if (JOINT_TRAIN_SRL_DEP)
    cerr << "JOINT TRAIN SRL -> DEP" << endl;
  JOINT_TRAIN_DEP_SRL = conf.count("joint_train_dep_srl");
  if (JOINT_TRAIN_DEP_SRL)
    cerr << "JOINT TRAIN DEP -> SRL" << endl;
  
  LABELER = conf.count("labeler");
  if (LABELER) {
    LABELER_TIME = true;
  }
  PARSER = conf.count("parser");
  if (PARSER) {
    PARSER_TIME = true;
  }
  if (LOOP_TRAIN) {
    LABELER = true;
    PARSER = true;
  }
  if (JOINT_TRAIN_SRL_DEP) {
    LABELER = true;
    PARSER = true;
  } 
  if (JOINT_TRAIN_DEP_SRL) {
    LABELER = true;
    PARSER = true;
  }
  //USE_PRETRAIN = 

  

  BiLSTMLAYERS = conf["bilstm_layers"].as<unsigned>();
  STACKLSTMLAYERS = conf["stack_lstm_layers"].as<unsigned>();
  if (USE_WORD) WORD_DIM = conf["word_dim"].as<unsigned>();
  if (USE_CHAR) {
    CHAR_DIM = conf["char_dim"].as<unsigned>();
    CHAR_HIDDEN_DIM = conf["char_hidden_dim"].as<unsigned>();
  }
  if (USE_POS) POS_DIM = conf["pos_dim"].as<unsigned>();
  if (USE_DEP) DEP_DIM = conf["dep_dim"].as<unsigned>();
  if (USE_SRL) SRL_DIM = conf["srl_dim"].as<unsigned>();
  if (USE_PRETRAIN) PRETRAINED_DIM = conf["pretrained_dim"].as<unsigned>();

  if (LABELER) {
    B_LSTM_INPUT_DIM = conf["b_lstm_input_dim"].as<unsigned>();
    B_HIDDEN_DIM = conf["b_hidden_dim"].as<unsigned>();
    TAG_HIDDEN_DIM = conf["tag_hidden_dim"].as<unsigned>();
  }

  if (PARSER) {
    S_LSTM_INPUT_DIM = conf["s_lstm_input_dim"].as<unsigned>();
    S_HIDDEN_DIM = conf["s_hidden_dim"].as<unsigned>();
    ACTION_DIM = conf["action_dim"].as<unsigned>();
    REL_DIM = conf["rel_dim"].as<unsigned>();
  }



  const unsigned unk_strategy = conf["unk_strategy"].as<unsigned>();
  cerr << "Unknown word strategy: ";
  if (unk_strategy == 0) {
    cerr << "ABSOLUTE REPLACEMENT\n";
  } else if (unk_strategy == 1) {
    cerr << "STOCHASTIC REPLACEMENT\n";
  } else {
    abort();
  }
  const double unk_prob = conf["unk_prob"].as<double>();
  assert(unk_prob >= 0.); assert(unk_prob <= 1.);
  ostringstream os;
  os << "model-" << "pid" << getpid() << ".params";

  

  double best_f_dev = 0;
  double best_p_dev = 0;
  double best_r_dev = 0;
  double best_uas_dev = 0;
  double best_las_dev = 0;
  double best_eva_score = 0;

  const string fname = os.str();
  cerr << "Writing parameters to file: " << fname << endl;

  corpus.load_correct_actions(conf["training_data"].as<string>());
  const unsigned kUNK = corpus.wordsToInt[cpyp::Corpus::UNK];
  //const unsigned kCUNK = corpus.charsToInt[cpyp::Corpus::CUNK];

  // Add pretrain word-embedding
  if (conf.count("words")) {
    pretrained[kUNK] = vector<float>(PRETRAINED_DIM, 0);
    cerr << "Loading from " << conf["words"].as<string>() << " with" << PRETRAINED_DIM << " dimensions\n";
    ifstream in(conf["words"].as<string>().c_str());
    string line;
    getline(in, line);
    vector<float> v(PRETRAINED_DIM, 0);
    string word;
    while (getline(in, line)) {
      istringstream lin(line);
      lin >> word;
      for (unsigned i = 0; i < PRETRAINED_DIM; ++i) lin >> v[i];
      unsigned id = corpus.get_or_add_word(word);
      pretrained[id] = v;
    }
  }

  set<unsigned> training_vocab; // words available in the training corpus
  set<unsigned> singletons;
  {  // compute the singletons in the parser's training data
    map<unsigned, unsigned> counts;
    for (auto sent : corpus.sentences)
      for (auto word : sent.second) { training_vocab.insert(word); counts[word]++; }
    for (auto wc : counts)
      if (wc.second == 1) singletons.insert(wc.first);
  }

  
  VOCAB_SIZE = corpus.nwords + 1;
  CHAR_SIZE = corpus.nchars + 1;
  POS_SIZE = corpus.npos + 1;
  DEP_SIZE = corpus.ndeps + 1;
  SRL_SIZE = corpus.nsrls + 1;
  ACTION_SIZE = corpus.nactions + 1;

  // Output all the parameters here
  cerr << "WORD_DIM: " <<WORD_DIM << endl;
  cerr << "CHAR_DIM: " <<CHAR_DIM << endl;
  cerr << "CHAR_HIDDEN_DIM: "<< CHAR_HIDDEN_DIM << endl;
  cerr << "POS_DIM: " << POS_DIM << endl;
  cerr << "DEP_DIM: " << DEP_DIM << endl;
  cerr << "SRL_DIM: " << SRL_DIM << endl;
  cerr << "ACTION_DIM: " << ACTION_DIM << endl;
  cerr << "REL_DIM: " << REL_DIM << endl; 
  cerr << "B_LSTM_INPUT_DIM: "<< B_LSTM_INPUT_DIM<< endl;
  cerr << "B_HIDDEN_DIM: "<< B_HIDDEN_DIM << endl;
  cerr << "S_LSTM_INPUT_DIM: " << S_LSTM_INPUT_DIM << endl;
  cerr << "S_HIDDEN_DIM: " << S_HIDDEN_DIM << endl; 


  possible_actions.resize(corpus.nactions);
  for (unsigned i = 0; i < corpus.nactions; ++i)
    possible_actions[i] = i;


  // If we can select some SRL for prediction, the result may be better
  possible_label.resize(SRL_SIZE);
  for (unsigned i = 0; i < SRL_SIZE; ++i)
    possible_label[i] = i;


  Model model;
  JointModel jointModel(model, pretrained);
  if (conf.count("model")) {
    ifstream in(conf["model"].as<string>().c_str());
    boost::archive::text_iarchive ia(in);
    ia >> model;
  }

  // OOV words will be replaced by UNK tokens
  
    corpus.load_correct_actionsDev(conf["dev_data"].as<string>());
  if (conf.count("test"))
    corpus.load_correct_actionsTest(conf["test_data"].as<string>());

  //Training
  if (conf.count("train")) {
    signal(SIGINT, signal_callback_handler);
    MomentumSGDTrainer sgd(&model);
    //SimpleSGDTrainer sgd(&model);
    sgd.eta_decay = 0.08;
    vector<unsigned> order(corpus.nsentencesTrain);
    for (unsigned i = 0; i < corpus.nsentencesTrain; ++i)
      order[i] = i;
    // save the epoch for learning loops
    double tot_seen = 0;
    double epoch = 0;
    status_every_i_iterations = min(status_every_i_iterations, corpus.nsentencesTrain);
    unsigned si = corpus.nsentencesTrain;
    cerr << "NUMBER OF TRAINING SENTENCES: " << corpus.nsentencesTrain << endl;
    unsigned trs = 0;
    double right = 0;
    double llh = 0;
    bool first = true;
    int iter = -1;

    while(!requested_stop) {
      ++iter;
      for (unsigned sii = 0; sii < status_every_i_iterations; ++sii) {
        if (si == corpus.nsentencesTrain) {
          si = 0;
          if (first) { first = false; } else { sgd.update_epoch(); }
          cerr << "**SHUFFLE\n";
          random_shuffle(order.begin(), order.end());
        }
        tot_seen += 1;
        epoch = tot_seen / corpus.nsentencesTrain;
        if (LOOP_TRAIN) {
          if (int(epoch) / 10 % 2 == 0) {
              LABELER_TIME = true;
              PARSER_TIME = false;
            } else {
              //USE_DEP_INLOOP = true;
              LABELER_TIME = false;
              PARSER_TIME = true;
          }
        } // end setting of loop train

        if (JOINT_TRAIN_SRL_DEP) {
          LABELER_TIME = true;
          PARSER_TIME = false;
        }

        if (JOINT_TRAIN_DEP_SRL) {
          LABELER_TIME = false;
          PARSER_TIME = true;
        }
        
        const vector<unsigned>& sentence=corpus.sentences[order[si]];
        vector<unsigned> tsentence=sentence;
        if (unk_strategy == 1) {
          for (auto& w : tsentence)
            if (singletons.count(w) && cnn::rand01() < unk_prob) w = kUNK;
        } else if (unk_strategy == 0) {
          for (auto& w : tsentence)
            if (singletons.count(w)) w = kUNK;
        } // end of UNK strategy process

        const vector<unsigned>& sentencePos=corpus.sentencesPos[order[si]];
        const vector<unsigned>& actions=corpus.correct_act_sent[order[si]];
        const vector<unsigned>& sentenceDep=corpus.sentencesDep[order[si]];
        const vector<unsigned>& sentenceSrl=corpus.sentencesSrl[order[si]];
        const vector<vector<unsigned>>& sentenceChar=corpus.sentencesChar[order[si]];
        if (LABELER_TIME) {
          ComputationGraph hg;
          vector<unsigned> pred_srl;
          jointModel.labeler(&hg, sentence, tsentence, sentenceChar, sentencePos, sentenceDep, sentenceSrl, corpus.intToWords, actions, corpus.actions, &right, pred_srl, true);
          corpus.sentencesPredSrl[order[si]] = pred_srl;
          double lp = as_scalar(hg.incremental_forward());
          /*
          if (lp < 0) {
            cerr << "Log prob < 0 on sentence " << order[si] << ": lp = " << lp << endl;
            assert(lp >= 0.0);
          }
          */
          hg.backward();
          sgd.update(1.0);
          llh += lp;
          ++si;
          trs += sentence.size() - 1;
          if (JOINT_TRAIN_SRL_DEP) trs += actions.size() - 1;
        } // end of labeler

        if (PARSER_TIME) {
          ComputationGraph hg;
          vector<unsigned> pred_dep;
          vector<unsigned> pred_srl;
          jointModel.parser(&hg, sentence, tsentence, sentenceChar, sentencePos, sentenceDep, sentenceSrl, corpus.intToWords, &right, pred_dep, pred_srl, actions, corpus.actions, true);
          corpus.sentencesPredDep[order[si]] = pred_dep;
          double lp = as_scalar(hg.incremental_forward());
          if (lp < 0) {
            cerr << "Log prob < 0 on sentence " << order[si] << ": lp = " << lp << endl;
            assert(lp >= 0.0);
          }
          hg.backward();
          sgd.update(1.0);
          llh += lp;
          ++si;
          trs += actions.size() - 1;
          if (JOINT_TRAIN_DEP_SRL) trs += sentence.size() - 1;
        } // end of parser

      } // end a iter, and then report the sgd status. 

      sgd.status();
      if (sgd.eta<=0.001) {
	sgd.eta_decay = 0;
	sgd.eta0 = 0.001;
      }
      cerr << "update #" << iter << " (epoch " << (tot_seen / corpus.nsentencesTrain) << ")\tllh: " << llh <<" ppl: " << exp(llh/trs) << " err: "<< (trs-right) / trs << endl;
      llh = trs = right = 0;

      static int logc = 0;
      ++logc;
      // report on dev set
      if (logc % 10 == 1) {
        unsigned dev_size = corpus.nsentencesDev;
        //cerr << dev_size<< endl;
        double right = 0;
        double correct_heads = 0;
        double total_heads = 0;
        double las_correct = 0;
        double correct_label_num = 0;
        double retrieved_label_num = 0;
        double relevant_label_num = 0;
        for (unsigned sii = 0; sii < dev_size; ++sii) {
          const vector<unsigned>& sentence=corpus.sentencesDev[sii];
          const vector<unsigned>& actions=corpus.correct_act_sentDev[sii];
          const vector<vector<unsigned>>& sentenceChar=corpus.sentencesCharDev[sii];
          const vector<unsigned>& sentencePos=corpus.sentencesPosDev[sii]; 
          const vector<unsigned>& sentenceDep=corpus.sentencesDepDev[sii]; 
          const vector<unsigned>& sentenceSrl=corpus.sentencesSrlDev[sii];
          vector<unsigned> tsentence=sentence;
          for (auto& w : tsentence)
            if (training_vocab.count(w) == 0) w = kUNK;
          //cerr << sii << " ";

          if (LABELER_TIME) {
            ComputationGraph hg;
            vector<unsigned> pred_srl;
            vector<unsigned> pred;
            pred = jointModel.labeler(&hg, sentence, tsentence, sentenceChar, sentencePos, sentenceDep, sentenceSrl, corpus.intToWords, actions, corpus.actions, &right, pred_srl, false);
            /*
            for (auto srl : pred_srl) {
              cerr << corpus.intToSrl[srl] << " ";
            }
            cerr << endl;
            */
            correct_label_num += compute_correct_for_label(sentenceSrl, pred_srl, sentence.size()-1);
            retrieved_label_num += compute_length_for_label_retrieved(sentenceSrl, pred_srl, sentence.size()-1);
            relevant_label_num += compute_length_for_label_relevant(sentenceSrl, pred_srl, sentence.size()-1);
            if (JOINT_TRAIN_SRL_DEP) {
              map<int, string> rel_ref, rel_hyp;
              map<int,int> ref = jointModel.compute_heads(sentence.size(), actions, corpus.actions, &rel_ref);
              map<int,int> hyp = jointModel.compute_heads(sentence.size(), pred, corpus.actions, &rel_hyp);
              correct_heads += compute_correct(ref, hyp, sentence.size() - 1);
              las_correct += compute_correct_las(rel_ref, rel_hyp, ref, hyp, sentence.size()-1);
              total_heads += sentence.size() - 1;
            }
          }
          
          if (PARSER_TIME) {
            ComputationGraph hg;
            vector<unsigned> pred;
            vector<unsigned> pred_dep;
            vector<unsigned> pred_srl;
            pred = jointModel.parser(&hg, sentence, tsentence, sentenceChar, sentencePos, sentenceDep, sentenceSrl, corpus.intToWords, &right, pred_dep, pred_srl, actions, corpus.actions, false);
            map<int, string> rel_ref, rel_hyp;
            map<int,int> ref = jointModel.compute_heads(sentence.size(), actions, corpus.actions, &rel_ref);
            map<int,int> hyp = jointModel.compute_heads(sentence.size(), pred, corpus.actions, &rel_hyp);
            correct_heads += compute_correct(ref, hyp, sentence.size() - 1);
            las_correct += compute_correct_las(rel_ref, rel_hyp, ref, hyp, sentence.size()-1);
            total_heads += sentence.size() - 1;
            if (JOINT_TRAIN_DEP_SRL) {
              correct_label_num += compute_correct_for_label(sentenceSrl, pred_srl, sentence.size()-1);
              retrieved_label_num += compute_length_for_label_retrieved(sentenceSrl, pred_srl, sentence.size()-1);
              relevant_label_num += compute_length_for_label_relevant(sentenceSrl, pred_srl, sentence.size()-1);
            }
          }
          


        }
        double uas = 0;
        double las = 0;
        double p = 0;
        double r = 0;
        double f = 0;
        cerr << " ** Dev (iter=" << iter << " epoch=" <<(tot_seen / corpus.nsentencesTrain) << ")" << endl;
        if (PARSER_TIME) {
          uas = correct_heads / total_heads;
          las = las_correct / total_heads;
          cerr << " ** uas = " << uas << " las = "<< las <<endl;
          if (JOINT_TRAIN_DEP_SRL) {
            p = correct_label_num / retrieved_label_num;
            r = correct_label_num / relevant_label_num;
            f = 2 * p * r / (p + r);
            cerr << " ** P = " << p <<"  R = "<< r << "  F = " << f << endl;
          }
        }
        if (LABELER_TIME) {
          p = correct_label_num / retrieved_label_num;
          r = correct_label_num / relevant_label_num;
          f = 2 * p * r / (p + r);
          cerr << " ** P = " << p <<"  R = "<< r << "  F = " << f << endl;
          if (JOINT_TRAIN_SRL_DEP) {
            uas = correct_heads / total_heads;
            las = las_correct / total_heads;
            cerr << " ** uas = " << uas << " las = "<< las <<endl;
          }
        }
        
        
        
        
        //if (f > best_f_dev || uas > best_uas_dev || las > best_las_dev) {
        double eva_score = 0;
        if (LABELER_TIME) eva_score = f;
        if (PARSER_TIME) eva_score = 0.5 * uas + 0.5 * las;
        if (LABELER_TIME && PARSER_TIME) eva_score = 0.5 * f + 0.25 * uas + 0.25 * las;
        if (JOINT_TRAIN_SRL_DEP | JOINT_TRAIN_DEP_SRL) eva_score = 0.5 * f + 0.25 * uas + 0.25 * las;
        if (eva_score > best_eva_score) {
          best_eva_score = eva_score;
          if (LABELER_TIME | JOINT_TRAIN_DEP_SRL) {
            best_f_dev = f;
            best_p_dev = p;
            best_r_dev = r;
          }
          if (PARSER_TIME | JOINT_TRAIN_SRL_DEP) {
            best_uas_dev = uas;
            best_las_dev = las;
          }
	        ofstream out(fname);
          boost::archive::text_oarchive oa(out);
          oa << model;
          



        } // select better result

        // Report the best result here
        cerr << " For now, one of the best result is: " << endl;
        cerr << " Dev : "<< endl;
        if (LABELER)
         cerr << " P = " << best_p_dev << " R = " << best_r_dev << " F = " << best_f_dev << endl;
        if (PARSER)
          cerr << " UAS = " << best_uas_dev << " LAS = " << best_las_dev << endl;
        
      } // end on the dev evaluation

      


    } // you can get out this infinit loop using ctrl+C
  } // end train


  if (conf.count("test")) {
    // Do not support loop train here
      if (JOINT_TRAIN_SRL_DEP) {
        LABELER_TIME = true;
        PARSER_TIME = false;
      }

      if (JOINT_TRAIN_DEP_SRL) {
        LABELER_TIME = false;
        PARSER_TIME = true;
      }
      unsigned test_size = corpus.nsentencesTest;
      //cerr << dev_size<< endl;
      double right = 0;
      double correct_heads = 0;
      double total_heads = 0;
      double las_correct = 0;
      double correct_label_num = 0;
      double retrieved_label_num = 0;
      double relevant_label_num = 0;
      for (unsigned sii = 0; sii < test_size; ++sii) {
        const vector<unsigned>& sentence=corpus.sentencesTest[sii];
        const vector<unsigned>& actions=corpus.correct_act_sentTest[sii];
        const vector<vector<unsigned>>& sentenceChar=corpus.sentencesCharTest[sii];
        const vector<unsigned>& sentencePos=corpus.sentencesPosTest[sii]; 
        const vector<unsigned>& sentenceDep=corpus.sentencesDepTest[sii]; 
        const vector<unsigned>& sentenceSrl=corpus.sentencesSrlTest[sii];
        vector<unsigned> tsentence=sentence;
        for (auto& w : tsentence)
          if (training_vocab.count(w) == 0) w = kUNK;
        //cerr << sii << " ";
        if (LABELER_TIME) {
          ComputationGraph hg;
          vector<unsigned> pred_srl;
          vector<unsigned> pred;
          pred = jointModel.labeler(&hg, sentence, tsentence, sentenceChar, sentencePos, sentenceDep, sentenceSrl, corpus.intToWords, actions, corpus.actions, &right, pred_srl, false);
          correct_label_num += compute_correct_for_label(sentenceSrl, pred_srl, sentence.size()-1);
          retrieved_label_num += compute_length_for_label_retrieved(sentenceSrl, pred_srl, sentence.size()-1);
          relevant_label_num += compute_length_for_label_relevant(sentenceSrl, pred_srl, sentence.size()-1);
          if (JOINT_TRAIN_SRL_DEP) {
            map<int, string> rel_ref, rel_hyp;
            map<int,int> ref = jointModel.compute_heads(sentence.size(), actions, corpus.actions, &rel_ref);
            map<int,int> hyp = jointModel.compute_heads(sentence.size(), pred, corpus.actions, &rel_hyp);
            correct_heads += compute_correct(ref, hyp, sentence.size() - 1);
            las_correct += compute_correct_las(rel_ref, rel_hyp, ref, hyp, sentence.size()-1);
            total_heads += sentence.size() - 1;
          }
          
        }
        
        if (PARSER_TIME) {
          ComputationGraph hg;
          vector<unsigned> pred;
          vector<unsigned> pred_dep;
          vector<unsigned> pred_srl;
          pred = jointModel.parser(&hg, sentence, tsentence, sentenceChar, sentencePos, sentenceDep, sentenceSrl, corpus.intToWords, &right, pred_dep, pred_srl, actions, corpus.actions, false);
          map<int, string> rel_ref, rel_hyp;
          map<int,int> ref = jointModel.compute_heads(sentence.size(), actions, corpus.actions, &rel_ref);
          map<int,int> hyp = jointModel.compute_heads(sentence.size(), pred, corpus.actions, &rel_hyp);
          correct_heads += compute_correct(ref, hyp, sentence.size() - 1);
          las_correct += compute_correct_las(rel_ref, rel_hyp, ref, hyp, sentence.size()-1);
          total_heads += sentence.size() - 1;
          if (JOINT_TRAIN_DEP_SRL) {
            correct_label_num += compute_correct_for_label(sentenceSrl, pred_srl, sentence.size()-1);
            retrieved_label_num += compute_length_for_label_retrieved(sentenceSrl, pred_srl, sentence.size()-1);
            relevant_label_num += compute_length_for_label_relevant(sentenceSrl, pred_srl, sentence.size()-1);
          }
        }
        


      }
      double uas = 0;
      double las = 0;
      double p = 0;
      double r = 0;
      double f = 0;
      cerr << " ** Test" << endl;
      if (PARSER_TIME) {
        uas = correct_heads / total_heads;
        las = las_correct / total_heads;
        cerr << " ** uas = " << uas << " las = "<< las <<endl;
        if (JOINT_TRAIN_DEP_SRL) {
          p = correct_label_num / retrieved_label_num;
          r = correct_label_num / relevant_label_num;
          f = 2 * p * r / (p + r);
          cerr << " ** P = " << p <<"  R = "<< r << "  F = " << f << endl;
        }
      }
      if (LABELER_TIME) {
        p = correct_label_num / retrieved_label_num;
        r = correct_label_num / relevant_label_num;
        f = 2 * p * r / (p + r);
        cerr << " ** P = " << p <<"  R = "<< r << "  F = " << f << endl;
        if (JOINT_TRAIN_SRL_DEP) {
          uas = correct_heads / total_heads;
          las = las_correct / total_heads;
          cerr << " ** uas = " << uas << " las = "<< las <<endl;
        }
      }
      


    } // end of test








}
