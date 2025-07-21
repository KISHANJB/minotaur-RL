//
//    Minotaur -- It's only 1/2 bull
//
//    (C)opyright 2008 - 2025 The Minotaur Team.
//

/**
 * \file QLBrancher.cpp
 * \brief Define methods for Q-Learning branching.
 * \author KISHAN, IEOR IIT Delhi
 */

#include <algorithm>
#include <cmath>
#include <omp.h>
#include <random>
#include <typeinfo>
#include "BrCand.h"
#include "BrVarCand.h"
#include "Branch.h"
#include "Engine.h"
#include "Environment.h"
#include "Handler.h"
#include "Logger.h"
#include "MinotaurConfig.h"
#include "Modification.h"
#include "Node.h"
#include "Option.h"
#include "ProblemSize.h"
#include "Relaxation.h"
#include "QLBrancher.h"
#include "Solution.h"
#include "SolutionPool.h"
#include "Timer.h"
#include "Variable.h"
#include "BranchAndBound.h" 
#include "NodeRelaxer.h"
#include <vector>
#include <unordered_map>       // For std::map
#include <iterator>  // Optional, for std::begin and std::end
#include <iostream>
#include <functional>
#include <string>
#include <limits> 
#include <iomanip>  // for std::setprecision
//#define SPEW 1


using namespace Minotaur;
const std::string QLBrancher::me_ = "QL brancher: ";

// --- Q-Learning Parameters ---
const int EPISODES = 100;
const double ALPHA = 0.1;     // Learning rate
const double GAMMA = 0.95;    // Discount factor
const double EPSILON = 0.2;   // Exploration rate
static constexpr double DEFAULT_Q = 0.0; // <-- Default value here
//std::vector<double> state;
//std::vector<double> current_state;
//std::vector<double> prev_state;
Minotaur::BrCandPtr  prev_best_cand, best_cand, action,root_best_cand;         // Using prev_best_cand to store the action taken for the previous state or at the parent node. 

//QTable QLBrancher::Q1;

//Q-Table  insert function
/**** void insert_qvalue(QTable& qtable, std::vector<double>& s, BrCandPtr a, double q_value)
 {
	 qtable[s][a] = q_value;
 };
***/

QLBrancher::QLBrancher(EnvPtr env, HandlerVector& handlers)
  : //qtable_(),
    engine_(EnginePtr()), // NULL
    eTol_(1e-6),
    handlers_(handlers), // Create a copy, the vector is not too big
    init_(false),
    maxDepth_(1000),
    maxIterations_(25),
    maxStrongCands_(20),
    minNodeDist_(50),
    rel_(RelaxationPtr()), // NULL
    status_(NotModifiedByBrancher),
    thresh_(4),
    trustCutoff_(true),
    x_(0)
   // current_state = std::vector<double>{};
   // prev_state = std::vector<double>{};
   // prev_best_cand=0;
   // best_cand=0;
   // action=0;
    
{ 
  timer_ = env->getNewTimer();
  logger_ = env->getLogger();
  stats_ = new QLBrStats();
  stats_->calls = 0;
  stats_->engProbs = 0;
  stats_->strBrCalls = 0;
  stats_->bndChange = 0;
  stats_->iters = 0;
  stats_->strTime = 0.0;
}



QLBrancher::~QLBrancher()
{
  delete stats_;
  delete timer_;
}

// Q-Table  access function
/*double get_qvalue(QTable& qtable, std::vector<double>& s, BrCandPtr a) {
  //       static constexpr double DEFAULT_Q = 0.0; // <-- Default value here
         auto s_entry = qtable.find(s);
         if (s_entry == qtable.end()) return DEFAULT_Q;
         auto a_entry = s_entry->second.find(a);
         return (a_entry != s_entry->second.end()) ? a_entry->second : DEFAULT_Q;
};*/

//QTable getQTable(){
//	return qtable_;
// }




void QLBrancher::printQTable() {
	std::cout << "======= Q-Table  =======" << std::endl;
	for (const auto& entry : qtable_) {
		std::cout << "State: [ ";
		for (double s : entry.first) {
			std::cout << std::fixed << std::setprecision(2) << s << " ";
		}
		std::cout << "]\n";
		for (const auto& subentry : entry.second) {
			if (subentry.first != nullptr){
                        BrVarCandPtr vc = dynamic_cast<BrVarCand*>(subentry.first);
			std::cout << "  Action: " << vc->getName()
				<< " => Q-Value: " << std::fixed << std::setprecision(4)
				<< subentry.second << std::endl;
			}
			else {
				std::cout << "  Action: NULL " << std::endl; 
			}
		std::cout << "------------------------" << std::endl;
	}
	std::cout << "======= End of Q-Table =======" << std::endl;
	}

}





// Print the Q-tabl
/*void QLBrancher::printQTable() {
	for (const auto& [state, actionMap] : qtable_) {
                  std::cout << "State: [ ";
                  for (double s : state) {
                          std::cout << std::fixed << std::setprecision(2) << s << " ";
                  }
                  std::cout << "]\n";
                  for (const auto& [action, q_value] : actionMap) {
                          std::cout << "  Action: " << action->getName()                 //name
                                  << " => Q-Value: " << std::fixed << std::setprecision(4) << q_value
                                  << std::endl;
                  }
                  std::cout << "------------------------" << std::endl;
          }
          std::cout << "======= End of Q-Table =======" << std::endl;
}

*/


BrCandPtr QLBrancher::findBestCandidate_(const double objval,
                                                  double cutoff, NodePtr node)
{
  double best_score = -INFINITY;
  double score, change_up, change_down, maxchange;
  UInt cnt, maxcnt;
  UInt depth = node->getDepth();
  EngineStatus status_up, status_down;
  BrCandPtr cand;
  std::vector<double> upperBounds;
  static std::vector<double> prev_state={},root_state={};
  static std::vector<double> current_state={};
  // why static here?
  static BrCandPtr best_cand = 0;
  //BrCandPtr cand, action, best_cand = 0;
  prev_state=current_state;
  prev_best_cand = best_cand;
  //BrCandPtr  prev_best_cand, best_cand, action;         // Using prev_best_cand to store the action taken for the previous state or at the parent node. 
  decltype(relCands_) var = relCands_;
  var.insert(var.end(),unrelCands_.begin(), unrelCands_.end());  // Automatically avoids duplicates


  for(BrCandVIter it = var.begin(); it != var.end(); ++it) {
                BrVarCandPtr varCand = dynamic_cast<BrVarCand*>(*it);
                double  lb = varCand->getVar()->getLb();
                double  ub = varCand->getVar()->getUb();
                current_state.push_back(lb);          // Collect LB
                upperBounds.push_back(ub);          // Collect UB
  }
  current_state.insert(current_state.end(), upperBounds.begin(), upperBounds.end());//Finally getting the current state vector----
   if (node->getDepth() == 0){
	   root_state = current_state;
   }
   if (node->getParent()->getDepth() == 0){
	   prev_state = root_state;
	   prev_best_cand = root_best_cand;
   }


   /*if (current_state == prev_state){
	   prev_state = root_state;
	   prev_best_cand = root_best_cand;
   }*/

     /** -------------------------------------------------------------- RL Framework starts here------------------------------------------------------------------------------*/

  if((node->getDepth())>0) {
	/* if (node->getParent()->getDepth() == 0){
           prev_state = root_state;
           prev_best_cand = root_best_cand;
	 }*/
         double current_lb,prev_lb,delta_lb;
    //---- Fetching lower bounds on the optimal solution for calculating reward----
         current_lb= node->getLb();
         prev_lb=node->getParent()->getLb();
         delta_lb= current_lb-prev_lb;
   
   //---- Q-value updates--------
         double maxNextQ = 0.0;  // Since nextState is always unseen, assume max Q-value is 0
         double qsa = 0;
	 std::string prev_action;
        // if (!prev_state.empty() && prev_best_cand) {
         	auto s_entry = qtable_.find(prev_state);
         	qsa = DEFAULT_Q;
         	if (s_entry != qtable_.end()) {
         		auto a_entry = s_entry->second.find(prev_best_cand);
         		qsa = (a_entry != s_entry->second.end()) ? a_entry->second : DEFAULT_Q;
         	}
         	qsa += ALPHA * (delta_lb + GAMMA * maxNextQ - qsa);
		//prev_action = prev_best_cand->getName();
	//	qtable_[prev_state][prev_action] = qsa;
	        //std::cout << "BrCand Pointer" << prev_best_cand << std::endl;	
	        //std::cout << "Variable Name" << prev_action << std::endl;
		//std::cout << "variable Pointer" << &(prev_best_cand->getName()) <<std::endl;
         	qtable_[prev_state][prev_best_cand] = qsa;
        // }


   // -------Epsilon-greedy action selection----
         if (var.empty()){
         	std::cout<<"No Actions Possible";
         	//return nullptr;
         }
	 std::random_device rd;
         std::mt19937 gen(rd());
         std::uniform_real_distribution<double> prob_dist(0.0, 1.0);

         if (prob_dist(gen) < EPSILON) {
    // Exploration: pick a random action
                std::uniform_int_distribution<size_t> action_dist(0, var.size() - 1);
                best_cand = var[action_dist(gen)];
		//return best_cand;
	 } else {
    // Exploitation: pick the best action (with random tie-breaking)
                double max_q = -std::numeric_limits<double>::infinity();
                std::vector<decltype(var)::value_type> best_actions;

                for (const auto& action : var) {
			double q = DEFAULT_Q;
                        auto st_entry = qtable_.find(current_state);
                        if (st_entry != qtable_.end()) {
				auto ac_entry = st_entry->second.find(action);
                                if (ac_entry != st_entry->second.end()) {
					q = ac_entry->second;
				}
			}
			if (q > max_q) {
				max_q = q;
                                best_actions = {action}; 
			}
			else if (q == max_q) {
				best_actions.push_back(action); 
			}
		}
		// Break ties randomly among best actions
		std::uniform_int_distribution<size_t> tie_dist(0, best_actions.size() - 1);
		best_cand = best_actions[tie_dist(gen)];
		
	 }
	 std::cout << "State: [ ";
                for (double s : prev_state) {
                        std::cout << std::fixed << std::setprecision(2) << s << " ";
                }
                std::cout << "]\n";
        /*BrVarCandPtr vc = dynamic_cast<BrVarCand*>(prev_best_cand);
	if (vc) {
		std::cout << "  Action: " << vc->getName()
		          << " => Q-Value: " << std::fixed << std::setprecision(4) << qsa
                                  << std::endl;
	} else {
		std::cout << "  Action: " << prev_best_cand->getName()
	                  << " => Q-Value: " << std::fixed << std::setprecision(4) << qsa
                                  << std::endl;	
	}*/
	std::cout << "  Action: " << prev_best_cand->getName() 
	          << " => Q-Value: " << std::fixed << std::setprecision(4) << qsa
		  << std::endl;
	 return best_cand;

  } else {
    // first evaluate candidates that have reliable pseudo costs
    for(BrCandVIter it = relCands_.begin(); it != relCands_.end(); ++it) { 
      getPCScore_(*it, &change_down, &change_up, &score);
      if(score > best_score) {
        best_score = score;
        if(change_up > change_down) {
          best_cand->setDir(DownBranch);
        } else {
          best_cand->setDir(UpBranch);
        }
      }
    }
    maxchange = cutoff - objval;
    // now do strong branching on unreliable candidates
    if(unrelCands_.size() > 0) {
            BrCandVIter it;
            engine_->enableStrBrSetup();
            engine_->setIterationLimit(maxIterations_); // TODO: make limit dynamic.
            cnt = 0;
            maxcnt = (node->getDepth() > maxDepth_) ? 0 : maxStrongCands_;
            for(it = unrelCands_.begin(); it != unrelCands_.end() && cnt < maxcnt;++it, ++cnt) {
          	  cand = *it;
          	  strongBranch_(cand, change_up, change_down, status_up, status_down);
          	  change_up = std::max(change_up - objval, 0.0);
          	  change_down = std::max(change_down - objval, 0.0);
          	  useStrongBranchInfo_(cand, maxchange, change_up, change_down, status_up,status_down);
          	  score = getScore_(change_up, change_down);
                    lastStrBranched_[cand->getPCostIndex()] = stats_->calls;
                    #if SPEW
                    writeScore_(cand, score, change_up, change_down);
                    #endif
                    if(status_ != NotModifiedByBrancher) {
                              break;
                    }
                    if(score > best_score) {
          		  best_score = score;
          		  best_cand = cand;
          		   if(change_up > change_down) {
          			   best_cand->setDir(DownBranch);
          		   } else {
          			   best_cand->setDir(UpBranch);
          		   }
          	  }
            }
     engine_->resetIterationLimit();
     engine_->disableStrBrSetup();
      if(NotModifiedByBrancher == status_) {
        // get score of remaining unreliable candidates as well.
        for(; it != unrelCands_.end(); ++it) {
          getPCScore_(*it, &change_down, &change_up, &score);
          if(score > best_score) {
            best_score = score;
            best_cand = *it;
            if(change_up > change_down) {
              best_cand->setDir(DownBranch);
            } else {
              best_cand->setDir(UpBranch);
            }
          }
        }
      }
    }
    //if (best_cand) {
      //#pragma omp critical
      //std::cout << "in rel: node " << node->getId() << " lb " << node->getLb()
      //<< " brCand " << best_cand->getName() << " best score " << best_score
      //<< " thread  " << omp_get_thread_num() << "\n";
    //} else {
      //std::cout << "in ql: no bestcand at node " << node->getId() << "\n";
    //}
   // std::cout <<"Reliable best_cand" << best_cand->getName() << "\n";
    root_best_cand = best_cand;
    return best_cand;
  }
}




Branches QLBrancher::findBranches(RelaxationPtr rel, NodePtr node,
                                           ConstSolutionPtr sol,
                                           SolutionPoolPtr s_pool,
                                           BrancherStatus& br_status,
                                           ModVector& mods)
{
  Branches branches = 0;
  BrCandPtr br_can = 0;
  const double* x = sol->getPrimal();
  UInt depth = node->getDepth(); 
  /*if (depth % 20 == 2) {
	 std::cout << "======= Q-Table(depth = "<< depth << ") =======" << std::endl;
	 //printQTable();
	 for (const auto& [state, actionMap] : qtable_) {
		  std::cout << "State: [ ";
		  for (double s : state) {
			  std::cout << std::fixed << std::setprecision(2) << s << " ";
		  }
		  std::cout << "]\n";
		  for (const auto& [action, q_value] : actionMap) { 
			  std::cout << "  Action: " << action->getName()                 //name
				  << " => Q-Value: " << std::fixed << std::setprecision(4) << q_value
				  << std::endl;
		  }
		  std::cout << "------------------------" << std::endl;
	  }
	  std::cout << "======= End of Q-Table =======" << std::endl;
  }
  std::cout << "State: [ ";
  for (double s : prev_state) {
	  std::cout << std::fixed << std::setprecision(2) << s << " ";
  }
  std::cout << "]\n";
  std::cout << "  Action: " << prev_best_cand->getName()
	  << " => Q-Value: " << std::fixed << std::setprecision(4) << qtable_[prev_state][prev_best_cand]
          << std::endl;
 */
 ++(stats_->calls);
  if(!init_) {
    init_ = true;
    initialize(rel);
  }
  rel_ = rel;
  br_status = NotModifiedByBrancher;
  status_ = NotModifiedByBrancher;
  mods_.clear();
  
  // make a copy of x, because it is overwritten while strong branching.
  x_.resize(rel->getNumVars());
  std::copy(x, x + rel->getNumVars(), x_.begin());

  findCandidates_();
  if(status_ == PrunedByBrancher) {
    br_status = status_;
    return 0;
  }
  if(status_ == NotModifiedByBrancher) {
		  br_can = findBestCandidate_(sol->getObjValue(),
                                s_pool->getBestSolutionValue(), node);
  }

  // status_ might have changed now. Check again.
  if(status_ == NotModifiedByBrancher) {
    // surrounded by br_can :-)
   //if (br_can && br_can->getHandler()){
    branches = br_can->getHandler()->getBranches(br_can, x_, rel_, s_pool);
    for(BranchConstIterator br_iter = branches->begin();
        br_iter != branches->end(); ++br_iter) {
      (*br_iter)->setBrCand(br_can);
    }
#if SPEW
    logger_->msgStream(LogDebug)
        << me_ << "best candidate = " << br_can->getName() << std::endl;
#endif
   // }
  } else {
    // we found some modifications that can be done to the node. Send these
    // back to the processor.
    if(mods_.size() > 0) {
      mods.insert(mods.end(), mods_.begin(), mods_.end());
    }
    br_status = status_;
#if SPEW
    logger_->msgStream(LogDebug) << me_ << "found modifications" << std::endl;
    if(mods_.size() > 0) {
      for(ModificationConstIterator miter = mods_.begin(); miter != mods_.end();
          ++miter) {
        (*miter)->write(logger_->msgStream(LogDebug));
      }
    } else if(status_ == PrunedByBrancher) {
      logger_->msgStream(LogDebug) << me_ << "Pruned." << std::endl;
    } else {
      logger_->msgStream(LogDebug)
          << me_ << "unexpected status = " << status_ << std::endl;
    }
#endif
  }

  freeCandidates_(br_can);
  if(status_ != NotModifiedByBrancher && br_can) {
    delete br_can;
  }
  return branches;
}


/*void QLBrancher::printQTable() {
        for (const auto& [state, actionMap] : qtable_) {
                  std::cout << "State: [ ";
                  for (double s : state) {
                          std::cout << std::fixed << std::setprecision(2) << s << " ";
                  }
                  std::cout << "]\n";
		  const auto& actions = actionMap.first;
                  const auto& q_value = actionMap.second;
		  for (const auto& act : actions) {
                               // BrCandPtr vc = act;
                                BrVarCandPtr vc = dynamic_cast<BrVarCand*>(act);
                                double q = act.second;
                                std::cout << "  Action: " << action                               //vc->getName()
                                << " => Q-Value: " << q << "\n";
                        }
                  for (const auto& [action, q_value] : actionMap) {
                          std::cout << "  Action: " << action->getName()                 //name
                                  << " => Q-Value: " << std::fixed << std::setprecision(4) << q_value
                                  << std::endl;
                  }
                  std::cout << "------------------------" << std::endl;
          }
          std::cout << "======= End of Q-Table =======" << std::endl;
}
*/



void QLBrancher::findCandidates_()
{
  VariableIterator v_iter, v_iter2, best_iter;
  VariableConstIterator cv_iter;
  int index;
  bool is_inf = false; // if true, then node can be pruned.

  BrVarCandSet cands;  // candidates from which to choose one.
  BrVarCandSet cands2; // Temporary set.
  BrCandVector gencands;
  BrCandVector gencands2; // Temporary vector.
  double s_wt = 1e-5;
  double i_wt = 1e-6;
  double score;

  assert(relCands_.empty());
  assert(unrelCands_.empty());

  for(HandlerIterator h = handlers_.begin(); h != handlers_.end(); ++h) {
    // ask each handler to give some candidates
    (*h)->getBranchingCandidates(rel_, x_, mods_, cands2, gencands2, is_inf);
    for(BrVarCandIter it = cands2.begin(); it != cands2.end(); ++it) {
      (*it)->setHandler(*h);
    }
    for(BrCandVIter it = gencands2.begin(); it != gencands2.end(); ++it) {
      (*it)->setHandler(*h);
    }
    cands.insert(cands2.begin(), cands2.end());
    gencands.insert(gencands.end(), gencands2.begin(), gencands2.end());
    cands2.clear();
    gencands2.clear();
    if(is_inf || mods_.size() > 0) {
      for(BrVarCandIter it = cands.begin(); it != cands.end(); ++it) {
        delete *it;
      }
      for(BrCandVIter it = gencands.begin(); it != gencands.end(); ++it) {
        delete *it;
      }
      if(is_inf) {
        status_ = PrunedByBrancher;
      } else {
 
       	      status_ = ModifiedByBrancher;
      }
      return;
    }
  }

  // visit each candidate in and check if it has reliable pseudo costs.
  for(BrVarCandIter it = cands.begin(); it != cands.end(); ++it) {
    index = (*it)->getPCostIndex();
    if((minNodeDist_ > fabs(stats_->calls - lastStrBranched_[index])) ||
       (timesUp_[index] >= thresh_ && timesDown_[index] >= thresh_)) {
      relCands_.push_back(*it);
    } else {
      score = timesUp_[index] + timesDown_[index] -
          s_wt * (pseudoUp_[index] + pseudoDown_[index]) -
          i_wt * std::max((*it)->getDDist(), (*it)->getUDist());
      (*it)->setScore(score);
      unrelCands_.push_back(*it);
    }
  }
  // push all general candidates (that are not variables) as reliable
  // candidates
  for(BrCandVIter it = gencands.begin(); it != gencands.end(); ++it) {
    relCands_.push_back(*it);
  }

  // sort unreliable candidates in the increasing order of their reliability.
  std::sort(unrelCands_.begin(), unrelCands_.end(), CompareScore);

#if SPEW
  logger_->msgStream(LogDebug)
      << me_ << "number of reliable candidates = " << relCands_.size()
      << std::endl
      << me_ << "number of unreliable candidates = " << unrelCands_.size()
      << std::endl;
  if(logger_->getMaxLevel() == LogDebug2) {
    writeScores_(logger_->msgStream(LogDebug2));
  }
#endif

  return;
}

void QLBrancher::freeCandidates_(BrCandPtr no_del)
{
  for(BrCandVIter it = unrelCands_.begin(); it != unrelCands_.end(); ++it) {
    if(no_del != *it) {
      delete *it;
    }
  }
  for(BrCandVIter it = relCands_.begin(); it != relCands_.end(); ++it) {
    if(no_del != *it) {
      delete *it;
    }
  }
  relCands_.clear();
  unrelCands_.clear();
}

bool QLBrancher::getTrustCutoff()
{
  return trustCutoff_;
}

UInt QLBrancher::getIterLim(){
  return maxIterations_;
}

std::string QLBrancher::getName() const
{
  return "QLBrancher";
}

void QLBrancher::getPCScore_(BrCandPtr cand, double* ch_down,
                                      double* ch_up, double* score)
{
  int index = cand->getPCostIndex();
  if(index > -1) {
    *ch_down = cand->getDDist() * pseudoDown_[index];
    *ch_up = cand->getUDist() * pseudoUp_[index];
    *score = getScore_(*ch_up, *ch_down);
  } else {
    *ch_down = 0.0;
    *ch_up = 0.0;
    *score = cand->getScore();
  }
}

double QLBrancher::getScore_(const double& up_score,
                                      const double& down_score)
{
  if(up_score > down_score) {
    return down_score * 0.8 + up_score * 0.2;
  } else {
    return up_score * 0.8 + down_score * 0.2;
  }
  return 0.;
}

UInt QLBrancher::getThresh() const
{
  return thresh_;
}

void QLBrancher::initialize(RelaxationPtr rel)
{
  int n = rel->getNumVars();
  // initialize to zero.
  pseudoUp_ = DoubleVector(n, 0.);
  pseudoDown_ = DoubleVector(n, 0.);
  lastStrBranched_ = UIntVector(n, 20000);
  timesUp_ = std::vector<UInt>(n, 0);
  timesDown_ = std::vector<UInt>(n, 0);

  // reserve space.
  relCands_.reserve(n);
  unrelCands_.reserve(n);
  x_.reserve(n);
}

void QLBrancher::setTrustCutoff(bool val)
{
  trustCutoff_ = val;
}

void QLBrancher::setEngine(EnginePtr engine)
{
  engine_ = engine;
}

void QLBrancher::setIterLim(UInt k)
{
  maxIterations_ = k;
}

void QLBrancher::setMaxDepth(UInt k)
{
  maxDepth_ = k;
}

void QLBrancher::setMinNodeDist(UInt k)
{
  minNodeDist_ = k;
}

void QLBrancher::setThresh(UInt k)
{
  thresh_ = k;
}

bool QLBrancher::shouldPrune_(const double& chcutoff,
                                       const double& change,
                                       const EngineStatus& status, bool* is_rel)
{
  switch(status) {
  case(ProvenLocalInfeasible):
    return true;
  case(ProvenInfeasible):
    return true;
  case(ProvenObjectiveCutOff):
    return true;
  case(ProvenLocalOptimal):
  case(ProvenOptimal):
    if(trustCutoff_ && change > chcutoff - eTol_) {
      return true;
    }
    // check feasiblity
    break;
  case(EngineUnknownStatus):
    assert(!"engine status is UnknownStatus in reliability branching!");
    break;
  case(EngineIterationLimit):
    break;
  case(ProvenFailedCQFeas):
  case(ProvenFailedCQInfeas):
    logger_->msgStream(LogInfo) << me_ << "Failed CQ."
                                << " Continuing." << std::endl;
    *is_rel = false;
    break;
  default:
    logger_->errStream() << me_ << "unexpected engine status. "
                         << "status = " << status << std::endl;
    *is_rel = false;
    stats_->engProbs += 1;
    break;
  }
  return false;
}

void QLBrancher::strongBranch_(BrCandPtr cand, double& obj_up,
                                        double& obj_down,
                                        EngineStatus& status_up,
                                        EngineStatus& status_down)
{
  HandlerPtr h = cand->getHandler();
  ModificationPtr mod;

  // first do down.
  mod = h->getBrMod(cand, x_, rel_, DownBranch);
  mod->applyToProblem(rel_);
  //std::cout << "down relax ******\n";
  //rel_->write(std::cout);

  timer_->start();
  status_down = engine_->solve();
  stats_->strTime += timer_->query();
  timer_->stop();
  ++(stats_->strBrCalls);
  obj_down = engine_->getSolutionValue();
  mod->undoToProblem(rel_);
  delete mod;

  // now go up.
  mod = h->getBrMod(cand, x_, rel_, UpBranch);
  mod->applyToProblem(rel_);
  //std::cout << "up relax ******\n";
  //rel_->write(std::cout);

  timer_->start();
  status_up = engine_->solve();
  stats_->strTime += timer_->query();
  timer_->stop();
  ++(stats_->strBrCalls);
  obj_up = engine_->getSolutionValue();
  mod->undoToProblem(rel_);
  delete mod;
}

void QLBrancher::updateAfterSolve(NodePtr node, ConstSolutionPtr sol)
{
  const double* x = sol->getPrimal();
  NodePtr parent = node->getParent();
  if(parent) {
    BrCandPtr cand = node->getBranch()->getBrCand();
    int index = cand->getPCostIndex();
    if(index > -1) {
      double oldval = node->getBranch()->getActivity();
      double newval = x[index];
      double cost =
          (node->getLb() - parent->getLb()) / (fabs(newval - oldval) + eTol_);
      if(cost < 0. || std::isinf(cost) || std::isnan(cost)) {
        cost = 0.;
      }
      if(newval < oldval) {
        updatePCost_(index, cost, pseudoDown_, timesDown_);
      } else {
        updatePCost_(index, cost, pseudoUp_, timesUp_);
      }
    }
  }
}

void QLBrancher::updatePCost_(const int& i, const double& new_cost,
                                       DoubleVector& cost, UIntVector& count)
{
  cost[i] = (cost[i] * count[i] + new_cost) / (count[i] + 1);
  count[i] += 1;
}

void QLBrancher::useStrongBranchInfo_(BrCandPtr cand,
                                               const double& chcutoff,
                                               double& change_up,
                                               double& change_down,
                                               const EngineStatus& status_up,
                                               const EngineStatus& status_down)
{
  const UInt index = cand->getPCostIndex();
  bool should_prune_up = false;
  bool should_prune_down = false;
  bool is_rel = true;
  double cost;

  should_prune_down = shouldPrune_(chcutoff, change_down, status_down, &is_rel);
  should_prune_up = shouldPrune_(chcutoff, change_up, status_up, &is_rel);

  if(!is_rel) {
    change_up = 0.;
    change_down = 0.;
  } else if(should_prune_up == true && should_prune_down == true) {
    status_ = PrunedByBrancher;
    stats_->bndChange += 2;
  } else if(should_prune_up) {
    status_ = ModifiedByBrancher;
    mods_.push_back(cand->getHandler()->getBrMod(cand, x_, rel_, DownBranch));
    ++(stats_->bndChange);
  } else if(should_prune_down) {
    status_ = ModifiedByBrancher;
    mods_.push_back(cand->getHandler()->getBrMod(cand, x_, rel_, UpBranch));
    ++(stats_->bndChange);
  } else {
    cost = fabs(change_down) / (fabs(cand->getDDist()) + eTol_);
    updatePCost_(index, cost, pseudoDown_, timesDown_);

    cost = fabs(change_up) / (fabs(cand->getUDist()) + eTol_);
    updatePCost_(index, cost, pseudoUp_, timesUp_);
  }
}

void QLBrancher::writeScore_(BrCandPtr cand, double score,
                                      double change_up, double change_down)
{
  logger_->msgStream(LogDebug2)
      << me_ << "candidate: " << cand->getName()
      << " down change = " << change_down << " up change = " << change_up
      << " score = " << score << std::endl;
}

void QLBrancher::writeScores_(std::ostream& out)
{
  out << me_ << "unreliable candidates:" << std::endl;
  for(BrCandVIter it = unrelCands_.begin(); it != unrelCands_.end(); ++it) {
    if((*it)->getPCostIndex() > -1) {
      out << std::setprecision(6) << (*it)->getName() << "\t"
          << timesDown_[(*it)->getPCostIndex()] << "\t"
          << timesUp_[(*it)->getPCostIndex()] << "\t"
          << pseudoDown_[(*it)->getPCostIndex()] << "\t"
          << pseudoUp_[(*it)->getPCostIndex()] << "\t"
          << x_[(*it)->getPCostIndex()] << "\t"
          << rel_->getVariable((*it)->getPCostIndex())->getLb() << "\t"
          << rel_->getVariable((*it)->getPCostIndex())->getUb() << "\t"
          << std::endl;
    } else {
      out << std::setprecision(6) << (*it)->getName() << "\t" << 0 << "\t" << 0
          << "\t" << (*it)->getScore() << "\t" << (*it)->getScore() << "\t"
          << (*it)->getDDist() << "\t" << 0.0 << "\t" << 1.0 << "\t"
          << std::endl;
    }
  }

  out << me_ << "reliable candidates:" << std::endl;
  for(BrCandVIter it = relCands_.begin(); it != relCands_.end(); ++it) {
    if((*it)->getPCostIndex() > -1) {
      out << (*it)->getName() << "\t" << timesDown_[(*it)->getPCostIndex()]
          << "\t" << timesUp_[(*it)->getPCostIndex()] << "\t"
          << pseudoDown_[(*it)->getPCostIndex()] << "\t"
          << pseudoUp_[(*it)->getPCostIndex()] << "\t"
          << x_[(*it)->getPCostIndex()] << "\t"
          << rel_->getVariable((*it)->getPCostIndex())->getLb() << "\t"
          << rel_->getVariable((*it)->getPCostIndex())->getUb() << "\t"
          << std::endl;
    } else {
      out << std::setprecision(6) << (*it)->getName() << "\t" << 0 << "\t" << 0
          << "\t" << (*it)->getScore() << "\t" << (*it)->getScore() << "\t"
          << (*it)->getDDist() << "\t" << 0.0 << "\t" << 1.0 << "\t"
          << std::endl;
    }
  }
}

void QLBrancher::writeStats(std::ostream& out) const
{
  if(stats_) {
    out << me_ << "times called                = " << stats_->calls << std::endl
        << me_ << "no. of problems in engine   = " << stats_->engProbs
        << std::endl
        << me_ << "times relaxation solved     = " << stats_->strBrCalls
        << std::endl
        << me_ << "times bounds changed        = " << stats_->bndChange
        << std::endl
        << me_ << "time in solving relaxations = " << stats_->strTime
        << std::endl;
  }
}

