// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-12-20 19:01
*
* Description: GraceQ/tensor project. A timer class for timing.
*/

/**
@file timer.h
@brief A timer class for timing.
*/
#ifndef GQTEN_UTILITY_TIMER_H
#define GQTEN_UTILITY_TIMER_H


#include <iostream>     // cout, endl
#include <string>       // string
#include <iomanip>      // setprecision

#include <time.h>
#include <sys/time.h>

#define STATUS bool
#define SLEEPING false
#define RUNNING true

namespace gqten {


/**
Timer.
*/
class Timer {
public:
  /**
  Create a timer with a note.

  @param notes Notes for the timer.
  */
  Timer(const std::string &notes) :
      start_(GetWallTime_()),
      pass_(0.0),
      status_(RUNNING),
      notes_(notes) {}

  Timer(void) : Timer("") {}

  void Suspend(void) { 
    assert(status_==RUNNING);
    pass_ += GetWallTime_() - start_;
    status_ = SLEEPING;
  }
  /// Restart the timer.
  void Restart(void) {
    assert(status_==SLEEPING);
    start_ = GetWallTime_(); 
    status_ = RUNNING;
  }
  void ClearAndRestart(void){
    pass_ = 0.0;
    start_ = GetWallTime_(); 
    status_ = RUNNING;
  }

  /// Return elapsed time (seconds).
  double Elapsed(void) { 
    if(status_==RUNNING){
      return GetWallTime_() - start_ + pass_; 
    }else{
      return pass_;
    }
  }

  /**
  Print elapsed time with the notes.

  @param precision Output precision.
  */
  double PrintElapsed(std::size_t precision = 5) {
    auto elapsed_time = Elapsed();
    std::cout << "[timing]";
    if (notes_ == "") {
      std::cout << " ";
    } else {
      std::cout << " " << notes_ << " ";
    }
    std::cout << std::setw(precision+3) << std::setprecision(precision)
              << std::fixed << elapsed_time
              << std::endl;
    return elapsed_time;
  }

private:
  double start_;
  double pass_;
  STATUS status_; 
  std::string notes_;

  double GetWallTime_(void) {
    struct timeval time;
    if (gettimeofday(&time, NULL)) { return 0; }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
  }
};
} /* gqten */ 
#endif /* ifndef GQTEN_UTILITY_TIMER_H */
