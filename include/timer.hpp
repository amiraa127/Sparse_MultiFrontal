#ifndef SMF_TIMER_H
#define SMF_TIMER_H

#include <string>
#include <map>
#include <ctime>

#ifndef ENABLE_TIMER
  #ifndef NDEBUG
    #define ENABLE_TIMER 0
  #else
    #define ENABLE_TIMER 1
  #endif
#endif


namespace smf 
{
  
  /// Helper class to distinguish between different time measurement methods
  class TimerMap : public std::map<std::string, double>
  {
  public:
    static void store(std::string key, double value) 
    { 
      initIntern();
      (*singlett)[key] += value;
    }
    
    static double get(std::string key)
    {
      return (*singlett)[key];
    }

  private:
    TimerMap() {}
    
    static void initIntern() 
    {
      if (singlett == NULL)
	singlett = new TimerMap;
    }
    
    /// pointer to the singleton that contains the data
    static TimerMap* singlett;
  };
  
  
  class Timer
  {
  public:
    /// start timer
    Timer(std::string key_, bool autostart = true)
	: key(key_),
	  startTime(0.0),
	  started(false),
	  intermediate(0.0)
    { 
      if (autostart)
	start();
    }
    
    /// destructor stores time measurement in TimerMap
    ~Timer()
    {
#if ENABLE_TIMER
      TimerMap::store(key, elapsed());
#endif
    }

    inline void stop()
    {
#if ENABLE_TIMER
      if (started)
	intermediate += (clock() - startTime)/CLOCKS_PER_SEC;
      started = false;
#endif
    }
    
    /// resets the timer to current time
    inline void start()
    {
#if ENABLE_TIMER
      startTime = clock();
      started = true;
#endif
    }

    /// returns the elapsed time (from construction or last reset) to now in seconds
    inline double elapsed() const
    {
#if ENABLE_TIMER
      return intermediate + (started ? (clock() - startTime)/CLOCKS_PER_SEC : 0.0);
#else
      return 0.0;
#endif
    }
    
  private:
    // key value to store measurement in TimerMap
    std::string key;
    
    /// begin value for measurement
    double startTime;
    
    /// indicates whether timer is started or not
    bool started;
    
    /// store intermediate measurements
    double intermediate;
  };
  
} // end namespace timer

#endif // SMF_TIMER_H