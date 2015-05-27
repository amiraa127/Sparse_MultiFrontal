#ifndef SMF_TIMER_H
#define SMF_TIMER_H

#include <string>
#include <map>
#include <ctime>

/** \addtogroup timer 
 *  @{
 */

/** 
 * \def SMF_ENABLE_TIMER
 * \brief The preprocessor constant enables the timer functionality
 * 
 * If the value is set to 1 the class \ref smf::Timer provides implemented
 * methods for time measurements, otherwise the class is empty.
 **/
#ifndef SMF_ENABLE_TIMER
  #ifndef NDEBUG
    #define SMF_ENABLE_TIMER 1
  #else
    #define SMF_ENABLE_TIMER 0
  #endif
#endif

/** @}*/

namespace smf 
{
  
  /** \addtogroup timer 
  *  @{
  */
  
  /// \brief Static storage for time measurements
  /**
   * This class allows to store measurements based on a (key, value) pair
   * in a static map. It provides two methods: \ref store and \ref get
   * to access the map.
   * 
   * Example:
   * ~~~~~~~~~~~~~~~~~~~
   * TimerMap::store("foo", 0.1);
   * TimerMap::get("foo");         // => 0.1
   * ~~~~~~~~~~~~~~~~~~~
   **/
  class TimerMap : public std::map<std::string, double>
  {
  public:
    /// \brief call this method to store a measurement \p value for the given 
    /// \p key in the map.
    static void store(std::string key, double value) 
    { 
      initIntern();
      (*singlett)[key] += value;
    }
    
    /// Return the value stored for a given \p key
    static double get(std::string key)
    {
      initIntern();
      return (*singlett)[key];
    }
    
    /// Delete all stored values in the map
    static void reset()
    {
      initIntern();
      singlett->clear();
    }

  private:
    // hidden constructor
    TimerMap() {}
    
    // must be called before access to the singlett.
    static void initIntern() 
    {
      if (singlett == NULL)
	singlett = new TimerMap;
    }
    
    // pointer to the singleton that contains the data
    static TimerMap* singlett;
  };
  
  
  /// Interface for \ref TimerMap::get 
  inline double get_time(std::string key)
  {
    return TimerMap::get(key);
  }
  
  
#if SMF_ENABLE_TIMER
  /// \brief Class to enable timing facilities.
  /**
   * Constructs a timer object that measures the time during the livetime
   * of the object. When the destructor is called, the measured time is
   * written into the \ref TimerMap static object, using the given key in
   * the constructor.
   * 
   * Example:
   * ~~~~~~~~~~~~~~~~~~~~
   * void foo() {
   *   Timer t("foo");
   *   // do something time consuming...
   * }
   * ~~~~~~~~~~~~~~~~~~~~
   * At the end of the scope the timer adds an entry "foo" into the 
   * \ref TimerMap with the value of the measured time in the function foo.
   * 
   * With \ref start() and \ref stop() you can control the timer, i.e. break
   * the timer to measure only in a part of the function scope. With 
   * \ref elapsed() the already mesured time can be returned.
   * 
   * This class has an implementation only if the preprocessor constant 
   * \ref SMF_ENABLE_TIMER is set to 1.
   **/
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
      TimerMap::store(key, elapsed());
    }
    

    /// stop the timer an store the measured time in the variable intermediate.
    inline void stop()
    {
      if (started)
	intermediate += (clock() - startTime)/CLOCKS_PER_SEC;
      started = false;
    }
    
    
    /// resets the timer to current time.
    inline void start()
    {
      startTime = clock();
      started = true;
    }
    

    /// returns the elapsed time (from construction or last reset) to now in seconds
    inline double elapsed() const
    {
      return intermediate + (started ? (clock() - startTime)/CLOCKS_PER_SEC : 0.0);
    }
    
  private:
    std::string key;		///< key value to store measurement in TimerMap
    
    double startTime;		///< begin value for measurement
    bool   started;		///< indicates whether timer is started or not
    double intermediate;	///< store intermediate measurements
  };
#else
  /// \cond HIDDEN_SYMBOLS
  // in release mode the timer object does nothing
  struct Timer
  {
    Timer(std::string, bool = true) {}
    inline void stop() const {}
    inline void start() const {}
    inline double elapsed() const { return 0.0; }
  };
  /// \endcond
#endif
  
  /** @}*/
  
} // end namespace timer

#endif // SMF_TIMER_H