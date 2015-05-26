#ifndef SMF_OUTPUT_H
#define SMF_OUTPUT_H

#define SMF_NO_THROW

#include <iostream>
#include <string>
#include <sstream>
#include <map>
#include <ctime>

#ifdef SMF_NO_THROW
  #include <cassert>
#else
  #include <stdexcept>
#endif

#ifndef ENABLE_MSG_DBG
  #ifndef NDEBUG
    #define ENABLE_MSG_DBG 0
  #else
    #define ENABLE_MSG_DBG 1
  #endif
#endif

namespace smf 
{
  
  namespace aux
  {
    template <typename OStream>
    OStream& concat(OStream& out) { return out; }
    
    template <typename OStream, typename Arg0, typename... Args>
    OStream& concat(OStream& out, Arg0&& arg0, Args&&... args)
    {
      out << arg0; concat(out, std::forward<Args>(args)...);
      return out;
    }
    
    template <typename... Args>
    std::string to_string(Args&&... args)
    {
      std::stringstream ss; concat(ss, std::forward<Args>(args)...);
      return ss.str();
    }
  }
    
  /// print a message
  template <typename... Args>
  void msg(Args&&... args)
  {
    aux::concat(std::cout, std::forward<Args>(args)..., "\n");
  }

  /// print a message (without appended newline)
  template <typename... Args>
  void msg_(Args&&... args)
  {
    aux::concat(std::cout, std::forward<Args>(args)...);
  }
  
  /// print a message and exit
  template <typename... Args>
  void error_exit(Args&&... args)
  {
#ifdef SMF_NO_THROW
    msg("ERROR: ", std::forward<Args>(args)...); 
    assert( false ); //exit(EXIT_FAILURE);
#else
    throw std::runtime_error( aux::to_string(std::forward<Args>(args)...) );
#endif
  }
  
  /// test for condition and in case of failure print message and exit
  template <typename... Args>
  void assert_msg(bool condition, Args&&... args)
  {
    if (!condition) { error_exit(std::forward<Args>(args)...); }
  }
  
  /// test for condition and in case of failure print message
  template <typename... Args>
  void warn_msg(bool condition, Args&&... args)
  {
    if (!condition) { msg("WARNING: ", std::forward<Args>(args)...); }
  }
  
#if ENABLE_MSG_DBG
  /// print message, in debug mode only
  template <typename... Args>
  void msg_dbg(Args&&... args) { msg(std::forward<Args>(args)...); }
  
  /// call TEST_EXIT, in debug mode only
  template <typename... Args>
  void assert_msg_dbg(bool condition, Args&&... args) 
  { 
    assert_msg_dbg(condition, std::forward<Args>(args)...); 
  }
#else
  template <typename... Args>
  void msg_dbg(Args&&...) {}
  
  template <typename... Args>
  void assert_msg_dbg(bool, Args&&...) {}
#endif

}


#endif // SMF_OUTPUT_H
