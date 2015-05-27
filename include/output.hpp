#ifndef SMF_OUTPUT_H
#define SMF_OUTPUT_H

#define SMF_NO_THROW

#include <iostream>
#include <string>
#include <sstream>
#include <map>
#include <ctime>

/** \addtogroup output 
 *  @{
 */

/** 
 * \def SMF_NO_THROW
 * \brief The preprocessor constant sets whether to use c-asserts (if defined) or 
 * to throw an exception in case of an error (if not defined). 
 **/
#ifdef SMF_NO_THROW
  #include <cassert>
#else
  #include <stdexcept>
#endif

/** 
 * \def SMF_ENABLE_MSG_DBG
 * \brief The preprocessor constant enables the functions \ref smf::msg_dbg
 * and \ref smf::assert_msg_dbg. 
 * 
 * If the value is set to 1 the functions \ref smf::msg_dbg and \ref smf::assert_msg_dbg 
 * are implemented, otherwise empty. Default is value 0 if \ref NDEBUG is not
 * defined, otherwise value 1.
 **/
#ifndef SMF_ENABLE_MSG_DBG
  #ifndef NDEBUG
    #define SMF_ENABLE_MSG_DBG 1
  #else
    #define SMF_ENABLE_MSG_DBG 0
  #endif
#endif

/** @}*/

namespace smf 
{
  
  /// \cond HIDDEN_SYMBOLS
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
  /// \endcond
    
  /** \addtogroup output 
  *  @{
  */
    
  /// \brief print a message
  /**
   * Example:
   * ~~~~~~~~~~~~~~~~~~
   * msg("Hello", " ", "World: ", 123); // prints "Hello World: 123\n"
   * ~~~~~~~~~~~~~~~~~~
   **/
  template <typename... Args>
  void msg(Args&&... args)
  {
    aux::concat(std::cout, std::forward<Args>(args)..., "\n");
  }
  

  /// \brief print a message (without appended newline)
  /**
   * Example:
   * ~~~~~~~~~~~~~~~~~~
   * msg("Hello", " ", "World: ", 123); // prints "Hello World: 123"
   * ~~~~~~~~~~~~~~~~~~
   **/
  template <typename... Args>
  void msg_(Args&&... args)
  {
    aux::concat(std::cout, std::forward<Args>(args)...);
  }
  
  
  /// \brief print a message and exit
  /**
   * If the preprocessor constant \ref SMF_NO_THROW is defined,
   * the c-assert macro is called, otherwise an exception of
   * type \ref std::runtime_Error is thrown.
   **/
  template <typename... Args>
  void error_exit(Args&&... args)
  {
#ifdef SMF_NO_THROW
    aux::concat(std::cerr, "ERROR: ", std::forward<Args>(args)..., "\n");
    #ifndef NDEBUG
      assert(false);
    #else
      std::exit(EXIT_FAILURE);
    #endif
#else
    throw std::runtime_error( aux::to_string("ERROR: ", std::forward<Args>(args)...) );
#endif
  }
  
  
  /// \brief test for condition and in case of failure print message and exit
  /**
   * This function is equivalent to
   * ~~~~~~~~~~~~~~~~~~~~~~~~
   * if (condition == false) error_exit(text);
   * ~~~~~~~~~~~~~~~~~~~~~~~~
   * where `text` correspond to the arguments passed after the 
   * \p condition argument.
   **/
  template <typename... Args>
  void assert_msg(bool condition, Args&&... args)
  {
    if (!condition) { error_exit(std::forward<Args>(args)...); }
  }
  
  
  /// \brief test for condition and in case of failure print message
  /**
   * Same as \ref assert_msg but does not throw an exception, or call assert.
   * It just tests for the condition and prints a message with prepended
   * string "WARNING".
   **/
  template <typename... Args>
  void warn_msg(bool condition, Args&&... args)
  {
    if (!condition) { msg("WARNING: ", std::forward<Args>(args)...); }
  }
  
  
#if SMF_ENABLE_MSG_DBG
  /// \brief print message, in debug mode only
  /**
   * Same as \ref msg, but is available only if preprocessor constant
   * \ref SMF_ENABLE_MSG_DBG is set to 1, otherwise the function is empty.
   **/
  template <typename... Args>
  void msg_dbg(Args&&... args) { msg(std::forward<Args>(args)...); }
  
  
  /// \brief call assert_msg, in debug mode only
  /**
   * Same as \ref assert_msg, but is available only if preprocessor constant
   * \ref SMF_ENABLE_MSG_DBG is set to 1, otherwise the function is empty.
   **/
  template <typename... Args>
  void assert_msg_dbg(bool condition, Args&&... args) 
  { 
    assert_msg(condition, std::forward<Args>(args)...); 
  }
#else
  /// \cond HIDDEN_SYMBOLS
  template <typename... Args>
  void msg_dbg(Args&&...) {}
  
  template <typename... Args>
  void assert_msg_dbg(bool, Args&&...) {}
  /// \endcond
#endif

  /** @}*/
}


#endif // SMF_OUTPUT_H
