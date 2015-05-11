#ifndef __LOGGER_LOGGER_HPP__
#define __LOGGER_LOGGER_HPP__

#include <boost/date_time/posix_time/posix_time.hpp>
#include <fstream>
#include <functional>
#include <map>
#include <string>
#include <ctime>

namespace 
{
    std::string now_str()
    {
        // Get current time from the clock, using microseconds resolution
        const boost::posix_time::ptime now = boost::posix_time::microsec_clock::local_time();

        // Get the time offset in current day
        const boost::posix_time::time_duration td = now.time_of_day();

        // Extract hours, minutes, seconds and milliseconds.
        // Since there is no direct accessor ".milliseconds()",
        // milliseconds are computed _by difference_ between total milliseconds
        // (for which there is an accessor), and the hours/minutes/seconds
        // values previously fetched.
        //
        const long hours        = td.hours();
        const long minutes      = td.minutes();
        const long seconds      = td.seconds();
        const long milliseconds = td.total_milliseconds() -
                                  ((hours * 3600 + minutes * 60 + seconds) * 1000);

        return std::to_string(hours)   + ":" +
               std::to_string(minutes) + ":" +
               std::to_string(seconds) + "." +
               std::to_string(milliseconds); 
    }
}

namespace logger
{
    enum LogLevels
    {
        ERROR  ,
        WARNING,
        INFO   ,
        DEBUG  
    };
    
    class Logger;
    
    class ConcreteLogger
    {
      friend class Logger;

      public:
        void error  (const std::string& msg, bool only_msg = false) { if (m_log_level >= ERROR  ) log(msg, "error"  , only_msg); }
        void warning(const std::string& msg, bool only_msg = false) { if (m_log_level >= WARNING) log(msg, "warning", only_msg); }
        void info   (const std::string& msg, bool only_msg = false) { if (m_log_level >= INFO   ) log(msg, "info"   , only_msg); }
        void debug  (const std::string& msg, bool only_msg = false) { if (m_log_level >= DEBUG  ) log(msg, "debug"  , only_msg); }
      
      private:
        void log(const std::string& msg, const std::string& level, bool only_msg)
        {
            if (!only_msg)
                m_of << now_str() << " [" << level << "] " << m_log_name << ": " << msg << std::endl;
            else
                m_of << msg;
        }
        
        ConcreteLogger(const std::string& log_name, const std::string& filename, LogLevels log_level = LogLevels::INFO)
        : m_log_name(log_name)
        , m_filename(filename)
        , m_log_level(log_level)
        {
            m_of.open(m_filename, std::ios::app);
        }

        ~ConcreteLogger()
        {
            m_of.close();
        }
          
        ConcreteLogger(const ConcreteLogger& log) = delete;
        ConcreteLogger& operator=(const ConcreteLogger& log) = delete;

        std::ofstream m_of;
        std::string m_log_name;
        std::string m_filename;  
        LogLevels m_log_level;
    };
    
    class Logger
    {
      public:
        static ConcreteLogger* getLog(const std::string& log_name, const std::string& filename, LogLevels log_level = LogLevels::INFO)
        {
            static Logger instance;
            return instance.getConcreteLogger(log_name, filename, log_level);
        }
        
        static ConcreteLogger* getLog(const std::string& log_name, LogLevels log_level = LogLevels::INFO)
        {
            return getLog(log_name, log_name + ".log", log_level);
        }

      private:
        Logger()
        { }
        
        ConcreteLogger* getConcreteLogger(const std::string& log_name, const std::string& filename, LogLevels log_level = LogLevels::INFO)
        {
            //insert if element with this log_name doesn't exist
            if (m_loggers.count(log_name) == 0)
            {
                ConcreteLogger* concr_log = new ConcreteLogger(log_name, filename, log_level);
                m_loggers.emplace(log_name, concr_log);
            }
            
            return m_loggers[log_name];    
        }

        ~Logger()
        {
            for (auto& it: m_loggers)
                delete it.second;
        }

        Logger(const Logger& log) = delete;
        Logger& operator=(const Logger& log) = delete;    

        std::map<std::string, ConcreteLogger*> m_loggers;
    };
}

#endif //__LOGGER_LOGGER_HPP__
