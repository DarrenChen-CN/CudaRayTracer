#include "stringutil.h"

std::string GetDirectoryFromPath(std::string filepath) {
    // Find last occurrence of the directory separator
    // This works for both Unix (/) and Windows (\) paths
    size_t last_slash_pos = filepath.find_last_of("/");

    // If a slash is found, return the substring up to and including the last slash
    if (last_slash_pos != std::string::npos) {
        return filepath.substr(0, last_slash_pos + 1);
    }

    // If no slash is found, return an empty string
    return "";
}

std::string GetFilenameFromPath(std::string filepath){
    size_t last_slash_pos = filepath.find_last_of("/");
    
    std::string filename;
    if (last_slash_pos != std::string::npos) {
        filename = filepath.substr(last_slash_pos + 1);
    } else {
        filename = filepath;
    }
    
    size_t last_dot_pos = filename.find_last_of('.');
    
    if (last_dot_pos != std::string::npos && last_dot_pos > 0) {
        return filename.substr(0, last_dot_pos);
    }
    
    return filename;
}